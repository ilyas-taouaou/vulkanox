use std::sync::Arc;

use anyhow::Result;
use gltf::camera::Projection;
use nalgebra::{Isometry3, OMatrix, Perspective3, Point3, Vector3};
use palette::angle::RealAngle;
use vulkano::buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
use vulkano::buffer::{BufferContents, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::{
    StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocatorCreateInfo;
use vulkano::descriptor_set::{
    allocator::StandardDescriptorSetAllocator, DescriptorSet, PersistentDescriptorSet,
    WriteDescriptorSet,
};
use vulkano::device::{Device, DeviceCreateInfo, Features, Queue, QueueCreateInfo};
use vulkano::format::Format;
use vulkano::image::SampleCount;
use vulkano::memory::allocator::{MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::depth_stencil::{DepthState, DepthStencilState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::{CullMode, RasterizationState};
use vulkano::pipeline::graphics::subpass::PipelineRenderingCreateInfo;
use vulkano::pipeline::graphics::vertex_input::{Vertex as VertexInputVertex, VertexDefinition};
use vulkano::pipeline::graphics::viewport::ViewportState;
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    DynamicState, GraphicsPipeline, Pipeline, PipelineLayout, PipelineShaderStageCreateInfo,
};
use vulkano::sync::GpuFuture;
use vulkano::{sync, DeviceSize};

use crate::vulkan_instance::VulkanInstance;

pub struct VulkanDevice {
    queue: Arc<Queue>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_allocator: Arc<StandardCommandBufferAllocator>,
    graphics_pipeline: Arc<GraphicsPipeline>,
    vertex_buffer: Subbuffer<[Vertex]>,
    index_buffer: Subbuffer<[u16]>,
    samples: SampleCount,
    set: Arc<PersistentDescriptorSet>,
}

pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
                #version 460

                layout(location = 0) in vec3 position;
                
                layout(location = 0) out vec3 fragColor;
                
                layout(set = 0, binding = 0) uniform Data {
                    mat4 view_projection;
                } uniforms;
                
                layout(push_constant) uniform PushConstantData {
                    float time;
                    vec2 mousePosition;
                } pc;

                void main() {
                    gl_Position = uniforms.view_projection * vec4(position, 1.0);
                    fragColor = position;
                }
            ",
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
                    #version 460

                    layout(location = 0) in vec3 fragColor;

                    layout(location = 0) out vec4 outColor;

                    void main() {
                        outColor = vec4(fragColor, 1.0);
                    }
            ",
    }
}

#[derive(VertexInputVertex, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
#[repr(C)]
pub struct Vertex {
    #[format(R32G32B32_SFLOAT)]
    position: [f32; 3],
}

fn align_usize(number: usize, alignment: usize) -> usize {
    ((number as f64 / alignment as f64).ceil()) as usize * alignment
}

impl VulkanDevice {
    pub(crate) fn new(instance: Arc<VulkanInstance>, samples: SampleCount) -> Result<Self> {
        let physical_device = instance.physical_device();
        let queue_family_index = instance.queue_family_index();
        let device_extensions = instance.device_extensions();

        let (device, mut queues) = Device::new(
            Arc::clone(physical_device),
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                enabled_extensions: *device_extensions,
                enabled_features: Features {
                    dynamic_rendering: true,
                    ..Features::empty()
                },
                ..Default::default()
            },
        )?;

        let queue = queues.next().unwrap();

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(Arc::clone(&device)));

        let command_allocator = Arc::new(StandardCommandBufferAllocator::new(
            Arc::clone(&device),
            StandardCommandBufferAllocatorCreateInfo::default(),
        ));

        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            Arc::clone(&device),
            StandardDescriptorSetAllocatorCreateInfo::default(),
        ));

        let (document, buffers, images) = gltf::import("assets/cube.gltf")?;

        let buffer = buffers.into_iter().next().unwrap().0;
        let mut views = document.views();
        let vertex_buffer_view = views.next().unwrap();
        let index_buffer_view = views.next().unwrap();
        let vertices =
            bytemuck::cast_slice(&buffer[vertex_buffer_view.offset()..vertex_buffer_view.length()]);
        let indices = bytemuck::cast_slice(
            &buffer[index_buffer_view.offset()
                ..index_buffer_view.offset() + index_buffer_view.length()],
        );

        let max_initial_data_size = align_usize(
            std::mem::size_of_val(&vertices) + std::mem::size_of_val(&indices),
            256,
        );

        let cameraNode = document.nodes().next().unwrap();

        let camera_projection = match cameraNode.camera().unwrap().projection() {
            Projection::Perspective(perspective) => Perspective3::new(
                800.0 / 600.0,
                f32::degrees_to_radians(70.0),
                perspective.znear(),
                perspective.zfar().unwrap(),
            ),
            _ => unimplemented!(),
        };
        // let camera_isometry = match cameraNode.transform() {
        //     gltf::scene::Transform::Decomposed {
        //         translation,
        //         rotation,
        //         ..
        //     } => Isometry3::from_parts(
        //         Translation3::new(translation[0], translation[1], translation[2]),
        //         UnitQuaternion::new_normalize(Quaternion::new(
        //             rotation[3],
        //             rotation[0],
        //             rotation[1],
        //             rotation[2],
        //         )),
        //     ),
        //     _ => unimplemented!(),
        // };

        let eye = Point3::new(2.0, -2.0, 2.0);
        let target = Point3::new(0.0, 0.0, 0.0);
        let camera_view = Isometry3::look_at_rh(&eye, &target, &Vector3::y());
        let view_projection = camera_projection.into_inner() * camera_view.to_homogeneous();

        let device_buffer_allocator = SubbufferAllocator::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                arena_size: max_initial_data_size as DeviceSize,
                buffer_usage: BufferUsage::TRANSFER_DST
                    | BufferUsage::VERTEX_BUFFER
                    | BufferUsage::INDEX_BUFFER
                    | BufferUsage::UNIFORM_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        );

        let host_buffer_allocator = SubbufferAllocator::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                arena_size: max_initial_data_size as DeviceSize,
                buffer_usage: BufferUsage::TRANSFER_SRC,
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );

        use nalgebra as na;

        #[derive(BufferContents)]
        #[repr(C)]
        struct Uniform {
            view_projection: OMatrix<f32, na::U4, na::U4>,
        }

        let uniform = Uniform { view_projection };

        let vertex_buffer = device_buffer_allocator.allocate_slice(vertices.len() as DeviceSize)?;
        let index_buffer = device_buffer_allocator.allocate_slice(indices.len() as DeviceSize)?;
        let uniform_buffer = device_buffer_allocator.allocate_sized::<Uniform>()?;

        let vertex_staging_buffer =
            host_buffer_allocator.allocate_slice::<Vertex>(vertices.len() as DeviceSize)?;
        let index_staging_buffer =
            host_buffer_allocator.allocate_slice::<u16>(indices.len() as DeviceSize)?;
        let uniform_staging_buffer = host_buffer_allocator.allocate_sized::<Uniform>()?;

        {
            let mut vertex_writer = vertex_staging_buffer.write()?;
            vertex_writer.copy_from_slice(&vertices);
            let mut indices_writer = index_staging_buffer.write()?;
            indices_writer.copy_from_slice(&indices);
            let mut uniform_writer = uniform_staging_buffer.write()?;
            *uniform_writer = uniform;
        }

        let mut command_builder = AutoCommandBufferBuilder::primary(
            &command_allocator,
            queue_family_index,
            CommandBufferUsage::OneTimeSubmit,
        )?;

        command_builder.copy_buffer(CopyBufferInfo::buffers(
            vertex_staging_buffer,
            vertex_buffer.clone(),
        ))?;
        command_builder.copy_buffer(CopyBufferInfo::buffers(
            index_staging_buffer,
            index_buffer.clone(),
        ))?;
        command_builder.copy_buffer(CopyBufferInfo::buffers(
            uniform_staging_buffer,
            uniform_buffer.clone(),
        ))?;

        let command_buffer = command_builder.build()?;

        let buffers_upload_future = sync::now(Arc::clone(&device))
            .then_execute(Arc::clone(&queue), command_buffer)?
            .then_signal_fence_and_flush()?;

        let graphics_pipeline = {
            let vertex_shader = vs::load(Arc::clone(&device))?.entry_point("main").unwrap();
            let fragment_shader = fs::load(Arc::clone(&device))?.entry_point("main").unwrap();

            let vertex_input_state = Vertex::per_vertex()
                .definition(&vertex_shader.info().input_interface)
                .unwrap();

            let stages = [
                PipelineShaderStageCreateInfo::new(vertex_shader),
                PipelineShaderStageCreateInfo::new(fragment_shader),
            ];

            let layout = PipelineLayout::new(
                Arc::clone(&device),
                PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap(),
            )?;

            let subpass = PipelineRenderingCreateInfo {
                color_attachment_formats: vec![Some(Format::B8G8R8A8_SRGB)],
                depth_attachment_format: Some(Format::D16_UNORM),
                ..Default::default()
            };

            GraphicsPipeline::new(
                device,
                None,
                GraphicsPipelineCreateInfo {
                    stages: stages.into_iter().collect(),
                    input_assembly_state: Some(InputAssemblyState::default()),
                    vertex_input_state: Some(vertex_input_state),
                    viewport_state: Some(ViewportState::default()),
                    rasterization_state: Some(RasterizationState {
                        cull_mode: CullMode::None,
                        ..Default::default()
                    }),
                    depth_stencil_state: Some(DepthStencilState {
                        depth: Some(DepthState::simple()),
                        ..Default::default()
                    }),
                    multisample_state: Some(MultisampleState {
                        rasterization_samples: samples,
                        ..Default::default()
                    }),
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        subpass.color_attachment_formats.len() as u32,
                        ColorBlendAttachmentState::default(),
                    )),
                    dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                    subpass: Some(subpass.into()),
                    ..GraphicsPipelineCreateInfo::layout(layout)
                },
            )
        }?;

        let set = PersistentDescriptorSet::new(
            &descriptor_set_allocator,
            Arc::clone(graphics_pipeline.layout().set_layouts().get(0).unwrap()),
            [WriteDescriptorSet::buffer(0, uniform_buffer)],
            [],
        )?;

        buffers_upload_future.wait(None)?;

        Ok(Self {
            queue,
            memory_allocator,
            command_allocator,
            graphics_pipeline,
            vertex_buffer,
            index_buffer,
            samples,
            set,
        })
    }

    pub fn queue(&self) -> &Arc<Queue> {
        &self.queue
    }

    pub fn memory_allocator(&self) -> &Arc<StandardMemoryAllocator> {
        &self.memory_allocator
    }

    pub fn command_allocator(&self) -> &Arc<StandardCommandBufferAllocator> {
        &self.command_allocator
    }

    pub fn graphics_pipeline(&self) -> &Arc<GraphicsPipeline> {
        &self.graphics_pipeline
    }

    pub fn vertex_buffer(&self) -> &Subbuffer<[Vertex]> {
        &self.vertex_buffer
    }

    pub fn index_buffer(&self) -> &Subbuffer<[u16]> {
        &self.index_buffer
    }

    pub fn samples(&self) -> SampleCount {
        self.samples
    }

    pub fn set(&self) -> &Arc<PersistentDescriptorSet> {
        &self.set
    }
}

use std::sync::Arc;

use anyhow::Result;
use vulkano::buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::{
    StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo};
use vulkano::device::{Device, DeviceCreateInfo, Features, Queue, QueueCreateInfo};
use vulkano::format::Format;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::memory::MemoryPropertyFlags;
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::{CullMode, RasterizationState};
use vulkano::pipeline::graphics::subpass::PipelineRenderingCreateInfo;
use vulkano::pipeline::graphics::vertex_input::{Vertex as VertexInputVertex, VertexDefinition};
use vulkano::pipeline::graphics::viewport::ViewportState;
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    DynamicState, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo,
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
}

pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
                #version 460

                layout(location = 0) in vec3 position;
                layout(location = 1) in vec3 color;
                
                layout(location = 0) out vec3 fragColor;
                
                layout(push_constant) uniform PushConstantData {
                    float time;
                    vec2 mousePosition;
                } pc;

                void main() {
                    gl_Position = vec4(position + vec3(pc.mousePosition.x, pc.mousePosition.y,0), 1.0);
                    fragColor = color * vec3(sin(pc.time*5.0)*0.5+0.5);
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
    #[format(R32G32B32_SFLOAT)]
    color: [f32; 3],
}

impl VulkanDevice {
    pub(crate) fn new(instance: Arc<VulkanInstance>) -> Result<Self> {
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

        let vertices = [
            Vertex {
                position: [-0.5, -0.25, 0.0],
                color: [1.0, 0.0, 0.0],
            },
            Vertex {
                position: [0.0, 0.5, 0.0],
                color: [0.0, 1.0, 0.0],
            },
            Vertex {
                position: [0.25, -0.1, 0.0],
                color: [0.0, 0.0, 1.0],
            },
        ];

        let indices = [0u16, 1, 2];

        let vertex_buffer = Buffer::new_slice(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter {
                    required_flags: MemoryPropertyFlags::DEVICE_LOCAL,
                    ..Default::default()
                },
                ..Default::default()
            },
            vertices.len() as DeviceSize,
        )?;

        let index_buffer = Buffer::new_slice(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::INDEX_BUFFER | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter {
                    required_flags: MemoryPropertyFlags::DEVICE_LOCAL,
                    ..Default::default()
                },
                ..Default::default()
            },
            indices.len() as DeviceSize,
        )?;

        let subbuffer_allocator = SubbufferAllocator::new(
            memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                arena_size: vertex_buffer.size() + index_buffer.size(),
                buffer_usage: BufferUsage::TRANSFER_SRC,
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );

        let vertex_staging_buffer =
            subbuffer_allocator.allocate_slice::<Vertex>(vertices.len() as DeviceSize)?;
        let index_staging_buffer =
            subbuffer_allocator.allocate_slice::<u16>(indices.len() as DeviceSize)?;

        {
            let mut vertex_writer = vertex_staging_buffer.write()?;
            vertex_writer.copy_from_slice(&vertices);
            let mut indices_writer = index_staging_buffer.write()?;
            indices_writer.copy_from_slice(&indices);
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
                        cull_mode: CullMode::Back,
                        ..Default::default()
                    }),
                    multisample_state: Some(MultisampleState::default()),
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

        buffers_upload_future.wait(None)?;

        Ok(Self {
            queue,
            memory_allocator,
            command_allocator,
            graphics_pipeline,
            vertex_buffer,
            index_buffer,
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
}

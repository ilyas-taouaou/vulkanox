use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use palette::Srgba;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, RenderingAttachmentInfo,
    RenderingAttachmentResolveInfo, RenderingInfo,
};
use vulkano::device::DeviceOwned;
use vulkano::format::ClearValue;
use vulkano::format::Format::B8G8R8A8_SRGB;
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageUsage, SampleCount};
use vulkano::memory::allocator::AllocationCreateInfo;
use vulkano::pipeline::graphics::viewport::Viewport;
use vulkano::pipeline::Pipeline;
use vulkano::render_pass::{AttachmentLoadOp, AttachmentStoreOp};
use vulkano::swapchain::{
    acquire_next_image, PresentMode, Surface, SurfaceInfo, Swapchain, SwapchainCreateInfo,
    SwapchainPresentInfo,
};
use vulkano::sync::GpuFuture;
use vulkano::{sync, Validated, VulkanError};
use winit::dpi::PhysicalPosition;
use winit::window::Window;

use crate::vulkan_device::{vs, VulkanDevice};

pub struct VulkanRenderer {
    vulkan_device: Arc<VulkanDevice>,
    window: Arc<Window>,
    swapchain: Arc<Swapchain>,
    swapchain_images: Vec<Arc<Image>>,
    swapchain_image_views: Vec<Arc<ImageView>>,
    intermediary_image: Arc<ImageView>,
    clear_color: [f32; 4],
    previous_frame_end: Option<Box<dyn GpuFuture>>,
    start_time: Instant,
    window_index: usize,
    window_count: usize,
    mouse_position: [f32; 2],
}

impl VulkanRenderer {
    pub fn new(
        vulkan_device: Arc<VulkanDevice>,
        window: Arc<Window>,
        clear_color: [f32; 4],
        is_vsync: bool,
        image_usage: ImageUsage,
        window_index: usize,
        window_count: usize,
    ) -> Result<Self> {
        let device = vulkan_device.queue().device();
        let physical_device = device.physical_device();
        let instance = device.instance();

        let window_inner_size = window.inner_size();

        let surface = Surface::from_window(Arc::clone(instance), Arc::clone(&window))?;
        let surface_info = SurfaceInfo::default();

        let surface_capabilities =
            physical_device.surface_capabilities(&surface, surface_info.clone())?;
        let surface_present_modes = physical_device
            .surface_present_modes(&surface, surface_info)?
            .collect::<Vec<_>>();

        let present_mode = if is_vsync {
            if surface_present_modes.contains(&PresentMode::Mailbox) {
                PresentMode::Mailbox
            } else {
                PresentMode::Fifo
            }
        } else {
            if surface_present_modes.contains(&PresentMode::Immediate) {
                PresentMode::Immediate
            } else if surface_present_modes.contains(&PresentMode::FifoRelaxed) {
                PresentMode::FifoRelaxed
            } else {
                PresentMode::Fifo
            }
        };

        let (swapchain, swapchain_images) = Swapchain::new(
            Arc::clone(device),
            surface,
            SwapchainCreateInfo {
                image_extent: surface_capabilities
                    .current_extent
                    .unwrap_or(window_inner_size.into()),
                image_format: B8G8R8A8_SRGB,
                min_image_count: (surface_capabilities.min_image_count + 1)
                    .min(surface_capabilities.max_image_count.unwrap_or(u32::MAX)),
                pre_transform: surface_capabilities.current_transform,
                present_mode,
                image_usage,
                ..Default::default()
            },
        )?;

        let swapchain_image_views = swapchain_images
            .iter()
            .map(|image| ImageView::new_default(Arc::clone(image)))
            .try_collect::<Vec<_>>()?;

        let intermediary_image = ImageView::new_default(
            Image::new(
                vulkan_device.memory_allocator().clone(),
                ImageCreateInfo {
                    format: swapchain.image_format(),
                    extent: [swapchain.image_extent()[0], swapchain.image_extent()[1], 1],
                    usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSIENT_ATTACHMENT,
                    samples: vulkan_device.samples(),
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .unwrap(),
        )
        .unwrap();

        let previous_frame_end = Some(sync::now(device.clone()).boxed());

        Ok(Self {
            vulkan_device,
            window,
            swapchain,
            swapchain_images,
            swapchain_image_views,
            intermediary_image,
            clear_color,
            previous_frame_end,
            start_time: Instant::now(),
            window_index,
            window_count,
            mouse_position: [0.0, 0.0],
        })
    }

    pub fn on_mouse_moved(&mut self, position: PhysicalPosition<f64>) {
        let size = self.window.inner_size();
        self.mouse_position = [
            position.x as f32 / size.width as f32,
            position.y as f32 / size.height as f32,
        ];
    }

    pub fn recreate(&mut self) -> Result<()> {
        let surface_info = SurfaceInfo::default();
        let surface_capabilities = self
            .swapchain
            .device()
            .physical_device()
            .surface_capabilities(&self.swapchain.surface(), surface_info.clone())?;

        self.swapchain_images.clear();
        self.swapchain_image_views.clear();

        let (new_swapchain, new_swapchain_images) =
            self.swapchain.recreate(SwapchainCreateInfo {
                image_extent: surface_capabilities
                    .current_extent
                    .unwrap_or(self.window.inner_size().into()),
                ..self.swapchain.create_info()
            })?;

        self.swapchain = new_swapchain;
        self.swapchain_image_views = new_swapchain_images
            .iter()
            .map(|image| ImageView::new_default(Arc::clone(image)))
            .try_collect::<Vec<_>>()?;
        self.swapchain_images = new_swapchain_images;

        Ok(())
    }

    pub fn render(&mut self) -> Result<()> {
        let image_extent: [u32; 2] = self.window.inner_size().into();
        if image_extent.contains(&0) {
            return Ok(());
        }

        self.previous_frame_end.as_mut().unwrap().cleanup_finished();

        let (image_index, suboptimal, acquire_future) =
            match acquire_next_image(self.swapchain.clone(), None).map_err(Validated::unwrap) {
                Ok(r) => r,
                Err(VulkanError::OutOfDate) => {
                    todo!()
                }
                Err(e) => panic!("failed to acquire next image: {e}"),
            };

        if suboptimal {
            todo!()
        }

        let mut builder = AutoCommandBufferBuilder::primary(
            self.vulkan_device.command_allocator(),
            self.vulkan_device.queue().queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let extent = self.swapchain.image_extent();

        let clear_color = Srgba::new(0.1, 0.1, 0.1, 1.0);

        let push_constants = vs::PushConstantData {
            time: (Instant::now() - self.start_time).as_secs_f32().into(),
            mousePosition: self.mouse_position,
        };

        builder
            .begin_rendering(RenderingInfo {
                color_attachments: vec![Some(RenderingAttachmentInfo {
                    load_op: AttachmentLoadOp::Clear,
                    store_op: AttachmentStoreOp::Store,
                    clear_value: Some(ClearValue::Float(clear_color.into_linear().into())),
                    resolve_info: Some(RenderingAttachmentResolveInfo::image_view(Arc::clone(
                        &self.swapchain_image_views[image_index as usize],
                    ))),
                    ..RenderingAttachmentInfo::image_view(Arc::clone(&self.intermediary_image))
                })],
                ..Default::default()
            })?
            .set_viewport(
                0,
                [Viewport {
                    offset: [0.0, 0.0],
                    extent: [extent[0] as f32, extent[1] as f32],
                    depth_range: 0.0..=1.0,
                }]
                .into_iter()
                .collect(),
            )?
            .bind_pipeline_graphics(Arc::clone(self.vulkan_device.graphics_pipeline()))?
            .bind_vertex_buffers(0, self.vulkan_device.vertex_buffer().clone())?
            .bind_index_buffer(self.vulkan_device.index_buffer().clone())?
            .push_constants(
                self.vulkan_device.graphics_pipeline().layout().clone(),
                0,
                push_constants,
            )?
            .draw(self.vulkan_device.vertex_buffer().len() as u32, 10, 0, 0)?
            .end_rendering()?;

        let command_buffer = builder.build()?;

        let future = self
            .previous_frame_end
            .take()
            .unwrap()
            .join(acquire_future)
            .then_execute(Arc::clone(self.vulkan_device.queue()), command_buffer)?
            .then_swapchain_present(
                Arc::clone(self.vulkan_device.queue()),
                SwapchainPresentInfo::swapchain_image_index(
                    Arc::clone(&self.swapchain),
                    image_index,
                ),
            )
            .then_signal_fence_and_flush();

        match future.map_err(Validated::unwrap) {
            Ok(future) => {
                self.previous_frame_end = Some(future.boxed());
            }
            Err(VulkanError::OutOfDate) => {
                self.recreate()?;
                self.previous_frame_end =
                    Some(sync::now(Arc::clone(self.vulkan_device.queue().device())).boxed());
            }
            Err(e) => {
                todo!()
            }
        }

        Ok(())
    }
}

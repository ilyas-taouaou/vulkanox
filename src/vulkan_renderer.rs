use std::sync::Arc;

use anyhow::Result;
use vulkano::command_buffer::{AutoCommandBufferBuilder, ClearColorImageInfo, CommandBufferUsage};
use vulkano::device::DeviceOwned;
use vulkano::format::ClearColorValue;
use vulkano::format::Format::B8G8R8A8_SRGB;
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageUsage};
use vulkano::swapchain::{
    acquire_next_image, PresentMode, Surface, SurfaceInfo, Swapchain, SwapchainCreateInfo,
    SwapchainPresentInfo,
};
use vulkano::sync::GpuFuture;
use vulkano::{sync, Validated, VulkanError};
use winit::window::Window;

use crate::vulkan_device::VulkanDevice;

pub struct VulkanRenderer {
    vulkan_device: Arc<VulkanDevice>,
    window: Arc<Window>,
    swapchain: Arc<Swapchain>,
    swapchain_images: Vec<Arc<Image>>,
    swapchain_image_views: Vec<Arc<ImageView>>,
    clear_color: [f32; 4],
    previous_frame_end: Option<Box<dyn GpuFuture>>,
}

impl VulkanRenderer {
    pub fn new(
        vulkan_device: Arc<VulkanDevice>,
        window: Arc<Window>,
        clear_color: [f32; 4],
        is_vsync: bool,
        image_usage: ImageUsage,
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

        let previous_frame_end = Some(sync::now(device.clone()).boxed());

        Ok(Self {
            vulkan_device,
            window,
            swapchain,
            swapchain_images,
            swapchain_image_views,
            clear_color,
            previous_frame_end,
        })
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

        builder.clear_color_image(ClearColorImageInfo {
            clear_value: ClearColorValue::Float(self.clear_color),
            ..ClearColorImageInfo::image(Arc::clone(&self.swapchain_images[image_index as usize]))
        })?;

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

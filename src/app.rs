use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use vulkano::image::ImageUsage;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{EventLoop, EventLoopWindowTarget};
use winit::window::{Window, WindowBuilder, WindowId};

use crate::vulkan_device::VulkanDevice;
use crate::vulkan_instance::VulkanInstance;
use crate::vulkan_renderer::VulkanRenderer;

pub struct VisualSystem {
    primary_window_id: WindowId,
    windows: HashMap<WindowId, Arc<Window>>,
    vulkan_instance: Arc<VulkanInstance>,
    vulkan_device: Arc<VulkanDevice>,
    vulkan_renderers: HashMap<WindowId, Arc<RefCell<VulkanRenderer>>>,
}

impl VisualSystem {
    pub fn new<T>(window_target: &EventLoopWindowTarget<T>) -> Result<Self> {
        let primary_window = Arc::new(
            WindowBuilder::new()
                .with_visible(false)
                .build(window_target)?,
        );
        let primary_window_id = primary_window.id();

        let vulkan_instance = Arc::new(VulkanInstance::new(&primary_window)?);
        let vulkan_device = Arc::new(VulkanDevice::new(Arc::clone(&vulkan_instance))?);

        let mut windows = HashMap::from([(primary_window_id, primary_window)]);

        for i in 0..3 {
            let window = Arc::new(
                WindowBuilder::new()
                    .with_visible(false)
                    .build(window_target)?,
            );
            windows.insert(window.id(), window);
        }

        let mut vulkan_renderers = HashMap::with_capacity(windows.len());

        for (window_index, (window_id, window)) in windows.iter().enumerate() {
            let c = (window_index as f32) / (windows.len() as f32);
            vulkan_renderers.insert(
                *window_id,
                Arc::new(RefCell::new(VulkanRenderer::new(
                    Arc::clone(&vulkan_device),
                    Arc::clone(&window),
                    [c, c, c, c],
                    true,
                    ImageUsage::COLOR_ATTACHMENT,
                    window_index,
                    windows.len(),
                )?)),
            );
        }

        windows.iter().for_each(|(_, window)| {
            window.set_visible(true);
        });

        Ok(Self {
            primary_window_id,
            windows,
            vulkan_instance,
            vulkan_device,
            vulkan_renderers,
        })
    }

    pub fn resume<T>(&mut self, window_target: &EventLoopWindowTarget<T>) -> Result<()> {
        for (window_index, (window_id, window)) in self.windows.iter().enumerate() {
            let c = (window_index as f32) / (self.windows.len() as f32);
            self.vulkan_renderers.insert(
                *window_id,
                Arc::new(RefCell::new(VulkanRenderer::new(
                    Arc::clone(&self.vulkan_device),
                    Arc::clone(&window),
                    [c, c, c, c],
                    true,
                    ImageUsage::COLOR_ATTACHMENT,
                    window_index,
                    self.windows.len(),
                )?)),
            );
        }
        Ok(())
    }

    pub fn suspend(&mut self) {
        self.vulkan_renderers.clear();
    }

    pub fn process_window_event(
        &mut self,
        event: WindowEvent,
        window_id: WindowId,
    ) -> Result<bool> {
        match event {
            WindowEvent::CloseRequested if self.primary_window_id == window_id => return Ok(true),
            WindowEvent::Resized(_) => self.vulkan_renderers[&window_id].borrow_mut().recreate()?,
            WindowEvent::RedrawRequested => {
                self.vulkan_renderers[&window_id].borrow_mut().render()?
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.vulkan_renderers[&window_id]
                    .borrow_mut()
                    .on_mouse_moved(position);
            }
            _ => {}
        };
        Ok(false)
    }

    pub fn request_redraw(&self) {
        self.windows
            .iter()
            .for_each(|(_, window)| window.request_redraw());
    }
}

pub struct App {
    is_started: bool,
    visual_system: Option<VisualSystem>,
}

impl App {
    pub fn process_event(
        &mut self,
        event: Event<()>,
        window_target: &EventLoopWindowTarget<()>,
    ) -> Result<()> {
        match event {
            Event::WindowEvent { event, window_id } => {
                if self
                    .visual_system
                    .as_mut()
                    .unwrap()
                    .process_window_event(event, window_id)?
                {
                    window_target.exit()
                }
            }
            Event::Resumed => {
                if self.is_started {
                    self.resume(window_target)?
                } else {
                    self.is_started = true;
                    self.start(window_target)?
                }
            }
            Event::Suspended => self.suspend(),
            Event::AboutToWait => self.visual_system.as_mut().unwrap().request_redraw(),
            _ => {}
        }
        Ok(())
    }
}

impl App {
    pub fn new<T>(event_loop: &EventLoop<T>) -> Result<Self> {
        Ok(Self {
            is_started: false,
            visual_system: None,
        })
    }

    pub fn start<T>(&mut self, window_target: &EventLoopWindowTarget<T>) -> Result<()> {
        self.visual_system = Some(VisualSystem::new(window_target)?);
        Ok(())
    }

    pub fn resume<T>(&mut self, window_target: &EventLoopWindowTarget<T>) -> Result<()> {
        self.visual_system.as_mut().unwrap().resume(window_target)?;
        Ok(())
    }

    pub fn suspend(&mut self) {
        self.visual_system.as_mut().unwrap().suspend();
    }
}

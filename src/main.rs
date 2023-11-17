#![feature(iterator_try_collect)]

use anyhow::{Context as AnyhowContext, Result};
use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoopBuilder;

use crate::app::App;

mod app;
mod vulkan_device;
mod vulkan_instance;
mod vulkan_renderer;

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let event_loop = EventLoopBuilder::new().build()?;
    let mut app = App::new(&event_loop)?;

    event_loop.run(move |event, window_target| app.process_event(event, window_target).unwrap())?;

    Ok(())
}

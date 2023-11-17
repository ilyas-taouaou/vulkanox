use std::sync::Arc;

use anyhow::Result;
use vulkano::command_buffer::allocator::{
    StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::device::{Device, DeviceCreateInfo, Features, Queue, QueueCreateInfo};
use vulkano::memory::allocator::StandardMemoryAllocator;

use crate::vulkan_instance::VulkanInstance;

pub struct VulkanDevice {
    queue: Arc<Queue>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_allocator: Arc<StandardCommandBufferAllocator>,
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

        Ok(Self {
            queue,
            memory_allocator,
            command_allocator,
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
}

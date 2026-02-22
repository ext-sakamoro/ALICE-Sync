//! Cloud-Side Multi-Device Spatial Synchronization
//!
//! Star topology sync where the cloud acts as hub for edge devices.
//! Each device maintains a local world state; the cloud merges SDF updates
//! and broadcasts deltas to other interested devices.
//!
//! Author: Moroya Sakamoto

use crate::{Node, NodeId, SyncError, WorldHash};
use std::collections::HashMap;

/// Device registration for cloud-side tracking
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Unique device identifier
    pub device_id: NodeId,
    /// Device name/label
    pub name: String,
    /// Last known world hash from this device
    pub last_world_hash: WorldHash,
    /// Last scene version received from this device
    pub last_scene_version: u32,
    /// Spatial region of interest (world coordinates)
    pub region_of_interest: Option<SpatialRegion>,
    /// Connection status
    pub connected: bool,
    /// Total frames received
    pub frames_received: u64,
    /// Last heartbeat timestamp (ms)
    pub last_heartbeat_ms: u64,
}

/// Axis-aligned spatial region
#[derive(Debug, Clone, Copy)]
pub struct SpatialRegion {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl SpatialRegion {
    pub fn new(min: [f32; 3], max: [f32; 3]) -> Self {
        Self { min, max }
    }

    /// Check if two regions overlap
    pub fn overlaps(&self, other: &SpatialRegion) -> bool {
        self.min[0] <= other.max[0]
            && self.max[0] >= other.min[0]
            && self.min[1] <= other.max[1]
            && self.max[1] >= other.min[1]
            && self.min[2] <= other.max[2]
            && self.max[2] >= other.min[2]
    }

    /// Check if a point is inside the region
    pub fn contains_point(&self, x: f32, y: f32, z: f32) -> bool {
        x >= self.min[0]
            && x <= self.max[0]
            && y >= self.min[1]
            && y <= self.max[1]
            && z >= self.min[2]
            && z <= self.max[2]
    }
}

/// Cloud synchronization hub
///
/// Manages connected edge devices and merges their spatial updates
/// into a unified world state.
pub struct CloudSyncHub {
    /// Registered devices
    devices: HashMap<u64, DeviceInfo>,
    /// Cloud-side merged world state
    cloud_node: Node,
    /// Global scene version (monotonically increasing across all devices)
    global_scene_version: u32,
    /// World hash for spatial consistency verification
    world_hash: WorldHash,
}

impl Default for CloudSyncHub {
    fn default() -> Self {
        Self::new()
    }
}

impl CloudSyncHub {
    /// Create a new cloud sync hub
    pub fn new() -> Self {
        Self {
            devices: HashMap::new(),
            cloud_node: Node::new(NodeId(0)), // Cloud is node 0
            global_scene_version: 0,
            world_hash: WorldHash::zero(),
        }
    }

    /// Register a new edge device
    pub fn register_device(&mut self, device_id: u64, name: String) -> &DeviceInfo {
        let node_id = NodeId(device_id);
        self.cloud_node.add_peer(node_id);

        self.devices.entry(device_id).or_insert(DeviceInfo {
            device_id: node_id,
            name,
            last_world_hash: WorldHash::zero(),
            last_scene_version: 0,
            region_of_interest: None,
            connected: true,
            frames_received: 0,
            last_heartbeat_ms: 0,
        })
    }

    /// Update device heartbeat
    pub fn heartbeat(&mut self, device_id: u64, timestamp_ms: u64) {
        if let Some(device) = self.devices.get_mut(&device_id) {
            device.last_heartbeat_ms = timestamp_ms;
            device.connected = true;
        }
    }

    /// Set device region of interest
    pub fn set_region_of_interest(&mut self, device_id: u64, region: SpatialRegion) {
        if let Some(device) = self.devices.get_mut(&device_id) {
            device.region_of_interest = Some(region);
        }
    }

    /// Process an SDF update from an edge device
    ///
    /// Returns the list of device IDs that should receive this update
    /// (devices with overlapping regions of interest).
    pub fn process_device_update(
        &mut self,
        device_id: u64,
        scene_version: u32,
        world_hash: WorldHash,
        update_region: SpatialRegion,
    ) -> Vec<u64> {
        if let Some(device) = self.devices.get_mut(&device_id) {
            device.last_world_hash = world_hash;
            device.last_scene_version = scene_version;
            device.frames_received += 1;
        }

        self.global_scene_version = self.global_scene_version.max(scene_version);
        self.world_hash = self.world_hash.xor(world_hash.0);

        let capacity = self.devices.len().saturating_sub(1);
        let mut recipients = Vec::with_capacity(capacity);
        for (&id, info) in &self.devices {
            if id != device_id
                && info.connected
                && info
                    .region_of_interest
                    .as_ref()
                    .map(|r| r.overlaps(&update_region))
                    .unwrap_or(true)
            {
                recipients.push(id);
            }
        }
        recipients
    }

    /// Verify spatial consistency between cloud and device
    pub fn verify_consistency(
        &self,
        device_id: u64,
        reported_hash: WorldHash,
    ) -> Result<(), SyncError> {
        if let Some(device) = self.devices.get(&device_id) {
            if device.last_world_hash != reported_hash {
                return Err(SyncError::StateDivergence {
                    local: device.last_world_hash,
                    remote: reported_hash,
                });
            }
        }
        Ok(())
    }

    /// Get all connected devices
    pub fn connected_devices(&self) -> Vec<&DeviceInfo> {
        self.devices.values().filter(|d| d.connected).collect()
    }

    /// Get device info by ID
    pub fn device(&self, device_id: u64) -> Option<&DeviceInfo> {
        self.devices.get(&device_id)
    }

    /// Disconnect a device
    pub fn disconnect_device(&mut self, device_id: u64) {
        if let Some(device) = self.devices.get_mut(&device_id) {
            device.connected = false;
        }
    }

    /// Current global scene version
    pub fn global_scene_version(&self) -> u32 {
        self.global_scene_version
    }

    /// Current world hash
    pub fn world_hash(&self) -> WorldHash {
        self.world_hash
    }

    /// Total registered device count
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cloud_hub_register_device() {
        let mut hub = CloudSyncHub::new();
        hub.register_device(1, "Pi5-001".to_string());
        hub.register_device(2, "Pi5-002".to_string());
        assert_eq!(hub.device_count(), 2);
    }

    #[test]
    fn test_spatial_region_overlap() {
        let r1 = SpatialRegion::new([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
        let r2 = SpatialRegion::new([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
        let r3 = SpatialRegion::new([5.0, 5.0, 5.0], [6.0, 6.0, 6.0]);

        assert!(r1.overlaps(&r2));
        assert!(!r1.overlaps(&r3));
    }

    #[test]
    fn test_process_update_routing() {
        let mut hub = CloudSyncHub::new();
        hub.register_device(1, "dev-1".to_string());
        hub.register_device(2, "dev-2".to_string());
        hub.register_device(3, "dev-3".to_string());

        // Set ROI for device 2 (overlaps with update)
        hub.set_region_of_interest(2, SpatialRegion::new([-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]));

        // Set ROI for device 3 (doesn't overlap)
        hub.set_region_of_interest(
            3,
            SpatialRegion::new([100.0, 100.0, 100.0], [200.0, 200.0, 200.0]),
        );

        let recipients = hub.process_device_update(
            1,
            1,
            WorldHash(0x1234),
            SpatialRegion::new([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]),
        );

        // Only device 2 should receive (overlapping ROI)
        assert_eq!(recipients.len(), 1);
        assert_eq!(recipients[0], 2);
    }

    #[test]
    fn test_heartbeat_tracking() {
        let mut hub = CloudSyncHub::new();
        hub.register_device(1, "dev-1".to_string());

        hub.heartbeat(1, 1000);
        assert_eq!(hub.device(1).unwrap().last_heartbeat_ms, 1000);
    }

    #[test]
    fn test_disconnect_device() {
        let mut hub = CloudSyncHub::new();
        hub.register_device(1, "dev-1".to_string());
        assert!(hub.device(1).unwrap().connected);

        hub.disconnect_device(1);
        assert!(!hub.device(1).unwrap().connected);

        assert_eq!(hub.connected_devices().len(), 0);
    }
}

//! C-ABI FFI for Unity/UE5 — 71 `#[no_mangle] extern "C"` functions
//!
//! Prefix: `as_sync_*`
//!
//! # Safety
//!
//! All functions receiving raw pointers require non-null, valid pointers
//! obtained from the corresponding `_new` / `_create` functions.
//! Caller must free handles with the matching `_free` function.

#![allow(clippy::missing_safety_doc, clippy::missing_const_for_fn)]

use crate::event::{Event, EventKind, EventStream};
use crate::fixed_point::{Fixed, Vec3Fixed};
use crate::input_sync::{InputFrame, LockstepSession, RollbackAction, RollbackSession};
use crate::node::{Node, NodeId, NodeState};
use crate::world::World;
use crate::world_soa::WorldSoA;
use std::ptr;

// ============================================================================
// Opaque handle types
// ============================================================================

#[repr(C)]
pub struct AsSyncWorld(World);

#[repr(C)]
pub struct AsSyncWorldSoA(WorldSoA);

#[repr(C)]
pub struct AsSyncNode(Node);

#[repr(C)]
pub struct AsSyncEventStream(EventStream);

#[repr(C)]
pub struct AsSyncEvent(Event);

#[repr(C)]
pub struct AsSyncInputFrame(InputFrame);

#[repr(C)]
pub struct AsSyncLockstep(LockstepSession);

#[repr(C)]
pub struct AsSyncRollback(RollbackSession);

#[repr(C)]
pub struct AsSyncInputFrameArray(Vec<InputFrame>);

// ============================================================================
// World (AoS)
// ============================================================================

#[no_mangle]
pub extern "C" fn as_sync_world_new(seed: u64) -> *mut AsSyncWorld {
    Box::into_raw(Box::new(AsSyncWorld(World::new(seed))))
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_world_free(ptr: *mut AsSyncWorld) {
    if !ptr.is_null() {
        drop(Box::from_raw(ptr));
    }
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_world_hash(ptr: *const AsSyncWorld) -> u64 {
    if ptr.is_null() {
        return 0;
    }
    (*ptr).0.hash().0
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_world_entity_count(ptr: *const AsSyncWorld) -> u32 {
    if ptr.is_null() {
        return 0;
    }
    (*ptr).0.entity_count() as u32
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_world_frame(ptr: *const AsSyncWorld) -> u64 {
    if ptr.is_null() {
        return 0;
    }
    (*ptr).0.frame()
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_world_apply_event(
    ptr: *mut AsSyncWorld,
    event: *const AsSyncEvent,
) -> i32 {
    if ptr.is_null() || event.is_null() {
        return -1;
    }
    match (*ptr).0.apply(&(*event).0) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

/// entity位置を取得。成功=0, 失敗=-1
#[no_mangle]
pub unsafe extern "C" fn as_sync_world_get_entity_position(
    ptr: *const AsSyncWorld,
    entity_id: u32,
    out_x: *mut f32,
    out_y: *mut f32,
    out_z: *mut f32,
) -> i32 {
    if ptr.is_null() || out_x.is_null() || out_y.is_null() || out_z.is_null() {
        return -1;
    }
    (*ptr).0.get_entity(entity_id).map_or(-1, |e| {
        let pos = e.position.to_f32_array();
        *out_x = pos[0];
        *out_y = pos[1];
        *out_z = pos[2];
        0
    })
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_world_recalculate_hash(ptr: *mut AsSyncWorld) -> u64 {
    if ptr.is_null() {
        return 0;
    }
    (*ptr).0.recalculate_hash().0
}

// ============================================================================
// WorldSoA
// ============================================================================

#[no_mangle]
pub extern "C" fn as_sync_world_soa_new(seed: u64) -> *mut AsSyncWorldSoA {
    Box::into_raw(Box::new(AsSyncWorldSoA(WorldSoA::new(seed))))
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_world_soa_free(ptr: *mut AsSyncWorldSoA) {
    if !ptr.is_null() {
        drop(Box::from_raw(ptr));
    }
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_world_soa_hash(ptr: *const AsSyncWorldSoA) -> u64 {
    if ptr.is_null() {
        return 0;
    }
    (*ptr).0.hash().0
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_world_soa_entity_count(ptr: *const AsSyncWorldSoA) -> u32 {
    if ptr.is_null() {
        return 0;
    }
    (*ptr).0.entity_count() as u32
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_world_soa_apply_event(
    ptr: *mut AsSyncWorldSoA,
    event: *const AsSyncEvent,
) -> i32 {
    if ptr.is_null() || event.is_null() {
        return -1;
    }
    match (*ptr).0.apply(&(*event).0) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

// ============================================================================
// Node
// ============================================================================

#[no_mangle]
pub extern "C" fn as_sync_node_new(node_id: u64) -> *mut AsSyncNode {
    Box::into_raw(Box::new(AsSyncNode(Node::new(NodeId(node_id)))))
}

#[no_mangle]
pub extern "C" fn as_sync_node_with_seed(node_id: u64, seed: u64) -> *mut AsSyncNode {
    Box::into_raw(Box::new(AsSyncNode(Node::with_seed(NodeId(node_id), seed))))
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_node_free(ptr: *mut AsSyncNode) {
    if !ptr.is_null() {
        drop(Box::from_raw(ptr));
    }
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_node_world_hash(ptr: *const AsSyncNode) -> u64 {
    if ptr.is_null() {
        return 0;
    }
    (*ptr).0.world_hash().0
}

/// `NodeState`: 0=Disconnected, 1=Connecting, 2=Synced, 3=Diverged
#[no_mangle]
pub unsafe extern "C" fn as_sync_node_state(ptr: *const AsSyncNode) -> u8 {
    if ptr.is_null() {
        return 0;
    }
    match (*ptr).0.state() {
        NodeState::Disconnected => 0,
        NodeState::Connecting => 1,
        NodeState::Synced => 2,
        NodeState::Diverged => 3,
    }
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_node_emit(ptr: *mut AsSyncNode, event: *const AsSyncEvent) -> i32 {
    if ptr.is_null() || event.is_null() {
        return -1;
    }
    match (*ptr).0.emit((*event).0.clone()) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_node_apply_event(
    ptr: *mut AsSyncNode,
    event: *const AsSyncEvent,
) -> i32 {
    if ptr.is_null() || event.is_null() {
        return -1;
    }
    match (*ptr).0.apply_event(&(*event).0) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_node_add_peer(ptr: *mut AsSyncNode, peer_id: u64) {
    if ptr.is_null() {
        return;
    }
    (*ptr).0.add_peer(NodeId(peer_id));
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_node_entity_count(ptr: *const AsSyncNode) -> u32 {
    if ptr.is_null() {
        return 0;
    }
    (*ptr).0.world().entity_count() as u32
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_node_events_count(ptr: *const AsSyncNode) -> u32 {
    if ptr.is_null() {
        return 0;
    }
    (*ptr).0.events().len() as u32
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_node_events_bytes(ptr: *const AsSyncNode) -> u32 {
    if ptr.is_null() {
        return 0;
    }
    (*ptr).0.events().total_bytes() as u32
}

// ============================================================================
// Event construction
// ============================================================================

#[no_mangle]
pub extern "C" fn as_sync_event_new_motion(
    entity: u32,
    dx: i16,
    dy: i16,
    dz: i16,
) -> *mut AsSyncEvent {
    Box::into_raw(Box::new(AsSyncEvent(Event::new(EventKind::Motion {
        entity,
        delta: [dx, dy, dz],
    }))))
}

#[no_mangle]
pub extern "C" fn as_sync_event_new_spawn(
    entity: u32,
    kind: u16,
    px: i16,
    py: i16,
    pz: i16,
) -> *mut AsSyncEvent {
    Box::into_raw(Box::new(AsSyncEvent(Event::new(EventKind::Spawn {
        entity,
        kind,
        pos: [px, py, pz],
    }))))
}

#[no_mangle]
pub extern "C" fn as_sync_event_new_despawn(entity: u32) -> *mut AsSyncEvent {
    Box::into_raw(Box::new(AsSyncEvent(Event::new(EventKind::Despawn {
        entity,
    }))))
}

#[no_mangle]
pub extern "C" fn as_sync_event_new_property(
    entity: u32,
    prop: u16,
    value: i32,
) -> *mut AsSyncEvent {
    Box::into_raw(Box::new(AsSyncEvent(Event::new(EventKind::Property {
        entity,
        prop,
        value,
    }))))
}

#[no_mangle]
pub extern "C" fn as_sync_event_new_input(player: u16, code: u32) -> *mut AsSyncEvent {
    Box::into_raw(Box::new(AsSyncEvent(Event::new(EventKind::Input {
        player,
        code,
    }))))
}

#[no_mangle]
pub extern "C" fn as_sync_event_new_tick(frame: u64) -> *mut AsSyncEvent {
    Box::into_raw(Box::new(AsSyncEvent(Event::new(EventKind::Tick { frame }))))
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_event_free(ptr: *mut AsSyncEvent) {
    if !ptr.is_null() {
        drop(Box::from_raw(ptr));
    }
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_event_size_bytes(ptr: *const AsSyncEvent) -> u32 {
    if ptr.is_null() {
        return 0;
    }
    (*ptr).0.size_bytes() as u32
}

// ============================================================================
// EventStream
// ============================================================================

#[no_mangle]
pub extern "C" fn as_sync_event_stream_new() -> *mut AsSyncEventStream {
    Box::into_raw(Box::new(AsSyncEventStream(EventStream::new())))
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_event_stream_free(ptr: *mut AsSyncEventStream) {
    if !ptr.is_null() {
        drop(Box::from_raw(ptr));
    }
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_event_stream_push(
    ptr: *mut AsSyncEventStream,
    event: *const AsSyncEvent,
    origin: u64,
) -> u64 {
    if ptr.is_null() || event.is_null() {
        return 0;
    }
    (*ptr).0.push((*event).0.clone(), origin).0
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_event_stream_len(ptr: *const AsSyncEventStream) -> u32 {
    if ptr.is_null() {
        return 0;
    }
    (*ptr).0.len() as u32
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_event_stream_is_empty(ptr: *const AsSyncEventStream) -> i32 {
    if ptr.is_null() {
        return 1;
    }
    i32::from((*ptr).0.is_empty())
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_event_stream_current_seq(ptr: *const AsSyncEventStream) -> u64 {
    if ptr.is_null() {
        return 0;
    }
    (*ptr).0.current_seq().0
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_event_stream_total_bytes(ptr: *const AsSyncEventStream) -> u32 {
    if ptr.is_null() {
        return 0;
    }
    (*ptr).0.total_bytes() as u32
}

// ============================================================================
// InputFrame
// ============================================================================

#[no_mangle]
pub extern "C" fn as_sync_input_frame_new(frame: u64, player_id: u8) -> *mut AsSyncInputFrame {
    Box::into_raw(Box::new(AsSyncInputFrame(InputFrame::new(
        frame, player_id,
    ))))
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_input_frame_free(ptr: *mut AsSyncInputFrame) {
    if !ptr.is_null() {
        drop(Box::from_raw(ptr));
    }
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_input_frame_set_movement(
    ptr: *mut AsSyncInputFrame,
    x: i16,
    y: i16,
    z: i16,
) {
    if ptr.is_null() {
        return;
    }
    (*ptr).0.movement = [x, y, z];
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_input_frame_set_actions(ptr: *mut AsSyncInputFrame, actions: u32) {
    if ptr.is_null() {
        return;
    }
    (*ptr).0.actions = actions;
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_input_frame_set_aim(
    ptr: *mut AsSyncInputFrame,
    x: i16,
    y: i16,
    z: i16,
) {
    if ptr.is_null() {
        return;
    }
    (*ptr).0.aim = [x, y, z];
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_input_frame_get_frame(ptr: *const AsSyncInputFrame) -> u64 {
    if ptr.is_null() {
        return 0;
    }
    (*ptr).0.frame
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_input_frame_get_player_id(ptr: *const AsSyncInputFrame) -> u8 {
    if ptr.is_null() {
        return 0;
    }
    (*ptr).0.player_id
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_input_frame_get_movement(
    ptr: *const AsSyncInputFrame,
    out_x: *mut i16,
    out_y: *mut i16,
    out_z: *mut i16,
) {
    if ptr.is_null() || out_x.is_null() || out_y.is_null() || out_z.is_null() {
        return;
    }
    *out_x = (*ptr).0.movement[0];
    *out_y = (*ptr).0.movement[1];
    *out_z = (*ptr).0.movement[2];
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_input_frame_get_actions(ptr: *const AsSyncInputFrame) -> u32 {
    if ptr.is_null() {
        return 0;
    }
    (*ptr).0.actions
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_input_frame_get_aim(
    ptr: *const AsSyncInputFrame,
    out_x: *mut i16,
    out_y: *mut i16,
    out_z: *mut i16,
) {
    if ptr.is_null() || out_x.is_null() || out_y.is_null() || out_z.is_null() {
        return;
    }
    *out_x = (*ptr).0.aim[0];
    *out_y = (*ptr).0.aim[1];
    *out_z = (*ptr).0.aim[2];
}

// ============================================================================
// LockstepSession
// ============================================================================

#[no_mangle]
pub extern "C" fn as_sync_lockstep_new(player_count: u8) -> *mut AsSyncLockstep {
    Box::into_raw(Box::new(AsSyncLockstep(LockstepSession::new(player_count))))
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_lockstep_free(ptr: *mut AsSyncLockstep) {
    if !ptr.is_null() {
        drop(Box::from_raw(ptr));
    }
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_lockstep_add_local_input(
    ptr: *mut AsSyncLockstep,
    input: *const AsSyncInputFrame,
) {
    if ptr.is_null() || input.is_null() {
        return;
    }
    (*ptr).0.add_local_input((*input).0);
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_lockstep_add_remote_input(
    ptr: *mut AsSyncLockstep,
    input: *const AsSyncInputFrame,
) {
    if ptr.is_null() || input.is_null() {
        return;
    }
    (*ptr).0.add_remote_input((*input).0);
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_lockstep_ready(ptr: *const AsSyncLockstep) -> i32 {
    if ptr.is_null() {
        return 0;
    }
    i32::from((*ptr).0.ready_to_advance())
}

/// advance成功時: `InputFrameArray`へのポインタ（`as_sync_input_frame_array_free`で解放）
/// 未ready時: null
#[no_mangle]
pub unsafe extern "C" fn as_sync_lockstep_advance(
    ptr: *mut AsSyncLockstep,
) -> *mut AsSyncInputFrameArray {
    if ptr.is_null() {
        return ptr::null_mut();
    }
    (*ptr).0.advance().map_or(ptr::null_mut(), |inputs| {
        Box::into_raw(Box::new(AsSyncInputFrameArray(inputs)))
    })
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_lockstep_confirmed_frame(ptr: *const AsSyncLockstep) -> u64 {
    if ptr.is_null() {
        return 0;
    }
    (*ptr).0.confirmed_frame()
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_lockstep_record_checksum(
    ptr: *mut AsSyncLockstep,
    frame: u64,
    checksum: u64,
) {
    if ptr.is_null() {
        return;
    }
    (*ptr).0.record_checksum(frame, checksum);
}

// ============================================================================
// RollbackSession
// ============================================================================

#[no_mangle]
pub extern "C" fn as_sync_rollback_new(
    player_count: u8,
    local_player: u8,
    max_rollback: u64,
) -> *mut AsSyncRollback {
    Box::into_raw(Box::new(AsSyncRollback(RollbackSession::new(
        player_count,
        local_player,
        max_rollback,
    ))))
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_rollback_free(ptr: *mut AsSyncRollback) {
    if !ptr.is_null() {
        drop(Box::from_raw(ptr));
    }
}

/// ローカル入力追加 → 全プレイヤーのフレーム入力を返す
#[no_mangle]
pub unsafe extern "C" fn as_sync_rollback_add_local_input(
    ptr: *mut AsSyncRollback,
    input: *const AsSyncInputFrame,
) -> *mut AsSyncInputFrameArray {
    if ptr.is_null() || input.is_null() {
        return ptr::null_mut();
    }
    let inputs = (*ptr).0.add_local_input((*input).0);
    Box::into_raw(Box::new(AsSyncInputFrameArray(inputs)))
}

/// リモート入力追加 → `RollbackAction`: 0=None, 1=Rollback, 2=Desync
/// `rollback_frame`: Rollback/Desyncの場合のフレーム番号（out）
#[no_mangle]
pub unsafe extern "C" fn as_sync_rollback_add_remote_input(
    ptr: *mut AsSyncRollback,
    input: *const AsSyncInputFrame,
    out_rollback_frame: *mut u64,
) -> u8 {
    if ptr.is_null() || input.is_null() {
        return 0;
    }
    match (*ptr).0.add_remote_input((*input).0) {
        RollbackAction::None => 0,
        RollbackAction::Rollback { to_frame } => {
            if !out_rollback_frame.is_null() {
                *out_rollback_frame = to_frame;
            }
            1
        }
        RollbackAction::Desync { frame } => {
            if !out_rollback_frame.is_null() {
                *out_rollback_frame = frame;
            }
            2
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_rollback_confirmed_frame(ptr: *const AsSyncRollback) -> u64 {
    if ptr.is_null() {
        return 0;
    }
    (*ptr).0.confirmed_frame()
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_rollback_predicted_frame(ptr: *const AsSyncRollback) -> u64 {
    if ptr.is_null() {
        return 0;
    }
    (*ptr).0.predicted_frame()
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_rollback_frames_ahead(ptr: *const AsSyncRollback) -> u64 {
    if ptr.is_null() {
        return 0;
    }
    (*ptr).0.frames_ahead()
}

// ============================================================================
// InputFrameArray (for lockstep/rollback results)
// ============================================================================

#[no_mangle]
pub unsafe extern "C" fn as_sync_input_frame_array_free(ptr: *mut AsSyncInputFrameArray) {
    if !ptr.is_null() {
        drop(Box::from_raw(ptr));
    }
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_input_frame_array_len(ptr: *const AsSyncInputFrameArray) -> u32 {
    if ptr.is_null() {
        return 0;
    }
    (*ptr).0.len() as u32
}

/// 配列のi番目の`InputFrame`をコピーして返す。解放は`as_sync_input_frame_free`
#[no_mangle]
pub unsafe extern "C" fn as_sync_input_frame_array_get(
    ptr: *const AsSyncInputFrameArray,
    index: u32,
) -> *mut AsSyncInputFrame {
    if ptr.is_null() {
        return ptr::null_mut();
    }
    let arr = &(*ptr).0;
    arr.get(index as usize).map_or(ptr::null_mut(), |f| {
        Box::into_raw(Box::new(AsSyncInputFrame(*f)))
    })
}

// ============================================================================
// Fixed-point utilities
// ============================================================================

#[no_mangle]
pub extern "C" fn as_sync_fixed_from_f32(f: f32) -> i32 {
    Fixed::from_f32(f).0
}

#[no_mangle]
pub extern "C" fn as_sync_fixed_to_f32(bits: i32) -> f32 {
    Fixed::from_bits(bits).to_f32()
}

#[no_mangle]
pub extern "C" fn as_sync_vec3_hash(x: f32, y: f32, z: f32) -> u64 {
    Vec3Fixed::from_f32(x, y, z).hash_bits()
}

// ============================================================================
// Version
// ============================================================================

#[no_mangle]
pub extern "C" fn as_sync_version() -> u32 {
    // 0.6.0 → 0*10000 + 6*100 + 0 = 600
    600
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffi_world_roundtrip() {
        unsafe {
            let world = as_sync_world_new(42);
            assert_eq!(as_sync_world_entity_count(world), 0);

            let spawn = as_sync_event_new_spawn(1, 0, 100, 200, 300);
            assert_eq!(as_sync_world_apply_event(world, spawn), 0);
            assert_eq!(as_sync_world_entity_count(world), 1);

            let mut x = 0.0f32;
            let mut y = 0.0f32;
            let mut z = 0.0f32;
            assert_eq!(
                as_sync_world_get_entity_position(world, 1, &mut x, &mut y, &mut z),
                0
            );
            assert!(x.abs() > 0.0 || y.abs() > 0.0 || z.abs() > 0.0);

            as_sync_event_free(spawn);
            as_sync_world_free(world);
        }
    }

    #[test]
    fn test_ffi_node_sync() {
        unsafe {
            let node_a = as_sync_node_new(1);
            let node_b = as_sync_node_new(2);

            let spawn = as_sync_event_new_spawn(1, 0, 0, 0, 0);
            as_sync_node_apply_event(node_a, spawn);
            as_sync_node_apply_event(node_b, spawn);
            as_sync_event_free(spawn);

            assert_eq!(
                as_sync_node_world_hash(node_a),
                as_sync_node_world_hash(node_b)
            );

            as_sync_node_free(node_a);
            as_sync_node_free(node_b);
        }
    }

    #[test]
    fn test_ffi_input_frame() {
        unsafe {
            let frame = as_sync_input_frame_new(42, 1);
            as_sync_input_frame_set_movement(frame, 100, 0, -50);
            as_sync_input_frame_set_actions(frame, 0x05);

            assert_eq!(as_sync_input_frame_get_frame(frame), 42);
            assert_eq!(as_sync_input_frame_get_player_id(frame), 1);
            assert_eq!(as_sync_input_frame_get_actions(frame), 0x05);

            let mut mx = 0i16;
            let mut my = 0i16;
            let mut mz = 0i16;
            as_sync_input_frame_get_movement(frame, &mut mx, &mut my, &mut mz);
            assert_eq!(mx, 100);
            assert_eq!(mz, -50);

            as_sync_input_frame_free(frame);
        }
    }

    #[test]
    fn test_ffi_lockstep() {
        unsafe {
            let session = as_sync_lockstep_new(2);
            let input0 = as_sync_input_frame_new(1, 0);
            let input1 = as_sync_input_frame_new(1, 1);

            as_sync_lockstep_add_local_input(session, input0);
            assert_eq!(as_sync_lockstep_ready(session), 0);

            as_sync_lockstep_add_remote_input(session, input1);
            assert_eq!(as_sync_lockstep_ready(session), 1);

            let arr = as_sync_lockstep_advance(session);
            assert!(!arr.is_null());
            assert_eq!(as_sync_input_frame_array_len(arr), 2);

            as_sync_input_frame_array_free(arr);
            as_sync_input_frame_free(input0);
            as_sync_input_frame_free(input1);
            as_sync_lockstep_free(session);
        }
    }

    #[test]
    fn test_ffi_rollback() {
        unsafe {
            let session = as_sync_rollback_new(2, 0, 8);
            let input = as_sync_input_frame_new(1, 0);
            as_sync_input_frame_set_movement(input, 1, 0, 0);

            let arr = as_sync_rollback_add_local_input(session, input);
            assert_eq!(as_sync_input_frame_array_len(arr), 2);
            as_sync_input_frame_array_free(arr);

            let remote = as_sync_input_frame_new(1, 1);
            let mut rb_frame = 0u64;
            let action = as_sync_rollback_add_remote_input(session, remote, &mut rb_frame);
            assert_eq!(action, 0); // None

            as_sync_input_frame_free(input);
            as_sync_input_frame_free(remote);
            as_sync_rollback_free(session);
        }
    }

    #[test]
    fn test_ffi_fixed_point() {
        let bits = as_sync_fixed_from_f32(1.5);
        let back = as_sync_fixed_to_f32(bits);
        assert!((back - 1.5).abs() < 0.001);
    }

    #[test]
    fn test_ffi_version() {
        assert_eq!(as_sync_version(), 600);
    }

    #[test]
    fn test_ffi_event_stream() {
        unsafe {
            let stream = as_sync_event_stream_new();
            assert_eq!(as_sync_event_stream_len(stream), 0);
            assert_eq!(as_sync_event_stream_is_empty(stream), 1);

            let event = as_sync_event_new_motion(1, 10, 20, 30);
            as_sync_event_stream_push(stream, event, 42);
            assert_eq!(as_sync_event_stream_len(stream), 1);
            assert_eq!(as_sync_event_stream_is_empty(stream), 0);

            as_sync_event_free(event);
            as_sync_event_stream_free(stream);
        }
    }

    #[test]
    fn test_ffi_world_soa() {
        unsafe {
            let world = as_sync_world_soa_new(42);
            let spawn = as_sync_event_new_spawn(1, 0, 100, 0, 0);
            assert_eq!(as_sync_world_soa_apply_event(world, spawn), 0);
            assert_eq!(as_sync_world_soa_entity_count(world), 1);
            assert_ne!(as_sync_world_soa_hash(world), 0);

            as_sync_event_free(spawn);
            as_sync_world_soa_free(world);
        }
    }

    #[test]
    fn test_ffi_null_safety() {
        unsafe {
            assert_eq!(as_sync_world_hash(ptr::null()), 0);
            assert_eq!(as_sync_world_entity_count(ptr::null()), 0);
            assert_eq!(as_sync_node_world_hash(ptr::null()), 0);
            as_sync_world_free(ptr::null_mut());
            as_sync_node_free(ptr::null_mut());
            as_sync_event_free(ptr::null_mut());
        }
    }
}

//! C-ABI FFI for Unity/UE5 — 75 `#[no_mangle] extern "C"` functions
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
    ffi_guard(ptr::null_mut(), || {
        Box::into_raw(Box::new(AsSyncWorld(World::new(seed))))
    })
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_world_free(ptr: *mut AsSyncWorld) {
    ffi_guard((), || {
        if !ptr.is_null() {
            drop(Box::from_raw(ptr));
        }
    })
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_world_hash(ptr: *const AsSyncWorld) -> u64 {
    ffi_guard(0, || {
        if ptr.is_null() {
            return 0;
        }
        (*ptr).0.hash().0
    })
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_world_entity_count(ptr: *const AsSyncWorld) -> u32 {
    ffi_guard(0, || {
        if ptr.is_null() {
            return 0;
        }
        (*ptr).0.entity_count() as u32
    })
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_world_frame(ptr: *const AsSyncWorld) -> u64 {
    ffi_guard(0, || {
        if ptr.is_null() {
            return 0;
        }
        (*ptr).0.frame()
    })
}

#[no_mangle]
#[allow(
    clippy::disallowed_methods,
    reason = "the C-ABI of the single-writer world, applied in the order called; peers' events go through as_sync_node_receive (signed, ALICE-Law G13)"
)]
pub unsafe extern "C" fn as_sync_world_apply_event(
    ptr: *mut AsSyncWorld,
    event: *const AsSyncEvent,
) -> i32 {
    ffi_guard(-1, || {
        if ptr.is_null() || event.is_null() {
            return -1;
        }
        match (*ptr).0.apply(&(*event).0) {
            Ok(()) => 0,
            Err(_) => -1,
        }
    })
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
    ffi_guard(-1, || {
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
    })
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_world_recalculate_hash(ptr: *mut AsSyncWorld) -> u64 {
    ffi_guard(0, || {
        if ptr.is_null() {
            return 0;
        }
        (*ptr).0.recalculate_hash().0
    })
}

// ============================================================================
// WorldSoA
// ============================================================================

#[no_mangle]
pub extern "C" fn as_sync_world_soa_new(seed: u64) -> *mut AsSyncWorldSoA {
    ffi_guard(ptr::null_mut(), || {
        Box::into_raw(Box::new(AsSyncWorldSoA(WorldSoA::new(seed))))
    })
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_world_soa_free(ptr: *mut AsSyncWorldSoA) {
    ffi_guard((), || {
        if !ptr.is_null() {
            drop(Box::from_raw(ptr));
        }
    })
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_world_soa_hash(ptr: *const AsSyncWorldSoA) -> u64 {
    ffi_guard(0, || {
        if ptr.is_null() {
            return 0;
        }
        (*ptr).0.hash().0
    })
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_world_soa_entity_count(ptr: *const AsSyncWorldSoA) -> u32 {
    ffi_guard(0, || {
        if ptr.is_null() {
            return 0;
        }
        (*ptr).0.entity_count() as u32
    })
}

#[no_mangle]
#[allow(
    clippy::disallowed_methods,
    reason = "the C-ABI of the single-writer world, applied in the order called; peers' events go through as_sync_node_receive (signed, ALICE-Law G13)"
)]
pub unsafe extern "C" fn as_sync_world_soa_apply_event(
    ptr: *mut AsSyncWorldSoA,
    event: *const AsSyncEvent,
) -> i32 {
    ffi_guard(-1, || {
        if ptr.is_null() || event.is_null() {
            return -1;
        }
        match (*ptr).0.apply(&(*event).0) {
            Ok(()) => 0,
            Err(_) => -1,
        }
    })
}

// ============================================================================
// Node
// ============================================================================

#[no_mangle]
pub extern "C" fn as_sync_node_new(node_id: u64) -> *mut AsSyncNode {
    ffi_guard(ptr::null_mut(), || {
        Box::into_raw(Box::new(AsSyncNode(Node::new(NodeId(node_id)))))
    })
}

#[no_mangle]
pub extern "C" fn as_sync_node_with_seed(node_id: u64, seed: u64) -> *mut AsSyncNode {
    ffi_guard(ptr::null_mut(), || {
        Box::into_raw(Box::new(AsSyncNode(Node::with_seed(NodeId(node_id), seed))))
    })
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_node_free(ptr: *mut AsSyncNode) {
    ffi_guard((), || {
        if !ptr.is_null() {
            drop(Box::from_raw(ptr));
        }
    })
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_node_world_hash(ptr: *const AsSyncNode) -> u64 {
    ffi_guard(0, || {
        if ptr.is_null() {
            return 0;
        }
        (*ptr).0.world_hash().0
    })
}

/// `NodeState`: 0=Disconnected, 1=Connecting, 2=Synced, 3=Diverged
#[no_mangle]
pub unsafe extern "C" fn as_sync_node_state(ptr: *const AsSyncNode) -> u8 {
    ffi_guard(0, || {
        if ptr.is_null() {
            return 0;
        }
        match (*ptr).0.state() {
            NodeState::Disconnected => 0,
            NodeState::Connecting => 1,
            NodeState::Synced => 2,
            NodeState::Diverged => 3,
        }
    })
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_node_emit(ptr: *mut AsSyncNode, event: *const AsSyncEvent) -> i32 {
    ffi_guard(-1, || {
        if ptr.is_null() || event.is_null() {
            return -1;
        }
        match (*ptr).0.emit((*event).0.clone()) {
            Ok(()) => 0,
            Err(_) => -1,
        }
    })
}

/// A trust store for `as_sync_node_receive`: the master key of each subject whose devices' events are
/// accepted
pub struct AsSyncTrust(Vec<(String, alice_sign::PublicKey)>);

struct FfiPolicy<'a> {
    trust: &'a AsSyncTrust,
    now: u64,
}

impl crate::authorship::TrustPolicy for FfiPolicy<'_> {
    fn master_of(&self, subject: &alice_identity::did::Did) -> Option<alice_sign::PublicKey> {
        let s = subject.to_string();
        self.trust.0.iter().find(|(k, _)| *k == s).map(|(_, m)| *m)
    }
    fn revocation(&self) -> alice_identity::artifact::Revocation<'_> {
        alice_identity::artifact::Revocation::NoListAvailable
    }
    fn now(&self) -> u64 {
        self.now
    }
}

/// An empty trust store. Free it with `as_sync_trust_free`.
#[no_mangle]
pub extern "C" fn as_sync_trust_new() -> *mut AsSyncTrust {
    ffi_guard(ptr::null_mut(), || {
        Box::into_raw(Box::new(AsSyncTrust(Vec::new())))
    })
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_trust_free(ptr: *mut AsSyncTrust) {
    ffi_guard((), || {
        if !ptr.is_null() {
            drop(Box::from_raw(ptr));
        }
    })
}

/// Trusts the 32-byte master key `master` for the subject `subject` (`subject_len` bytes of UTF-8,
/// `did:method:id`). Returns 0, or -1 for a null pointer or a subject that is not UTF-8.
#[no_mangle]
pub unsafe extern "C" fn as_sync_trust_add(
    ptr: *mut AsSyncTrust,
    subject: *const u8,
    subject_len: u32,
    master: *const u8,
) -> i32 {
    ffi_guard(-1, || {
        if ptr.is_null() || subject.is_null() || master.is_null() {
            return -1;
        }
        let bytes = std::slice::from_raw_parts(subject, subject_len as usize);
        let Ok(text) = std::str::from_utf8(bytes) else {
            return -1;
        };
        let mut key = [0u8; 32];
        key.copy_from_slice(std::slice::from_raw_parts(master, 32));
        (*ptr).0.push((text.to_owned(), alice_sign::PublicKey(key)));
        0
    })
}

/// A node whose events are signed by the device key made from the 32-byte `device_seed`, certified by
/// `cert` (`cert_len` bytes of an encoded ALICE-Identity device certificate naming that key). Null (with
/// `as_sync_last_error`) when the certificate does not decode or does not name the key. Free it with
/// `as_sync_node_free`.
#[no_mangle]
pub unsafe extern "C" fn as_sync_node_with_author(
    node_id: u64,
    device_seed: *const u8,
    cert: *const u8,
    cert_len: u32,
) -> *mut AsSyncNode {
    ffi_guard(ptr::null_mut(), || {
        if device_seed.is_null() || cert.is_null() {
            return ptr::null_mut();
        }
        let mut seed = [0u8; 32];
        seed.copy_from_slice(std::slice::from_raw_parts(device_seed, 32));
        let made = alice_identity::device_cert::DeviceCert::from_bytes(std::slice::from_raw_parts(
            cert,
            cert_len as usize,
        ))
        .map_err(|e| e.to_string())
        .and_then(|c| {
            crate::authorship::Author::new(alice_sign::KeyPair::from_seed(seed), c)
                .map_err(|e| e.to_string())
        });
        match made {
            Ok(author) => Box::into_raw(Box::new(AsSyncNode(Node::with_author(
                NodeId(node_id),
                author,
            )))),
            Err(e) => {
                guard::set_last_error(format!("as_sync_node_with_author: {e}"));
                ptr::null_mut()
            }
        }
    })
}

/// The node's signed events (its own with a checkpoint at the last, and those it accepted from other
/// authors), encoded; their length is written to `out_len`. Null for a null pointer or a node without an
/// author. Free the bytes with `as_sync_bytes_free`.
#[no_mangle]
pub unsafe extern "C" fn as_sync_node_signed_events(
    ptr: *mut AsSyncNode,
    out_len: *mut u32,
) -> *mut u8 {
    ffi_guard(ptr::null_mut(), || {
        if ptr.is_null() || out_len.is_null() {
            return ptr::null_mut();
        }
        match (*ptr)
            .0
            .signed_events_since(&std::collections::BTreeMap::new())
        {
            Ok(batch) => {
                let bytes = bitcode::encode(&batch).into_boxed_slice();
                let Ok(n) = u32::try_from(bytes.len()) else {
                    return ptr::null_mut();
                };
                *out_len = n;
                Box::into_raw(bytes).cast::<u8>()
            }
            Err(e) => {
                guard::set_last_error(format!("as_sync_node_signed_events: {e}"));
                ptr::null_mut()
            }
        }
    })
}

/// Frees bytes returned by `as_sync_node_signed_events` (`len` is the length it wrote).
#[no_mangle]
pub unsafe extern "C" fn as_sync_bytes_free(bytes: *mut u8, len: u32) {
    ffi_guard((), || {
        if !bytes.is_null() {
            drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
                bytes,
                len as usize,
            )));
        }
    })
}

/// Takes encoded signed events from another node (`as_sync_node_signed_events`), trusting `trust` at the
/// time `now`: each author's events apply once its verified checkpoint covers them (ALICE-Law G13: the
/// world is the same for every arrival order and grouping). Returns how many events applied, or (with
/// `as_sync_last_error`) -1 for a null pointer or bytes that do not decode, -2 for an event no checkpoint
/// covers, -3 for a refused checkpoint, -4 for events that do not match the author's checkpoint, -5 for a
/// second history of one author, -6 when the author's held events are full, -7 otherwise.
#[no_mangle]
pub unsafe extern "C" fn as_sync_node_receive(
    ptr: *mut AsSyncNode,
    bytes: *const u8,
    len: u32,
    trust: *const AsSyncTrust,
    now: u64,
) -> i64 {
    use crate::authorship::{AuthorshipError as E, SignedEvents};
    ffi_guard(-1, || {
        if ptr.is_null() || bytes.is_null() || trust.is_null() {
            return -1;
        }
        let Ok(batch) =
            bitcode::decode::<SignedEvents>(std::slice::from_raw_parts(bytes, len as usize))
        else {
            guard::set_last_error("as_sync_node_receive: the bytes do not decode".to_owned());
            return -1;
        };
        let policy = FfiPolicy {
            trust: &*trust,
            now,
        };
        match (*ptr).0.receive(&batch, &policy) {
            Ok(n) => i64::try_from(n).unwrap_or(i64::MAX),
            Err(e) => {
                guard::set_last_error(format!("as_sync_node_receive: {e}"));
                match e {
                    E::Uncovered { .. } => -2,
                    E::BadCheckpoint { .. } => -3,
                    E::ChainMismatch { .. } => -4,
                    E::Forked { .. } => -5,
                    E::PendingFull { .. } => -6,
                    _ => -7,
                }
            }
        }
    })
}

/// A copy of the node's `index`-th emitted event (0 to `as_sync_node_events_count` - 1), stamped; null for a null pointer or an index out of range. Free it with
/// `as_sync_event_free`.
#[no_mangle]
pub unsafe extern "C" fn as_sync_node_event_at(
    ptr: *const AsSyncNode,
    index: u32,
) -> *mut AsSyncEvent {
    ffi_guard(ptr::null_mut(), || {
        if ptr.is_null() {
            return ptr::null_mut();
        }
        match (*ptr).0.events().all().get(index as usize) {
            Some(e) => Box::into_raw(Box::new(AsSyncEvent(e.clone()))),
            None => ptr::null_mut(),
        }
    })
}

/// Adds a peer after checking its 32-byte `SYNC_SEMANTICS_ID` (ALICE-Law G7).
///
/// Returns 0 when the peer's identifier is this build's, 1 when it is a
/// verified earlier one, -1 for a null pointer, and -2 for any other
/// identifier (the peer is not added and `as_sync_last_error` names both).
///
/// # Safety
/// `ptr` is a node handle and `peer_semantics_id` points to 32 readable bytes.
#[no_mangle]
pub unsafe extern "C" fn as_sync_node_add_peer(
    ptr: *mut AsSyncNode,
    peer_id: u64,
    peer_semantics_id: *const u8,
) -> i32 {
    ffi_guard(-1, || {
        if ptr.is_null() {
            return -1;
        }
        match handshake(peer_semantics_id) {
            Ok(status) => {
                (*ptr).0.add_peer(NodeId(peer_id));
                status
            }
            Err(code) => code,
        }
    })
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_node_entity_count(ptr: *const AsSyncNode) -> u32 {
    ffi_guard(0, || {
        if ptr.is_null() {
            return 0;
        }
        (*ptr).0.world().entity_count() as u32
    })
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_node_events_count(ptr: *const AsSyncNode) -> u32 {
    ffi_guard(0, || {
        if ptr.is_null() {
            return 0;
        }
        (*ptr).0.events().len() as u32
    })
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_node_events_bytes(ptr: *const AsSyncNode) -> u32 {
    ffi_guard(0, || {
        if ptr.is_null() {
            return 0;
        }
        (*ptr).0.events().total_bytes() as u32
    })
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
    ffi_guard(ptr::null_mut(), || {
        Box::into_raw(Box::new(AsSyncEvent(Event::new(EventKind::Motion {
            entity,
            delta: [dx, dy, dz],
        }))))
    })
}

#[no_mangle]
pub extern "C" fn as_sync_event_new_spawn(
    entity: u32,
    kind: u16,
    px: i16,
    py: i16,
    pz: i16,
) -> *mut AsSyncEvent {
    ffi_guard(ptr::null_mut(), || {
        Box::into_raw(Box::new(AsSyncEvent(Event::new(EventKind::Spawn {
            entity,
            kind,
            pos: [px, py, pz],
        }))))
    })
}

#[no_mangle]
pub extern "C" fn as_sync_event_new_despawn(entity: u32) -> *mut AsSyncEvent {
    ffi_guard(ptr::null_mut(), || {
        Box::into_raw(Box::new(AsSyncEvent(Event::new(EventKind::Despawn {
            entity,
        }))))
    })
}

#[no_mangle]
pub extern "C" fn as_sync_event_new_property(
    entity: u32,
    prop: u16,
    value: i32,
) -> *mut AsSyncEvent {
    ffi_guard(ptr::null_mut(), || {
        Box::into_raw(Box::new(AsSyncEvent(Event::new(EventKind::Property {
            entity,
            prop,
            value,
        }))))
    })
}

#[no_mangle]
pub extern "C" fn as_sync_event_new_input(player: u16, code: u32) -> *mut AsSyncEvent {
    ffi_guard(ptr::null_mut(), || {
        Box::into_raw(Box::new(AsSyncEvent(Event::new(EventKind::Input {
            player,
            code,
        }))))
    })
}

#[no_mangle]
pub extern "C" fn as_sync_event_new_tick(frame: u64) -> *mut AsSyncEvent {
    ffi_guard(ptr::null_mut(), || {
        Box::into_raw(Box::new(AsSyncEvent(Event::new(EventKind::Tick { frame }))))
    })
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_event_free(ptr: *mut AsSyncEvent) {
    ffi_guard((), || {
        if !ptr.is_null() {
            drop(Box::from_raw(ptr));
        }
    })
}

/// Writes the event's 16-byte origin (its author's, zero for an unstamped event) to `out`. Returns 0, or -1
/// for a null pointer.
#[no_mangle]
pub unsafe extern "C" fn as_sync_event_origin(ptr: *const AsSyncEvent, out: *mut u8) -> i32 {
    ffi_guard(-1, || {
        if ptr.is_null() || out.is_null() {
            return -1;
        }
        std::ptr::copy_nonoverlapping((*ptr).0.origin.0.as_ptr(), out, 16);
        0
    })
}

/// The event's sequence number in its origin's log, from 1 (0 for an unstamped event or null).
#[no_mangle]
pub unsafe extern "C" fn as_sync_event_seq(ptr: *const AsSyncEvent) -> u64 {
    ffi_guard(0, || if ptr.is_null() { 0 } else { (*ptr).0.seq.0 })
}

/// The event's Lamport time (0 for an unstamped event or null).
#[no_mangle]
pub unsafe extern "C" fn as_sync_event_timestamp(ptr: *const AsSyncEvent) -> u64 {
    ffi_guard(0, || if ptr.is_null() { 0 } else { (*ptr).0.timestamp })
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_event_size_bytes(ptr: *const AsSyncEvent) -> u32 {
    ffi_guard(0, || {
        if ptr.is_null() {
            return 0;
        }
        (*ptr).0.size_bytes() as u32
    })
}

// ============================================================================
// EventStream
// ============================================================================

#[no_mangle]
pub extern "C" fn as_sync_event_stream_new() -> *mut AsSyncEventStream {
    ffi_guard(ptr::null_mut(), || {
        Box::into_raw(Box::new(AsSyncEventStream(EventStream::new())))
    })
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_event_stream_free(ptr: *mut AsSyncEventStream) {
    ffi_guard((), || {
        if !ptr.is_null() {
            drop(Box::from_raw(ptr));
        }
    })
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_event_stream_push(
    ptr: *mut AsSyncEventStream,
    event: *const AsSyncEvent,
    origin: *const u8,
) -> u64 {
    ffi_guard(0, || {
        if ptr.is_null() || event.is_null() || origin.is_null() {
            return 0;
        }
        let mut o = [0u8; 16];
        o.copy_from_slice(std::slice::from_raw_parts(origin, 16));
        (*ptr).0.push((*event).0.clone(), crate::event::Origin(o)).0
    })
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_event_stream_len(ptr: *const AsSyncEventStream) -> u32 {
    ffi_guard(0, || {
        if ptr.is_null() {
            return 0;
        }
        (*ptr).0.len() as u32
    })
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_event_stream_is_empty(ptr: *const AsSyncEventStream) -> i32 {
    ffi_guard(1, || {
        if ptr.is_null() {
            return 1;
        }
        i32::from((*ptr).0.is_empty())
    })
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_event_stream_current_seq(ptr: *const AsSyncEventStream) -> u64 {
    ffi_guard(0, || {
        if ptr.is_null() {
            return 0;
        }
        (*ptr).0.current_seq().0
    })
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_event_stream_total_bytes(ptr: *const AsSyncEventStream) -> u32 {
    ffi_guard(0, || {
        if ptr.is_null() {
            return 0;
        }
        (*ptr).0.total_bytes() as u32
    })
}

// ============================================================================
// InputFrame
// ============================================================================

#[no_mangle]
pub extern "C" fn as_sync_input_frame_new(frame: u64, player_id: u8) -> *mut AsSyncInputFrame {
    ffi_guard(ptr::null_mut(), || {
        Box::into_raw(Box::new(AsSyncInputFrame(InputFrame::new(
            frame, player_id,
        ))))
    })
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_input_frame_free(ptr: *mut AsSyncInputFrame) {
    ffi_guard((), || {
        if !ptr.is_null() {
            drop(Box::from_raw(ptr));
        }
    })
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_input_frame_set_movement(
    ptr: *mut AsSyncInputFrame,
    x: i16,
    y: i16,
    z: i16,
) {
    ffi_guard((), || {
        if ptr.is_null() {
            return;
        }
        (*ptr).0.movement = [x, y, z];
    })
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_input_frame_set_actions(ptr: *mut AsSyncInputFrame, actions: u32) {
    ffi_guard((), || {
        if ptr.is_null() {
            return;
        }
        (*ptr).0.actions = actions;
    })
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_input_frame_set_aim(
    ptr: *mut AsSyncInputFrame,
    x: i16,
    y: i16,
    z: i16,
) {
    ffi_guard((), || {
        if ptr.is_null() {
            return;
        }
        (*ptr).0.aim = [x, y, z];
    })
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_input_frame_get_frame(ptr: *const AsSyncInputFrame) -> u64 {
    ffi_guard(0, || {
        if ptr.is_null() {
            return 0;
        }
        (*ptr).0.frame
    })
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_input_frame_get_player_id(ptr: *const AsSyncInputFrame) -> u8 {
    ffi_guard(0, || {
        if ptr.is_null() {
            return 0;
        }
        (*ptr).0.player_id
    })
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_input_frame_get_movement(
    ptr: *const AsSyncInputFrame,
    out_x: *mut i16,
    out_y: *mut i16,
    out_z: *mut i16,
) {
    ffi_guard((), || {
        if ptr.is_null() || out_x.is_null() || out_y.is_null() || out_z.is_null() {
            return;
        }
        *out_x = (*ptr).0.movement[0];
        *out_y = (*ptr).0.movement[1];
        *out_z = (*ptr).0.movement[2];
    })
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_input_frame_get_actions(ptr: *const AsSyncInputFrame) -> u32 {
    ffi_guard(0, || {
        if ptr.is_null() {
            return 0;
        }
        (*ptr).0.actions
    })
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_input_frame_get_aim(
    ptr: *const AsSyncInputFrame,
    out_x: *mut i16,
    out_y: *mut i16,
    out_z: *mut i16,
) {
    ffi_guard((), || {
        if ptr.is_null() || out_x.is_null() || out_y.is_null() || out_z.is_null() {
            return;
        }
        *out_x = (*ptr).0.aim[0];
        *out_y = (*ptr).0.aim[1];
        *out_z = (*ptr).0.aim[2];
    })
}

// ============================================================================
// LockstepSession
// ============================================================================

/// Creates a lockstep session with peers whose 32-byte `SYNC_SEMANTICS_ID` is
/// `peer_semantics_id` (ALICE-Law G7): null, with `as_sync_last_error`
/// naming both identifiers, when it is neither this build's nor a verified
/// earlier one, or when the pointer is null.
///
/// # Safety
/// `peer_semantics_id` points to 32 readable bytes.
#[no_mangle]
pub unsafe extern "C" fn as_sync_lockstep_new(
    player_count: u8,
    peer_semantics_id: *const u8,
) -> *mut AsSyncLockstep {
    ffi_guard(ptr::null_mut(), || {
        if handshake(peer_semantics_id).is_err() {
            return ptr::null_mut();
        }
        Box::into_raw(Box::new(AsSyncLockstep(LockstepSession::new(player_count))))
    })
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_lockstep_free(ptr: *mut AsSyncLockstep) {
    ffi_guard((), || {
        if !ptr.is_null() {
            drop(Box::from_raw(ptr));
        }
    })
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_lockstep_add_local_input(
    ptr: *mut AsSyncLockstep,
    input: *const AsSyncInputFrame,
) {
    ffi_guard((), || {
        if ptr.is_null() || input.is_null() {
            return;
        }
        (*ptr).0.add_local_input((*input).0);
    })
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_lockstep_add_remote_input(
    ptr: *mut AsSyncLockstep,
    input: *const AsSyncInputFrame,
) {
    ffi_guard((), || {
        if ptr.is_null() || input.is_null() {
            return;
        }
        (*ptr).0.add_remote_input((*input).0);
    })
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_lockstep_ready(ptr: *const AsSyncLockstep) -> i32 {
    ffi_guard(0, || {
        if ptr.is_null() {
            return 0;
        }
        i32::from((*ptr).0.ready_to_advance())
    })
}

/// advance成功時: `InputFrameArray`へのポインタ（`as_sync_input_frame_array_free`で解放）
/// 未ready時: null
#[no_mangle]
pub unsafe extern "C" fn as_sync_lockstep_advance(
    ptr: *mut AsSyncLockstep,
) -> *mut AsSyncInputFrameArray {
    ffi_guard(ptr::null_mut(), || {
        if ptr.is_null() {
            return ptr::null_mut();
        }
        (*ptr).0.advance().map_or(ptr::null_mut(), |inputs| {
            Box::into_raw(Box::new(AsSyncInputFrameArray(inputs)))
        })
    })
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_lockstep_confirmed_frame(ptr: *const AsSyncLockstep) -> u64 {
    ffi_guard(0, || {
        if ptr.is_null() {
            return 0;
        }
        (*ptr).0.confirmed_frame()
    })
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_lockstep_record_checksum(
    ptr: *mut AsSyncLockstep,
    frame: u64,
    checksum: u64,
) {
    ffi_guard((), || {
        if ptr.is_null() {
            return;
        }
        (*ptr).0.record_checksum(frame, checksum);
    })
}

// ============================================================================
// RollbackSession
// ============================================================================

/// Creates a rollback session with peers whose 32-byte `SYNC_SEMANTICS_ID` is
/// `peer_semantics_id` (ALICE-Law G7): null, with `as_sync_last_error`
/// naming both identifiers, when it is neither this build's nor a verified
/// earlier one, or when the pointer is null.
///
/// # Safety
/// `peer_semantics_id` points to 32 readable bytes.
#[no_mangle]
pub unsafe extern "C" fn as_sync_rollback_new(
    player_count: u8,
    local_player: u8,
    max_rollback: u64,
    peer_semantics_id: *const u8,
) -> *mut AsSyncRollback {
    ffi_guard(ptr::null_mut(), || {
        if handshake(peer_semantics_id).is_err() {
            return ptr::null_mut();
        }
        Box::into_raw(Box::new(AsSyncRollback(RollbackSession::new(
            player_count,
            local_player,
            max_rollback,
        ))))
    })
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_rollback_free(ptr: *mut AsSyncRollback) {
    ffi_guard((), || {
        if !ptr.is_null() {
            drop(Box::from_raw(ptr));
        }
    })
}

/// ローカル入力追加 → 全プレイヤーのフレーム入力を返す
#[no_mangle]
pub unsafe extern "C" fn as_sync_rollback_add_local_input(
    ptr: *mut AsSyncRollback,
    input: *const AsSyncInputFrame,
) -> *mut AsSyncInputFrameArray {
    ffi_guard(ptr::null_mut(), || {
        if ptr.is_null() || input.is_null() {
            return ptr::null_mut();
        }
        let inputs = (*ptr).0.add_local_input((*input).0);
        Box::into_raw(Box::new(AsSyncInputFrameArray(inputs)))
    })
}

/// リモート入力追加 → `RollbackAction`: 0=None, 1=Rollback, 2=Desync
/// `rollback_frame`: Rollback/Desyncの場合のフレーム番号（out）
#[no_mangle]
pub unsafe extern "C" fn as_sync_rollback_add_remote_input(
    ptr: *mut AsSyncRollback,
    input: *const AsSyncInputFrame,
    out_rollback_frame: *mut u64,
) -> u8 {
    ffi_guard(0, || {
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
    })
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_rollback_confirmed_frame(ptr: *const AsSyncRollback) -> u64 {
    ffi_guard(0, || {
        if ptr.is_null() {
            return 0;
        }
        (*ptr).0.confirmed_frame()
    })
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_rollback_predicted_frame(ptr: *const AsSyncRollback) -> u64 {
    ffi_guard(0, || {
        if ptr.is_null() {
            return 0;
        }
        (*ptr).0.predicted_frame()
    })
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_rollback_frames_ahead(ptr: *const AsSyncRollback) -> u64 {
    ffi_guard(0, || {
        if ptr.is_null() {
            return 0;
        }
        (*ptr).0.frames_ahead()
    })
}

// ============================================================================
// InputFrameArray (for lockstep/rollback results)
// ============================================================================

#[no_mangle]
pub unsafe extern "C" fn as_sync_input_frame_array_free(ptr: *mut AsSyncInputFrameArray) {
    ffi_guard((), || {
        if !ptr.is_null() {
            drop(Box::from_raw(ptr));
        }
    })
}

#[no_mangle]
pub unsafe extern "C" fn as_sync_input_frame_array_len(ptr: *const AsSyncInputFrameArray) -> u32 {
    ffi_guard(0, || {
        if ptr.is_null() {
            return 0;
        }
        (*ptr).0.len() as u32
    })
}

/// 配列のi番目の`InputFrame`をコピーして返す。解放は`as_sync_input_frame_free`
#[no_mangle]
pub unsafe extern "C" fn as_sync_input_frame_array_get(
    ptr: *const AsSyncInputFrameArray,
    index: u32,
) -> *mut AsSyncInputFrame {
    ffi_guard(ptr::null_mut(), || {
        if ptr.is_null() {
            return ptr::null_mut();
        }
        let arr = &(*ptr).0;
        arr.get(index as usize).map_or(ptr::null_mut(), |f| {
            Box::into_raw(Box::new(AsSyncInputFrame(*f)))
        })
    })
}

// ============================================================================
// Fixed-point utilities
// ============================================================================

#[no_mangle]
pub extern "C" fn as_sync_fixed_from_f32(f: f32) -> i32 {
    ffi_guard(0, || Fixed::from_f32(f).0)
}

#[no_mangle]
pub extern "C" fn as_sync_fixed_to_f32(bits: i32) -> f32 {
    ffi_guard(f32::NAN, || Fixed::from_bits(bits).to_f32())
}

#[no_mangle]
pub extern "C" fn as_sync_vec3_hash(x: f32, y: f32, z: f32) -> u64 {
    ffi_guard(0, || Vec3Fixed::from_f32(x, y, z).hash_bits())
}

// ============================================================================
// Version
// ============================================================================

#[no_mangle]
pub extern "C" fn as_sync_version() -> u32 {
    // major * 10000 + minor * 100 + patch, from Cargo.toml (0.7.0 -> 700)
    ffi_guard(0, || VERSION_CODE)
}

const fn parse_u32(s: &str) -> u32 {
    let b = s.as_bytes();
    let mut v = 0u32;
    let mut i = 0;
    while i < b.len() {
        v = v * 10 + (b[i] - b'0') as u32;
        i += 1;
    }
    v
}

const VERSION_CODE: u32 = parse_u32(env!("CARGO_PKG_VERSION_MAJOR")) * 10000
    + parse_u32(env!("CARGO_PKG_VERSION_MINOR")) * 100
    + parse_u32(env!("CARGO_PKG_VERSION_PATCH"));

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

    /// A device certificate (layout version 2) and the seed of the device key it names
    fn ffi_party(seed: u8, name: &str) -> (alice_sign::KeyPair, Vec<u8>, [u8; 32]) {
        let master = alice_sign::KeyPair::from_seed([seed; 32]);
        let device_seed = [seed.wrapping_add(100); 32];
        let device = alice_sign::KeyPair::from_seed(device_seed);
        let cert = alice_identity::device_cert::DeviceCert::issue(
            &master,
            &alice_identity::did::Did::parse(&format!("did:alice:{name}")).unwrap(),
            device.public().0,
            [0x4e; 32],
            1_800_000_000,
            1_800_086_400,
            [seed; 16],
        )
        .unwrap();
        (master, cert.to_bytes(), device_seed)
    }

    #[test]
    fn test_ffi_node_sync() {
        unsafe {
            let (master_a, cert_a, seed_a) = ffi_party(1, "alice");
            let (_, cert_b, seed_b) = ffi_party(2, "bob");
            let node_a = as_sync_node_with_author(
                1,
                seed_a.as_ptr(),
                cert_a.as_ptr(),
                u32::try_from(cert_a.len()).unwrap(),
            );
            let node_b = as_sync_node_with_author(
                2,
                seed_b.as_ptr(),
                cert_b.as_ptr(),
                u32::try_from(cert_b.len()).unwrap(),
            );
            assert!(!node_a.is_null() && !node_b.is_null());
            // a certificate that does not name the key is refused
            let wrong = as_sync_node_with_author(
                3,
                seed_b.as_ptr(),
                cert_a.as_ptr(),
                u32::try_from(cert_a.len()).unwrap(),
            );
            assert!(wrong.is_null());

            let made = [
                as_sync_event_new_spawn(1, 0, 0, 0, 0),
                as_sync_event_new_motion(1, 10, 0, 0),
                as_sync_event_new_motion(1, 0, 20, 0),
                as_sync_event_new_motion(1, 0, 0, 30),
            ];
            for e in made {
                assert_eq!(as_sync_node_emit(node_a, e), 0);
                as_sync_event_free(e);
            }
            assert_eq!(as_sync_node_events_count(node_a), 4);
            assert!(as_sync_node_event_at(node_a, 4).is_null());
            let first = as_sync_node_event_at(node_a, 0);
            let mut origin = [0u8; 16];
            assert_eq!(as_sync_event_origin(first, origin.as_mut_ptr()), 0);
            assert_ne!(origin, [0u8; 16]);
            assert_eq!(as_sync_event_seq(first), 1);
            as_sync_event_free(first);

            let mut len = 0u32;
            let bytes = as_sync_node_signed_events(node_a, &raw mut len);
            assert!(!bytes.is_null() && len > 0);
            let trust = as_sync_trust_new();
            let subject = b"did:alice:alice";
            // not trusted yet: the checkpoint is refused and nothing applies
            let before = as_sync_node_world_hash(node_b);
            assert_eq!(
                as_sync_node_receive(node_b, bytes, len, trust, 1_800_000_100),
                -3
            );
            assert_eq!(as_sync_node_world_hash(node_b), before);
            assert_eq!(
                as_sync_trust_add(trust, subject.as_ptr(), 15, master_a.public().0.as_ptr()),
                0
            );
            assert_eq!(
                as_sync_node_receive(node_b, bytes, len, trust, 1_800_000_100),
                4
            );
            assert_eq!(
                as_sync_node_receive(node_b, bytes, len, trust, 1_800_000_100),
                0,
                "a duplicate changes nothing"
            );
            assert_eq!(
                as_sync_node_world_hash(node_a),
                as_sync_node_world_hash(node_b)
            );
            assert_eq!(as_sync_node_entity_count(node_b), 1);
            // bytes that do not decode are refused
            assert_eq!(
                as_sync_node_receive(node_b, bytes, 3, trust, 1_800_000_100),
                -1
            );
            as_sync_bytes_free(bytes, len);

            // a node without an author has no signed events
            let plain = as_sync_node_new(9);
            assert!(as_sync_node_signed_events(plain, &raw mut len).is_null());
            as_sync_node_free(plain);

            as_sync_trust_free(trust);
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
            let session = as_sync_lockstep_new(2, crate::SYNC_SEMANTICS_ID.as_ptr());
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
            let session = as_sync_rollback_new(2, 0, 8, crate::SYNC_SEMANTICS_ID.as_ptr());
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
        assert_eq!(as_sync_version(), 700);
    }

    #[test]
    fn test_ffi_event_stream() {
        unsafe {
            let stream = as_sync_event_stream_new();
            assert_eq!(as_sync_event_stream_len(stream), 0);
            assert_eq!(as_sync_event_stream_is_empty(stream), 1);

            let event = as_sync_event_new_motion(1, 10, 20, 30);
            let origin = [42u8; 16];
            as_sync_event_stream_push(stream, event, origin.as_ptr());
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

// ============================================================================
// Panic isolation (see `guard`)
// ============================================================================

/// Message of the most recent panic caught at the FFI boundary on this thread
/// (NUL-terminated, owned by the callee — release with
/// `as_sync_free_error_string`), or null if none.
#[no_mangle]
pub extern "C" fn as_sync_last_error() -> *mut std::os::raw::c_char {
    ffi_guard(ptr::null_mut(), || match guard::take_last_error() {
        Some(msg) => {
            std::ffi::CString::new(msg).map_or(ptr::null_mut(), std::ffi::CString::into_raw)
        }
        None => ptr::null_mut(),
    })
}

/// Clear the most recent FFI error message.
#[no_mangle]
pub extern "C" fn as_sync_clear_last_error() {
    ffi_guard((), guard::clear_last_error);
}

/// Release a string returned by `as_sync_last_error`.
///
/// # Safety
/// `s` must be null or a pointer returned by `as_sync_last_error` (freed once).
#[no_mangle]
pub unsafe extern "C" fn as_sync_free_error_string(s: *mut std::os::raw::c_char) {
    ffi_guard((), || {
        if !s.is_null() {
            drop(std::ffi::CString::from_raw(s));
        }
    });
}

/// Panic isolation for the C ABI.
///
/// A panic that reaches an `extern "C"` boundary aborts the whole process
/// (Rust 1.81+), taking the host (Unity, Unreal, a Python interpreter) down
/// with it. Every exported function therefore runs its body through
/// [`guard::ffi_guard`]: a panic is caught inside the function, its message is
/// stored in a thread-local slot the host reads with `as_sync_last_error`, and
/// the function returns the caller's sentinel (null handle, 0, -1, NaN, ()).
/// Requires the crate to be built with `panic = "unwind"` (the default);
/// `panic = "abort"` makes `catch_unwind` a no-op.
mod guard {
    use std::cell::RefCell;
    use std::panic::{catch_unwind, AssertUnwindSafe};

    thread_local! {
        static LAST_ERROR: RefCell<Option<String>> = const { RefCell::new(None) };
    }

    /// Record an error message for `as_sync_last_error`.
    pub fn set_last_error(msg: impl Into<String>) {
        LAST_ERROR.with(|slot| *slot.borrow_mut() = Some(msg.into()));
    }

    /// Take the most recent error message (leaves the slot empty).
    pub fn take_last_error() -> Option<String> {
        LAST_ERROR.with(|slot| slot.borrow_mut().take())
    }

    /// Clear the most recent error message.
    pub fn clear_last_error() {
        LAST_ERROR.with(|slot| *slot.borrow_mut() = None);
    }

    /// Run `body`, converting a panic into `default` plus a recorded message.
    ///
    /// The closure is treated as unwind-safe: every FFI body only touches its
    /// arguments and heap handles owned by the caller, so no partially-updated
    /// shared state is observable afterwards.
    #[inline]
    pub fn ffi_guard<T>(default: T, body: impl FnOnce() -> T) -> T {
        match catch_unwind(AssertUnwindSafe(body)) {
            Ok(v) => v,
            Err(payload) => {
                let msg = payload
                    .downcast_ref::<&str>()
                    .map(|s| (*s).to_string())
                    .or_else(|| payload.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "panic with non-string payload".to_string());
                set_last_error(format!("as_sync FFI panic: {msg}"));
                default
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn panic_becomes_default_and_message() {
            clear_last_error();
            let v = ffi_guard(-1i32, || -> i32 { panic!("boom {}", 42) });
            assert_eq!(v, -1);
            let msg = take_last_error().expect("message recorded");
            assert!(msg.contains("boom 42"), "{msg}");
            assert!(take_last_error().is_none(), "take clears the slot");
        }

        #[test]
        fn ok_call_records_no_error() {
            clear_last_error();
            assert_eq!(ffi_guard(0, || 5), 5);
            assert!(take_last_error().is_none());
        }
    }
}

use guard::ffi_guard;

/// Writes this build's 32-byte `SYNC_SEMANTICS_ID` to `out` (ALICE-Law G7):
/// the value a host sends to its peers and passes to their
/// `as_sync_*_new` / `as_sync_node_add_peer`. Returns 0, or -1 when `out` is
/// null or `len` is below 32.
///
/// # Safety
/// `out` points to `len` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn as_sync_semantics_id(out: *mut u8, len: usize) -> i32 {
    ffi_guard(-1, || {
        if out.is_null() || len < 32 {
            return -1;
        }
        ptr::copy_nonoverlapping(crate::SYNC_SEMANTICS_ID.as_ptr(), out, 32);
        0
    })
}

/// Reads a peer's 32-byte identifier and checks it: `Ok(0)` this build's,
/// `Ok(1)` a verified earlier one; `Err(-1)` null, `Err(-2)` any other (with
/// `as_sync_last_error` naming both identifiers).
unsafe fn handshake(peer_semantics_id: *const u8) -> Result<i32, i32> {
    if peer_semantics_id.is_null() {
        guard::set_last_error("as_sync: peer semantics identifier is null");
        return Err(-1);
    }
    let mut id = [0u8; 32];
    ptr::copy_nonoverlapping(peer_semantics_id, id.as_mut_ptr(), 32);
    match crate::semantics::check_semantics(&id) {
        Ok(crate::semantics::IdStatus::Current) => Ok(0),
        Ok(crate::semantics::IdStatus::VerifiedPrevious(_)) => Ok(1),
        Err(m) => {
            guard::set_last_error(format!("as_sync: {m}"));
            Err(-2)
        }
    }
}

#[cfg(test)]
mod handshake_ffi_tests {
    use super::*;

    fn foreign() -> [u8; 32] {
        let mut id = crate::SYNC_SEMANTICS_ID;
        id[5] ^= 0x10;
        id
    }

    fn last_error() -> String {
        let s = as_sync_last_error();
        assert!(!s.is_null(), "an error message is recorded");
        let msg = unsafe { std::ffi::CStr::from_ptr(s) }
            .to_string_lossy()
            .into_owned();
        unsafe { as_sync_free_error_string(s) };
        msg
    }

    fn hex(b: &[u8]) -> String {
        b.iter().map(|x| format!("{x:02x}")).collect()
    }

    fn assert_names_both(msg: &str) {
        assert!(msg.contains(&hex(&foreign())), "{msg}");
        assert!(msg.contains(&hex(&crate::SYNC_SEMANTICS_ID)), "{msg}");
    }

    #[test]
    fn semantics_id_is_written_and_short_buffers_are_refused() {
        let mut out = [0u8; 32];
        assert_eq!(unsafe { as_sync_semantics_id(out.as_mut_ptr(), 32) }, 0);
        assert_eq!(out, crate::SYNC_SEMANTICS_ID);
        assert_eq!(unsafe { as_sync_semantics_id(out.as_mut_ptr(), 31) }, -1);
        assert_eq!(unsafe { as_sync_semantics_id(ptr::null_mut(), 32) }, -1);
    }

    #[test]
    fn lockstep_new_refuses_another_identifier() {
        as_sync_clear_last_error();
        let own = crate::SYNC_SEMANTICS_ID;
        let ok = unsafe { as_sync_lockstep_new(2, own.as_ptr()) };
        assert!(!ok.is_null());
        unsafe { as_sync_lockstep_free(ok) };
        let bad = foreign();
        assert!(unsafe { as_sync_lockstep_new(2, bad.as_ptr()) }.is_null());
        assert_names_both(&last_error());
        assert!(unsafe { as_sync_lockstep_new(2, ptr::null()) }.is_null());
    }

    #[test]
    fn rollback_new_refuses_another_identifier() {
        as_sync_clear_last_error();
        let own = crate::SYNC_SEMANTICS_ID;
        let ok = unsafe { as_sync_rollback_new(2, 0, 8, own.as_ptr()) };
        assert!(!ok.is_null());
        unsafe { as_sync_rollback_free(ok) };
        let bad = foreign();
        assert!(unsafe { as_sync_rollback_new(2, 0, 8, bad.as_ptr()) }.is_null());
        assert_names_both(&last_error());
    }

    #[test]
    fn node_add_peer_refuses_another_identifier() {
        as_sync_clear_last_error();
        let node = as_sync_node_new(1);
        let own = crate::SYNC_SEMANTICS_ID;
        let bad = foreign();
        unsafe {
            assert_eq!(as_sync_node_add_peer(node, 2, own.as_ptr()), 0);
            assert_eq!((*node).0.peers().count(), 1);
            assert_eq!(as_sync_node_add_peer(node, 3, bad.as_ptr()), -2);
            assert_names_both(&last_error());
            assert_eq!(
                (*node).0.peers().count(),
                1,
                "the refused peer is not added"
            );
            assert_eq!(as_sync_node_add_peer(node, 4, ptr::null()), -1);
            assert_eq!(as_sync_node_add_peer(ptr::null_mut(), 4, own.as_ptr()), -1);
            as_sync_node_free(node);
        }
    }
}

#[cfg(test)]
mod guard_ffi_tests {
    use super::*;

    /// panic が sentinel + `as_sync_last_error` の message に変換され、host が
    /// 文字列を取得 / 解放できること (extern "C" 3 本の end-to-end)
    #[test]
    fn last_error_roundtrip_through_c_abi() {
        as_sync_clear_last_error();
        assert!(as_sync_last_error().is_null(), "初期状態は null");
        let v = ffi_guard(-1i32, || -> i32 { panic!("ffi test panic") });
        assert_eq!(v, -1);
        let s = as_sync_last_error();
        assert!(!s.is_null());
        let msg = unsafe { std::ffi::CStr::from_ptr(s) }
            .to_string_lossy()
            .into_owned();
        assert!(msg.contains("ffi test panic"), "{msg}");
        unsafe { as_sync_free_error_string(s) };
        assert!(as_sync_last_error().is_null(), "take で slot は空になる");
    }

    /// 無効 handle は panic せず sentinel を返す (guard 導入で挙動が変わらないこと)
    #[test]
    fn null_handles_return_sentinels() {
        unsafe {
            assert_eq!(as_sync_world_frame(ptr::null()), 0);
            assert_eq!(as_sync_world_apply_event(ptr::null_mut(), ptr::null()), -1);
            assert_eq!(as_sync_event_stream_is_empty(ptr::null()), 1);
        }
        assert!(
            as_sync_last_error().is_null(),
            "sentinel 経路は error を記録しない"
        );
    }
}

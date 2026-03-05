// ALICE-Sync UE5 C++ Bindings — 71 extern "C" declarations + RAII handles
// Auto-generated from src/ffi.rs
// Copyright (C) 2026 Moroya Sakamoto — AGPL-3.0

#pragma once

#include <cstdint>
#include <memory>

// ============================================================================
// Opaque handle forward declarations
// ============================================================================

struct AsSyncWorld;
struct AsSyncWorldSoA;
struct AsSyncNode;
struct AsSyncEventStream;
struct AsSyncEvent;
struct AsSyncInputFrame;
struct AsSyncLockstep;
struct AsSyncRollback;
struct AsSyncInputFrameArray;

// ============================================================================
// C-ABI declarations (65 functions)
// ============================================================================

extern "C"
{
    // World (AoS)
    AsSyncWorld* as_sync_world_new(uint64_t seed);
    void as_sync_world_free(AsSyncWorld* ptr);
    uint64_t as_sync_world_hash(const AsSyncWorld* ptr);
    uint32_t as_sync_world_entity_count(const AsSyncWorld* ptr);
    uint64_t as_sync_world_frame(const AsSyncWorld* ptr);
    int32_t as_sync_world_apply_event(AsSyncWorld* ptr, const AsSyncEvent* evt);
    int32_t as_sync_world_get_entity_position(const AsSyncWorld* ptr, uint32_t entityId, float* outX, float* outY, float* outZ);
    uint64_t as_sync_world_recalculate_hash(AsSyncWorld* ptr);

    // WorldSoA
    AsSyncWorldSoA* as_sync_world_soa_new(uint64_t seed);
    void as_sync_world_soa_free(AsSyncWorldSoA* ptr);
    uint64_t as_sync_world_soa_hash(const AsSyncWorldSoA* ptr);
    uint32_t as_sync_world_soa_entity_count(const AsSyncWorldSoA* ptr);
    int32_t as_sync_world_soa_apply_event(AsSyncWorldSoA* ptr, const AsSyncEvent* evt);

    // Node
    AsSyncNode* as_sync_node_new(uint64_t nodeId);
    AsSyncNode* as_sync_node_with_seed(uint64_t nodeId, uint64_t seed);
    void as_sync_node_free(AsSyncNode* ptr);
    uint64_t as_sync_node_world_hash(const AsSyncNode* ptr);
    uint8_t as_sync_node_state(const AsSyncNode* ptr);
    int32_t as_sync_node_emit(AsSyncNode* ptr, const AsSyncEvent* evt);
    int32_t as_sync_node_apply_event(AsSyncNode* ptr, const AsSyncEvent* evt);
    void as_sync_node_add_peer(AsSyncNode* ptr, uint64_t peerId);
    uint32_t as_sync_node_entity_count(const AsSyncNode* ptr);
    uint32_t as_sync_node_events_count(const AsSyncNode* ptr);
    uint32_t as_sync_node_events_bytes(const AsSyncNode* ptr);

    // Event construction
    AsSyncEvent* as_sync_event_new_motion(uint32_t entity, int16_t dx, int16_t dy, int16_t dz);
    AsSyncEvent* as_sync_event_new_spawn(uint32_t entity, uint16_t kind, int16_t px, int16_t py, int16_t pz);
    AsSyncEvent* as_sync_event_new_despawn(uint32_t entity);
    AsSyncEvent* as_sync_event_new_property(uint32_t entity, uint16_t prop, int32_t value);
    AsSyncEvent* as_sync_event_new_input(uint16_t player, uint32_t code);
    AsSyncEvent* as_sync_event_new_tick(uint64_t frame);
    void as_sync_event_free(AsSyncEvent* ptr);
    uint32_t as_sync_event_size_bytes(const AsSyncEvent* ptr);

    // EventStream
    AsSyncEventStream* as_sync_event_stream_new();
    void as_sync_event_stream_free(AsSyncEventStream* ptr);
    uint64_t as_sync_event_stream_push(AsSyncEventStream* ptr, const AsSyncEvent* evt, uint64_t origin);
    uint32_t as_sync_event_stream_len(const AsSyncEventStream* ptr);
    int32_t as_sync_event_stream_is_empty(const AsSyncEventStream* ptr);
    uint64_t as_sync_event_stream_current_seq(const AsSyncEventStream* ptr);
    uint32_t as_sync_event_stream_total_bytes(const AsSyncEventStream* ptr);

    // InputFrame
    AsSyncInputFrame* as_sync_input_frame_new(uint64_t frame, uint8_t playerId);
    void as_sync_input_frame_free(AsSyncInputFrame* ptr);
    void as_sync_input_frame_set_movement(AsSyncInputFrame* ptr, int16_t x, int16_t y, int16_t z);
    void as_sync_input_frame_set_actions(AsSyncInputFrame* ptr, uint32_t actions);
    void as_sync_input_frame_set_aim(AsSyncInputFrame* ptr, int16_t x, int16_t y, int16_t z);
    uint64_t as_sync_input_frame_get_frame(const AsSyncInputFrame* ptr);
    uint8_t as_sync_input_frame_get_player_id(const AsSyncInputFrame* ptr);
    void as_sync_input_frame_get_movement(const AsSyncInputFrame* ptr, int16_t* outX, int16_t* outY, int16_t* outZ);
    uint32_t as_sync_input_frame_get_actions(const AsSyncInputFrame* ptr);
    void as_sync_input_frame_get_aim(const AsSyncInputFrame* ptr, int16_t* outX, int16_t* outY, int16_t* outZ);

    // InputFrameArray
    void as_sync_input_frame_array_free(AsSyncInputFrameArray* ptr);
    uint32_t as_sync_input_frame_array_len(const AsSyncInputFrameArray* ptr);
    AsSyncInputFrame* as_sync_input_frame_array_get(const AsSyncInputFrameArray* ptr, uint32_t index);

    // Lockstep
    AsSyncLockstep* as_sync_lockstep_new(uint8_t playerCount);
    void as_sync_lockstep_free(AsSyncLockstep* ptr);
    void as_sync_lockstep_add_local_input(AsSyncLockstep* ptr, const AsSyncInputFrame* input);
    void as_sync_lockstep_add_remote_input(AsSyncLockstep* ptr, const AsSyncInputFrame* input);
    int32_t as_sync_lockstep_ready(const AsSyncLockstep* ptr);
    AsSyncInputFrameArray* as_sync_lockstep_advance(AsSyncLockstep* ptr);
    uint64_t as_sync_lockstep_confirmed_frame(const AsSyncLockstep* ptr);
    void as_sync_lockstep_record_checksum(AsSyncLockstep* ptr, uint64_t frame, uint64_t checksum);

    // Rollback
    AsSyncRollback* as_sync_rollback_new(uint8_t playerCount, uint8_t localPlayer, uint64_t maxRollback);
    void as_sync_rollback_free(AsSyncRollback* ptr);
    AsSyncInputFrameArray* as_sync_rollback_add_local_input(AsSyncRollback* ptr, const AsSyncInputFrame* input);
    uint8_t as_sync_rollback_add_remote_input(AsSyncRollback* ptr, const AsSyncInputFrame* input, uint64_t* outRollbackFrame);
    uint64_t as_sync_rollback_confirmed_frame(const AsSyncRollback* ptr);
    uint64_t as_sync_rollback_predicted_frame(const AsSyncRollback* ptr);
    uint64_t as_sync_rollback_frames_ahead(const AsSyncRollback* ptr);

    // Utilities
    int32_t as_sync_fixed_from_f32(float f);
    float as_sync_fixed_to_f32(int32_t bits);
    uint64_t as_sync_vec3_hash(float x, float y, float z);
    uint32_t as_sync_version();
}

// ============================================================================
// RAII Handle Wrappers
// ============================================================================

namespace Alice { namespace Sync {

struct WorldDeleter { void operator()(AsSyncWorld* p) const { as_sync_world_free(p); } };
using WorldHandle = std::unique_ptr<AsSyncWorld, WorldDeleter>;

struct WorldSoADeleter { void operator()(AsSyncWorldSoA* p) const { as_sync_world_soa_free(p); } };
using WorldSoAHandle = std::unique_ptr<AsSyncWorldSoA, WorldSoADeleter>;

struct NodeDeleter { void operator()(AsSyncNode* p) const { as_sync_node_free(p); } };
using NodeHandle = std::unique_ptr<AsSyncNode, NodeDeleter>;

struct EventDeleter { void operator()(AsSyncEvent* p) const { as_sync_event_free(p); } };
using EventHandle = std::unique_ptr<AsSyncEvent, EventDeleter>;

struct EventStreamDeleter { void operator()(AsSyncEventStream* p) const { as_sync_event_stream_free(p); } };
using EventStreamHandle = std::unique_ptr<AsSyncEventStream, EventStreamDeleter>;

struct InputFrameDeleter { void operator()(AsSyncInputFrame* p) const { as_sync_input_frame_free(p); } };
using InputFrameHandle = std::unique_ptr<AsSyncInputFrame, InputFrameDeleter>;

struct InputFrameArrayDeleter { void operator()(AsSyncInputFrameArray* p) const { as_sync_input_frame_array_free(p); } };
using InputFrameArrayHandle = std::unique_ptr<AsSyncInputFrameArray, InputFrameArrayDeleter>;

struct LockstepDeleter { void operator()(AsSyncLockstep* p) const { as_sync_lockstep_free(p); } };
using LockstepHandle = std::unique_ptr<AsSyncLockstep, LockstepDeleter>;

struct RollbackDeleter { void operator()(AsSyncRollback* p) const { as_sync_rollback_free(p); } };
using RollbackHandle = std::unique_ptr<AsSyncRollback, RollbackDeleter>;

}} // namespace Alice::Sync

// ALICE-Sync Unity C# Bindings — 71 DllImport + RAII wrappers
// Auto-generated from src/ffi.rs
// Copyright (C) 2026 Moroya Sakamoto — AGPL-3.0

using System;
using System.Runtime.InteropServices;

namespace Alice.Sync
{
    // ========================================================================
    // Enumerations
    // ========================================================================

    public enum NodeState : byte
    {
        Disconnected = 0,
        Connecting = 1,
        Synced = 2,
        Diverged = 3,
    }

    public enum RollbackActionType : byte
    {
        None = 0,
        Rollback = 1,
        Desync = 2,
    }

    // ========================================================================
    // World (AoS)
    // ========================================================================

    public sealed class World : IDisposable
    {
        internal IntPtr Ptr;
        private bool _disposed;

        public World(ulong seed) { Ptr = NativeMethods.as_sync_world_new(seed); }
        internal World(IntPtr ptr) { Ptr = ptr; }

        public ulong Hash => NativeMethods.as_sync_world_hash(Ptr);
        public uint EntityCount => NativeMethods.as_sync_world_entity_count(Ptr);
        public ulong Frame => NativeMethods.as_sync_world_frame(Ptr);

        public int ApplyEvent(Event evt) => NativeMethods.as_sync_world_apply_event(Ptr, evt.Ptr);

        public bool GetEntityPosition(uint entityId, out float x, out float y, out float z)
        {
            return NativeMethods.as_sync_world_get_entity_position(Ptr, entityId, out x, out y, out z) == 0;
        }

        public ulong RecalculateHash() => NativeMethods.as_sync_world_recalculate_hash(Ptr);

        public void Dispose()
        {
            if (!_disposed && Ptr != IntPtr.Zero)
            {
                NativeMethods.as_sync_world_free(Ptr);
                Ptr = IntPtr.Zero;
                _disposed = true;
            }
        }

        ~World() { Dispose(); }
    }

    // ========================================================================
    // WorldSoA
    // ========================================================================

    public sealed class WorldSoA : IDisposable
    {
        internal IntPtr Ptr;
        private bool _disposed;

        public WorldSoA(ulong seed) { Ptr = NativeMethods.as_sync_world_soa_new(seed); }

        public ulong Hash => NativeMethods.as_sync_world_soa_hash(Ptr);
        public uint EntityCount => NativeMethods.as_sync_world_soa_entity_count(Ptr);

        public int ApplyEvent(Event evt) => NativeMethods.as_sync_world_soa_apply_event(Ptr, evt.Ptr);

        public void Dispose()
        {
            if (!_disposed && Ptr != IntPtr.Zero)
            {
                NativeMethods.as_sync_world_soa_free(Ptr);
                Ptr = IntPtr.Zero;
                _disposed = true;
            }
        }

        ~WorldSoA() { Dispose(); }
    }

    // ========================================================================
    // Node
    // ========================================================================

    public sealed class Node : IDisposable
    {
        internal IntPtr Ptr;
        private bool _disposed;

        public Node(ulong nodeId) { Ptr = NativeMethods.as_sync_node_new(nodeId); }
        public Node(ulong nodeId, ulong seed) { Ptr = NativeMethods.as_sync_node_with_seed(nodeId, seed); }

        public ulong WorldHash => NativeMethods.as_sync_node_world_hash(Ptr);
        public NodeState State => (NodeState)NativeMethods.as_sync_node_state(Ptr);
        public uint EntityCount => NativeMethods.as_sync_node_entity_count(Ptr);
        public uint EventsCount => NativeMethods.as_sync_node_events_count(Ptr);
        public uint EventsBytes => NativeMethods.as_sync_node_events_bytes(Ptr);

        public int Emit(Event evt) => NativeMethods.as_sync_node_emit(Ptr, evt.Ptr);
        public int ApplyEvent(Event evt) => NativeMethods.as_sync_node_apply_event(Ptr, evt.Ptr);
        public void AddPeer(ulong peerId) => NativeMethods.as_sync_node_add_peer(Ptr, peerId);

        public void Dispose()
        {
            if (!_disposed && Ptr != IntPtr.Zero)
            {
                NativeMethods.as_sync_node_free(Ptr);
                Ptr = IntPtr.Zero;
                _disposed = true;
            }
        }

        ~Node() { Dispose(); }
    }

    // ========================================================================
    // Event
    // ========================================================================

    public sealed class Event : IDisposable
    {
        internal IntPtr Ptr;
        private bool _disposed;

        private Event(IntPtr ptr) { Ptr = ptr; }

        public static Event NewMotion(uint entity, short dx, short dy, short dz)
            => new Event(NativeMethods.as_sync_event_new_motion(entity, dx, dy, dz));

        public static Event NewSpawn(uint entity, ushort kind, short px, short py, short pz)
            => new Event(NativeMethods.as_sync_event_new_spawn(entity, kind, px, py, pz));

        public static Event NewDespawn(uint entity)
            => new Event(NativeMethods.as_sync_event_new_despawn(entity));

        public static Event NewProperty(uint entity, ushort prop, int value)
            => new Event(NativeMethods.as_sync_event_new_property(entity, prop, value));

        public static Event NewInput(ushort player, uint code)
            => new Event(NativeMethods.as_sync_event_new_input(player, code));

        public static Event NewTick(ulong frame)
            => new Event(NativeMethods.as_sync_event_new_tick(frame));

        public uint SizeBytes => NativeMethods.as_sync_event_size_bytes(Ptr);

        public void Dispose()
        {
            if (!_disposed && Ptr != IntPtr.Zero)
            {
                NativeMethods.as_sync_event_free(Ptr);
                Ptr = IntPtr.Zero;
                _disposed = true;
            }
        }

        ~Event() { Dispose(); }
    }

    // ========================================================================
    // EventStream
    // ========================================================================

    public sealed class EventStream : IDisposable
    {
        internal IntPtr Ptr;
        private bool _disposed;

        public EventStream() { Ptr = NativeMethods.as_sync_event_stream_new(); }

        public uint Length => NativeMethods.as_sync_event_stream_len(Ptr);
        public bool IsEmpty => NativeMethods.as_sync_event_stream_is_empty(Ptr) != 0;
        public ulong CurrentSeq => NativeMethods.as_sync_event_stream_current_seq(Ptr);
        public uint TotalBytes => NativeMethods.as_sync_event_stream_total_bytes(Ptr);

        public ulong Push(Event evt, ulong origin)
            => NativeMethods.as_sync_event_stream_push(Ptr, evt.Ptr, origin);

        public void Dispose()
        {
            if (!_disposed && Ptr != IntPtr.Zero)
            {
                NativeMethods.as_sync_event_stream_free(Ptr);
                Ptr = IntPtr.Zero;
                _disposed = true;
            }
        }

        ~EventStream() { Dispose(); }
    }

    // ========================================================================
    // InputFrame
    // ========================================================================

    public sealed class InputFrame : IDisposable
    {
        internal IntPtr Ptr;
        private bool _disposed;

        public InputFrame(ulong frame, byte playerId) { Ptr = NativeMethods.as_sync_input_frame_new(frame, playerId); }
        internal InputFrame(IntPtr ptr) { Ptr = ptr; }

        public ulong Frame => NativeMethods.as_sync_input_frame_get_frame(Ptr);
        public byte PlayerId => NativeMethods.as_sync_input_frame_get_player_id(Ptr);
        public uint Actions => NativeMethods.as_sync_input_frame_get_actions(Ptr);

        public void SetMovement(short x, short y, short z)
            => NativeMethods.as_sync_input_frame_set_movement(Ptr, x, y, z);

        public void SetActions(uint actions)
            => NativeMethods.as_sync_input_frame_set_actions(Ptr, actions);

        public void SetAim(short x, short y, short z)
            => NativeMethods.as_sync_input_frame_set_aim(Ptr, x, y, z);

        public void GetMovement(out short x, out short y, out short z)
            => NativeMethods.as_sync_input_frame_get_movement(Ptr, out x, out y, out z);

        public void GetAim(out short x, out short y, out short z)
            => NativeMethods.as_sync_input_frame_get_aim(Ptr, out x, out y, out z);

        public void Dispose()
        {
            if (!_disposed && Ptr != IntPtr.Zero)
            {
                NativeMethods.as_sync_input_frame_free(Ptr);
                Ptr = IntPtr.Zero;
                _disposed = true;
            }
        }

        ~InputFrame() { Dispose(); }
    }

    // ========================================================================
    // InputFrameArray
    // ========================================================================

    public sealed class InputFrameArray : IDisposable
    {
        internal IntPtr Ptr;
        private bool _disposed;

        internal InputFrameArray(IntPtr ptr) { Ptr = ptr; }

        public uint Length => NativeMethods.as_sync_input_frame_array_len(Ptr);

        public InputFrame Get(uint index)
        {
            IntPtr p = NativeMethods.as_sync_input_frame_array_get(Ptr, index);
            return p == IntPtr.Zero ? null : new InputFrame(p);
        }

        public void Dispose()
        {
            if (!_disposed && Ptr != IntPtr.Zero)
            {
                NativeMethods.as_sync_input_frame_array_free(Ptr);
                Ptr = IntPtr.Zero;
                _disposed = true;
            }
        }

        ~InputFrameArray() { Dispose(); }
    }

    // ========================================================================
    // LockstepSession
    // ========================================================================

    public sealed class LockstepSession : IDisposable
    {
        internal IntPtr Ptr;
        private bool _disposed;

        public LockstepSession(byte playerCount) { Ptr = NativeMethods.as_sync_lockstep_new(playerCount); }

        public bool Ready => NativeMethods.as_sync_lockstep_ready(Ptr) != 0;
        public ulong ConfirmedFrame => NativeMethods.as_sync_lockstep_confirmed_frame(Ptr);

        public void AddLocalInput(InputFrame input)
            => NativeMethods.as_sync_lockstep_add_local_input(Ptr, input.Ptr);

        public void AddRemoteInput(InputFrame input)
            => NativeMethods.as_sync_lockstep_add_remote_input(Ptr, input.Ptr);

        public InputFrameArray Advance()
        {
            IntPtr p = NativeMethods.as_sync_lockstep_advance(Ptr);
            return p == IntPtr.Zero ? null : new InputFrameArray(p);
        }

        public void RecordChecksum(ulong frame, ulong checksum)
            => NativeMethods.as_sync_lockstep_record_checksum(Ptr, frame, checksum);

        public void Dispose()
        {
            if (!_disposed && Ptr != IntPtr.Zero)
            {
                NativeMethods.as_sync_lockstep_free(Ptr);
                Ptr = IntPtr.Zero;
                _disposed = true;
            }
        }

        ~LockstepSession() { Dispose(); }
    }

    // ========================================================================
    // RollbackSession
    // ========================================================================

    public sealed class RollbackSession : IDisposable
    {
        internal IntPtr Ptr;
        private bool _disposed;

        public RollbackSession(byte playerCount, byte localPlayer, ulong maxRollback)
        {
            Ptr = NativeMethods.as_sync_rollback_new(playerCount, localPlayer, maxRollback);
        }

        public ulong ConfirmedFrame => NativeMethods.as_sync_rollback_confirmed_frame(Ptr);
        public ulong PredictedFrame => NativeMethods.as_sync_rollback_predicted_frame(Ptr);
        public ulong FramesAhead => NativeMethods.as_sync_rollback_frames_ahead(Ptr);

        public InputFrameArray AddLocalInput(InputFrame input)
        {
            IntPtr p = NativeMethods.as_sync_rollback_add_local_input(Ptr, input.Ptr);
            return p == IntPtr.Zero ? null : new InputFrameArray(p);
        }

        public RollbackActionType AddRemoteInput(InputFrame input, out ulong rollbackFrame)
        {
            rollbackFrame = 0;
            return (RollbackActionType)NativeMethods.as_sync_rollback_add_remote_input(Ptr, input.Ptr, out rollbackFrame);
        }

        public void Dispose()
        {
            if (!_disposed && Ptr != IntPtr.Zero)
            {
                NativeMethods.as_sync_rollback_free(Ptr);
                Ptr = IntPtr.Zero;
                _disposed = true;
            }
        }

        ~RollbackSession() { Dispose(); }
    }

    // ========================================================================
    // Static utilities
    // ========================================================================

    public static class AliceSync
    {
        public static int FixedFromF32(float f) => NativeMethods.as_sync_fixed_from_f32(f);
        public static float FixedToF32(int bits) => NativeMethods.as_sync_fixed_to_f32(bits);
        public static ulong Vec3Hash(float x, float y, float z) => NativeMethods.as_sync_vec3_hash(x, y, z);
        public static uint Version => NativeMethods.as_sync_version();
    }

    // ========================================================================
    // Native Methods (65 DllImport)
    // ========================================================================

    internal static class NativeMethods
    {
        private const string Lib = "alice_sync";

        // World
        [DllImport(Lib)] internal static extern IntPtr as_sync_world_new(ulong seed);
        [DllImport(Lib)] internal static extern void as_sync_world_free(IntPtr ptr);
        [DllImport(Lib)] internal static extern ulong as_sync_world_hash(IntPtr ptr);
        [DllImport(Lib)] internal static extern uint as_sync_world_entity_count(IntPtr ptr);
        [DllImport(Lib)] internal static extern ulong as_sync_world_frame(IntPtr ptr);
        [DllImport(Lib)] internal static extern int as_sync_world_apply_event(IntPtr ptr, IntPtr evt);
        [DllImport(Lib)] internal static extern int as_sync_world_get_entity_position(IntPtr ptr, uint entityId, out float x, out float y, out float z);
        [DllImport(Lib)] internal static extern ulong as_sync_world_recalculate_hash(IntPtr ptr);

        // WorldSoA
        [DllImport(Lib)] internal static extern IntPtr as_sync_world_soa_new(ulong seed);
        [DllImport(Lib)] internal static extern void as_sync_world_soa_free(IntPtr ptr);
        [DllImport(Lib)] internal static extern ulong as_sync_world_soa_hash(IntPtr ptr);
        [DllImport(Lib)] internal static extern uint as_sync_world_soa_entity_count(IntPtr ptr);
        [DllImport(Lib)] internal static extern int as_sync_world_soa_apply_event(IntPtr ptr, IntPtr evt);


        // Node
        [DllImport(Lib)] internal static extern IntPtr as_sync_node_new(ulong nodeId);
        [DllImport(Lib)] internal static extern IntPtr as_sync_node_with_seed(ulong nodeId, ulong seed);
        [DllImport(Lib)] internal static extern void as_sync_node_free(IntPtr ptr);
        [DllImport(Lib)] internal static extern ulong as_sync_node_world_hash(IntPtr ptr);
        [DllImport(Lib)] internal static extern byte as_sync_node_state(IntPtr ptr);
        [DllImport(Lib)] internal static extern int as_sync_node_emit(IntPtr ptr, IntPtr evt);
        [DllImport(Lib)] internal static extern int as_sync_node_apply_event(IntPtr ptr, IntPtr evt);
        [DllImport(Lib)] internal static extern void as_sync_node_add_peer(IntPtr ptr, ulong peerId);
        [DllImport(Lib)] internal static extern uint as_sync_node_entity_count(IntPtr ptr);
        [DllImport(Lib)] internal static extern uint as_sync_node_events_count(IntPtr ptr);
        [DllImport(Lib)] internal static extern uint as_sync_node_events_bytes(IntPtr ptr);

        // Event
        [DllImport(Lib)] internal static extern IntPtr as_sync_event_new_motion(uint entity, short dx, short dy, short dz);
        [DllImport(Lib)] internal static extern IntPtr as_sync_event_new_spawn(uint entity, ushort kind, short px, short py, short pz);
        [DllImport(Lib)] internal static extern IntPtr as_sync_event_new_despawn(uint entity);
        [DllImport(Lib)] internal static extern IntPtr as_sync_event_new_property(uint entity, ushort prop, int value);
        [DllImport(Lib)] internal static extern IntPtr as_sync_event_new_input(ushort player, uint code);
        [DllImport(Lib)] internal static extern IntPtr as_sync_event_new_tick(ulong frame);
        [DllImport(Lib)] internal static extern void as_sync_event_free(IntPtr ptr);
        [DllImport(Lib)] internal static extern uint as_sync_event_size_bytes(IntPtr ptr);

        // EventStream
        [DllImport(Lib)] internal static extern IntPtr as_sync_event_stream_new();
        [DllImport(Lib)] internal static extern void as_sync_event_stream_free(IntPtr ptr);
        [DllImport(Lib)] internal static extern ulong as_sync_event_stream_push(IntPtr ptr, IntPtr evt, ulong origin);
        [DllImport(Lib)] internal static extern uint as_sync_event_stream_len(IntPtr ptr);
        [DllImport(Lib)] internal static extern int as_sync_event_stream_is_empty(IntPtr ptr);
        [DllImport(Lib)] internal static extern ulong as_sync_event_stream_current_seq(IntPtr ptr);
        [DllImport(Lib)] internal static extern uint as_sync_event_stream_total_bytes(IntPtr ptr);

        // InputFrame
        [DllImport(Lib)] internal static extern IntPtr as_sync_input_frame_new(ulong frame, byte playerId);
        [DllImport(Lib)] internal static extern void as_sync_input_frame_free(IntPtr ptr);
        [DllImport(Lib)] internal static extern void as_sync_input_frame_set_movement(IntPtr ptr, short x, short y, short z);
        [DllImport(Lib)] internal static extern void as_sync_input_frame_set_actions(IntPtr ptr, uint actions);
        [DllImport(Lib)] internal static extern void as_sync_input_frame_set_aim(IntPtr ptr, short x, short y, short z);
        [DllImport(Lib)] internal static extern ulong as_sync_input_frame_get_frame(IntPtr ptr);
        [DllImport(Lib)] internal static extern byte as_sync_input_frame_get_player_id(IntPtr ptr);
        [DllImport(Lib)] internal static extern void as_sync_input_frame_get_movement(IntPtr ptr, out short x, out short y, out short z);
        [DllImport(Lib)] internal static extern uint as_sync_input_frame_get_actions(IntPtr ptr);
        [DllImport(Lib)] internal static extern void as_sync_input_frame_get_aim(IntPtr ptr, out short x, out short y, out short z);

        // InputFrameArray
        [DllImport(Lib)] internal static extern void as_sync_input_frame_array_free(IntPtr ptr);
        [DllImport(Lib)] internal static extern uint as_sync_input_frame_array_len(IntPtr ptr);
        [DllImport(Lib)] internal static extern IntPtr as_sync_input_frame_array_get(IntPtr ptr, uint index);

        // Lockstep
        [DllImport(Lib)] internal static extern IntPtr as_sync_lockstep_new(byte playerCount);
        [DllImport(Lib)] internal static extern void as_sync_lockstep_free(IntPtr ptr);
        [DllImport(Lib)] internal static extern void as_sync_lockstep_add_local_input(IntPtr ptr, IntPtr input);
        [DllImport(Lib)] internal static extern void as_sync_lockstep_add_remote_input(IntPtr ptr, IntPtr input);
        [DllImport(Lib)] internal static extern int as_sync_lockstep_ready(IntPtr ptr);
        [DllImport(Lib)] internal static extern IntPtr as_sync_lockstep_advance(IntPtr ptr);
        [DllImport(Lib)] internal static extern ulong as_sync_lockstep_confirmed_frame(IntPtr ptr);
        [DllImport(Lib)] internal static extern void as_sync_lockstep_record_checksum(IntPtr ptr, ulong frame, ulong checksum);

        // Rollback
        [DllImport(Lib)] internal static extern IntPtr as_sync_rollback_new(byte playerCount, byte localPlayer, ulong maxRollback);
        [DllImport(Lib)] internal static extern void as_sync_rollback_free(IntPtr ptr);
        [DllImport(Lib)] internal static extern IntPtr as_sync_rollback_add_local_input(IntPtr ptr, IntPtr input);
        [DllImport(Lib)] internal static extern byte as_sync_rollback_add_remote_input(IntPtr ptr, IntPtr input, out ulong rollbackFrame);
        [DllImport(Lib)] internal static extern ulong as_sync_rollback_confirmed_frame(IntPtr ptr);
        [DllImport(Lib)] internal static extern ulong as_sync_rollback_predicted_frame(IntPtr ptr);
        [DllImport(Lib)] internal static extern ulong as_sync_rollback_frames_ahead(IntPtr ptr);

        // Utilities
        [DllImport(Lib)] internal static extern int as_sync_fixed_from_f32(float f);
        [DllImport(Lib)] internal static extern float as_sync_fixed_to_f32(int bits);
        [DllImport(Lib)] internal static extern ulong as_sync_vec3_hash(float x, float y, float z);
        [DllImport(Lib)] internal static extern uint as_sync_version();
    }
}

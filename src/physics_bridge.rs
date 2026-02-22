/*
    ALICE-Sync
    Copyright (C) 2026 Moroya Sakamoto

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

//! Bridge between ALICE-Sync input synchronization and ALICE-Physics netcode.
//!
//! Converts between Sync's compact network-oriented [`InputFrame`] (i16 Q8.8,
//! ~24 bytes via bitcode) and Physics' full-precision [`FrameInput`] (Fix128).
//!
//! Also bridges checksum types and provides [`PhysicsRollbackSession`] that
//! combines rollback input sync with deterministic physics simulation.
//!
//! # Architecture
//!
//! ```text
//! Network ──► InputFrame (i16, 24B) ──► FrameInput (Fix128) ──► PhysicsWorld
//!                  ALICE-Sync                ALICE-Physics
//!
//! PhysicsWorld ──► SimulationChecksum ──► WorldHash ──► Desync Verification
//!                    ALICE-Physics           ALICE-Sync
//! ```

use crate::input_sync::{InputFrame, RollbackAction, RollbackSession, SyncResult};
use crate::world::WorldHash;
use alice_physics::{
    DeterministicSimulation, FrameInput, InputApplicator, NetcodeConfig, SimulationChecksum,
    Vec3Fix,
};

// ============================================================================
// Input Conversion
// ============================================================================

/// Convert a Sync [`InputFrame`] to a Physics [`FrameInput`].
///
/// Movement and aim are stored as i16 in `InputFrame` (compact for network)
/// and expanded to `Vec3Fix` (Fix128-based) for physics simulation.
///
/// The conversion is: `i16 → i64 → Fix128::from_int()` which places the
/// i16 value in the integer part of the fixed-point number.
#[inline]
pub fn sync_input_to_physics(input: &InputFrame) -> FrameInput {
    FrameInput {
        player_id: input.player_id,
        movement: Vec3Fix::from_int(
            input.movement[0] as i64,
            input.movement[1] as i64,
            input.movement[2] as i64,
        ),
        actions: input.actions,
        aim_direction: Vec3Fix::from_int(
            input.aim[0] as i64,
            input.aim[1] as i64,
            input.aim[2] as i64,
        ),
    }
}

/// Convert a Physics [`FrameInput`] to a Sync [`InputFrame`].
///
/// Truncates Fix128 to i16 by taking the integer part (`hi` field),
/// matching the serialization format used by `FrameInput::to_bytes()`.
#[inline]
pub fn physics_input_to_sync(input: &FrameInput, frame: u64) -> InputFrame {
    InputFrame {
        frame,
        player_id: input.player_id,
        movement: [
            input.movement.x.hi as i16,
            input.movement.y.hi as i16,
            input.movement.z.hi as i16,
        ],
        actions: input.actions,
        aim: [
            input.aim_direction.x.hi as i16,
            input.aim_direction.y.hi as i16,
            input.aim_direction.z.hi as i16,
        ],
    }
}

/// Batch-convert multiple Sync inputs to Physics inputs.
#[inline]
pub fn sync_inputs_to_physics(inputs: &[InputFrame]) -> Vec<FrameInput> {
    inputs.iter().map(sync_input_to_physics).collect()
}

// ============================================================================
// Checksum Conversion
// ============================================================================

/// Convert a Physics [`SimulationChecksum`] to a Sync [`WorldHash`].
///
/// Both are XOR rolling hashes stored as `u64` — direct inner value mapping.
#[inline]
pub fn physics_checksum_to_world_hash(checksum: SimulationChecksum) -> WorldHash {
    WorldHash(checksum.0)
}

/// Convert a Sync [`WorldHash`] to a Physics [`SimulationChecksum`].
#[inline]
pub fn world_hash_to_physics_checksum(hash: WorldHash) -> SimulationChecksum {
    SimulationChecksum(hash.0)
}

// ============================================================================
// Physics Rollback Session
// ============================================================================

/// Combined rollback input synchronization + deterministic physics simulation.
///
/// Wraps [`RollbackSession`] (input sync) and [`DeterministicSimulation`]
/// (physics) into a single game loop driver.
///
/// # Usage
///
/// ```rust,ignore
/// use alice_sync::physics_bridge::PhysicsRollbackSession;
/// use alice_physics::NetcodeConfig;
///
/// let mut session = PhysicsRollbackSession::new(2, 0, 8, NetcodeConfig::default());
///
/// // Each frame:
/// let input = InputFrame::new(frame, 0).with_movement(1, 0, 0);
/// let checksum = session.advance_frame(input);
///
/// // When remote input arrives:
/// let action = session.add_remote_input(remote_input);
/// if let RollbackAction::Rollback { to_frame } = action {
///     session.handle_rollback(to_frame);
/// }
/// ```
pub struct PhysicsRollbackSession {
    /// Input synchronization (rollback netcode)
    pub sync: RollbackSession,
    /// Deterministic physics simulation
    pub sim: DeterministicSimulation,
}

impl PhysicsRollbackSession {
    /// Create a new physics rollback session.
    pub fn new(
        player_count: u8,
        local_player: u8,
        max_rollback: u64,
        netcode_config: NetcodeConfig,
    ) -> Self {
        Self {
            sync: RollbackSession::new(player_count, local_player, max_rollback),
            sim: DeterministicSimulation::new(netcode_config),
        }
    }

    /// Advance one frame with local input.
    ///
    /// 1. Submits local input to rollback session (predicts remote inputs)
    /// 2. Converts all inputs to Physics format
    /// 3. Saves physics snapshot (for potential rollback)
    /// 4. Steps physics simulation
    /// 5. Records checksum in sync session
    ///
    /// Returns the simulation checksum for this frame.
    pub fn advance_frame(&mut self, local_input: InputFrame) -> SimulationChecksum {
        let frame = local_input.frame;

        // 1. Sync: submit local input, get all inputs (confirmed + predicted)
        let sync_inputs = self.sync.add_local_input(local_input);

        // 2. Convert to physics format
        let physics_inputs: Vec<FrameInput> = sync_inputs_to_physics(&sync_inputs);

        // 3. Save snapshot before advancing (for rollback)
        self.sim.save_snapshot();

        // 4. Step physics
        let checksum = self.sim.advance_frame(&physics_inputs);

        // 5. Record checksum in sync session
        let state = self.sim.world.serialize_state();
        self.sync.save_snapshot(frame, state, checksum.0);

        checksum
    }

    /// Advance one frame with a custom input applicator.
    pub fn advance_frame_with_applicator(
        &mut self,
        local_input: InputFrame,
        applicator: &dyn InputApplicator,
    ) -> SimulationChecksum {
        let frame = local_input.frame;

        let sync_inputs = self.sync.add_local_input(local_input);
        let physics_inputs: Vec<FrameInput> = sync_inputs_to_physics(&sync_inputs);

        self.sim.save_snapshot();
        let checksum = self
            .sim
            .advance_frame_with_applicator(&physics_inputs, applicator);

        let state = self.sim.world.serialize_state();
        self.sync.save_snapshot(frame, state, checksum.0);

        checksum
    }

    /// Add a remote player's confirmed input.
    ///
    /// Returns the rollback action the game loop should handle.
    pub fn add_remote_input(&mut self, input: InputFrame) -> RollbackAction {
        self.sync.add_remote_input(input)
    }

    /// Handle a rollback: restore physics state and re-simulate.
    ///
    /// 1. Load physics snapshot at `to_frame - 1` (state before the bad frame)
    /// 2. Re-simulate from `to_frame` to current predicted frame
    ///    using corrected (now confirmed) inputs
    ///
    /// Returns `true` if rollback succeeded.
    pub fn handle_rollback(&mut self, to_frame: u64) -> bool {
        let current_frame = self.sim.frame();

        // Load snapshot from just before the rollback point
        let snapshot_frame = to_frame.saturating_sub(1);
        if !self.sim.load_snapshot(snapshot_frame) {
            return false;
        }

        // Re-simulate from rollback point to current frame
        for frame in to_frame..=current_frame {
            let sync_inputs = self.sync.inputs_for_frame(frame);
            let physics_inputs = sync_inputs_to_physics(&sync_inputs);

            self.sim.save_snapshot();
            let checksum = self.sim.advance_frame(&physics_inputs);

            let state = self.sim.world.serialize_state();
            self.sync.save_snapshot(frame, state, checksum.0);
        }

        true
    }

    /// Handle a rollback with a custom input applicator.
    pub fn handle_rollback_with_applicator(
        &mut self,
        to_frame: u64,
        applicator: &dyn InputApplicator,
    ) -> bool {
        let current_frame = self.sim.frame();

        let snapshot_frame = to_frame.saturating_sub(1);
        if !self.sim.load_snapshot(snapshot_frame) {
            return false;
        }

        for frame in to_frame..=current_frame {
            let sync_inputs = self.sync.inputs_for_frame(frame);
            let physics_inputs = sync_inputs_to_physics(&sync_inputs);

            self.sim.save_snapshot();
            let checksum = self
                .sim
                .advance_frame_with_applicator(&physics_inputs, applicator);

            let state = self.sim.world.serialize_state();
            self.sync.save_snapshot(frame, state, checksum.0);
        }

        true
    }

    /// Verify a remote checksum against local physics state.
    pub fn verify_checksum(&self, frame: u64, remote_checksum: u64) -> SyncResult {
        self.sync.verify_checksum(frame, remote_checksum)
    }

    /// Current simulation frame.
    #[inline]
    pub fn frame(&self) -> u64 {
        self.sim.frame()
    }

    /// Current confirmed frame (all players' inputs received).
    #[inline]
    pub fn confirmed_frame(&self) -> u64 {
        self.sync.confirmed_frame()
    }

    /// Number of frames ahead of confirmation.
    #[inline]
    pub fn frames_ahead(&self) -> u64 {
        self.sync.frames_ahead()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use alice_physics::{Fix128, RigidBody};

    #[test]
    fn test_sync_input_to_physics_roundtrip() {
        let sync_input = InputFrame::new(42, 1)
            .with_movement(100, -50, 200)
            .with_actions(0x05)
            .with_aim(0, 1000, -500);

        let physics = sync_input_to_physics(&sync_input);
        assert_eq!(physics.player_id, 1);
        assert_eq!(physics.movement.x.hi, 100);
        assert_eq!(physics.movement.y.hi, -50);
        assert_eq!(physics.movement.z.hi, 200);
        assert_eq!(physics.actions, 0x05);
        assert_eq!(physics.aim_direction.y.hi, 1000);

        // Roundtrip
        let back = physics_input_to_sync(&physics, 42);
        assert_eq!(back, sync_input);
    }

    #[test]
    fn test_checksum_conversion() {
        let physics_checksum = SimulationChecksum(0xDEADBEEF_CAFEBABE);
        let world_hash = physics_checksum_to_world_hash(physics_checksum);
        assert_eq!(world_hash.0, 0xDEADBEEF_CAFEBABE);

        let back = world_hash_to_physics_checksum(world_hash);
        assert_eq!(back, physics_checksum);
    }

    #[test]
    fn test_physics_rollback_session_advance() {
        let netcode_config = NetcodeConfig {
            player_count: 2,
            ..Default::default()
        };
        let mut session = PhysicsRollbackSession::new(2, 0, 8, netcode_config);

        // Add player bodies
        for i in 0..2u8 {
            let body = RigidBody::new_dynamic(Vec3Fix::from_int(i as i64 * 5, 10, 0), Fix128::ONE);
            let idx = session.sim.add_body(body);
            session.sim.assign_player_body(i, idx);
        }

        // Ground
        session.sim.add_body(RigidBody::new_static(Vec3Fix::ZERO));

        // Advance 3 frames
        for f in 1..=3u64 {
            let input = InputFrame::new(f, 0).with_movement(1, 0, 0);
            let checksum = session.advance_frame(input);
            assert_ne!(checksum.0, 0);
        }

        assert_eq!(session.frame(), 3);
        assert_eq!(session.frames_ahead(), 3); // remote hasn't sent anything
    }

    #[test]
    fn test_physics_rollback_session_rollback() {
        let netcode_config = NetcodeConfig {
            player_count: 2,
            max_snapshots: 10,
            ..Default::default()
        };
        let mut session = PhysicsRollbackSession::new(2, 0, 8, netcode_config);

        // Add player bodies
        for i in 0..2u8 {
            let body = RigidBody::new_dynamic(Vec3Fix::from_int(i as i64 * 5, 10, 0), Fix128::ONE);
            let idx = session.sim.add_body(body);
            session.sim.assign_player_body(i, idx);
        }
        session.sim.add_body(RigidBody::new_static(Vec3Fix::ZERO));

        // Advance 3 frames (remote predicted as empty)
        for f in 1..=3u64 {
            let input = InputFrame::new(f, 0).with_movement(1, 0, 0);
            session.advance_frame(input);
        }

        // Remote sends frame 1 with different data → rollback
        let action = session.add_remote_input(InputFrame::new(1, 1).with_movement(5, 0, -5));
        assert_eq!(action, RollbackAction::Rollback { to_frame: 1 });

        // Handle the rollback
        let ok = session.handle_rollback(1);
        assert!(ok);
        assert_eq!(session.frame(), 3); // back to current frame after re-sim
    }

    #[test]
    fn test_batch_conversion() {
        let inputs = vec![
            InputFrame::new(1, 0).with_movement(10, 0, 0),
            InputFrame::new(1, 1).with_movement(0, 0, -10),
        ];

        let physics = sync_inputs_to_physics(&inputs);
        assert_eq!(physics.len(), 2);
        assert_eq!(physics[0].movement.x.hi, 10);
        assert_eq!(physics[1].movement.z.hi, -10);
    }

    #[test]
    fn test_negative_values_preserved() {
        let input = InputFrame::new(1, 0)
            .with_movement(-32768, 32767, -1)
            .with_aim(-100, -200, -300);

        let physics = sync_input_to_physics(&input);
        assert_eq!(physics.movement.x.hi, -32768);
        assert_eq!(physics.movement.y.hi, 32767);
        assert_eq!(physics.movement.z.hi, -1);
        assert_eq!(physics.aim_direction.x.hi, -100);

        let back = physics_input_to_sync(&physics, 1);
        assert_eq!(back.movement, [-32768, 32767, -1]);
        assert_eq!(back.aim, [-100, -200, -300]);
    }
}

/*
    ALICE-Sync
    Copyright (C) 2026 Moroya Sakamoto

    License: LicenseRef-Proprietary (see LICENSE-Proprietary)
*/

//! Bridge between ALICE-Sync input synchronization and ALICE-Physics netcode.
//!
//! Converts between Sync's [`InputFrame`] and Physics' [`FrameInput`] bit for bit: each movement and
//! aim component is an [`InputFix`] in Sync and a `Fix128` in Physics, with the same `hi` / `lo` words.
//!
//! Also bridges checksum types and provides [`PhysicsRollbackSession`] that
//! combines rollback input sync with deterministic physics simulation.
//!
//! # Architecture
//!
//! ```text
//! Network ──► InputFrame (InputFix) ──► FrameInput (Fix128) ──► PhysicsWorld
//!                  ALICE-Sync                ALICE-Physics
//!
//! PhysicsWorld ──► SimulationChecksum ──► WorldHash ──► Desync Verification
//!                    ALICE-Physics           ALICE-Sync
//! ```

use crate::input_sync::{InputFix, InputFrame, RollbackAction, RollbackSession, SyncResult};
use crate::semantics::{check_semantics_against, SemanticsMismatch};
use crate::world::WorldHash;
use alice_physics::{
    DeterministicSimulation, Fix128, FrameInput, InputApplicator, NetcodeConfig,
    SimulationChecksum, Vec3Fix,
};

// ============================================================================
// Input Conversion
// ============================================================================

#[inline]
#[must_use]
const fn to_fix(v: InputFix) -> Fix128 {
    Fix128::from_raw(v.hi, v.lo)
}

#[inline]
#[must_use]
const fn to_vec3fix(v: [InputFix; 3]) -> Vec3Fix {
    Vec3Fix::new(to_fix(v[0]), to_fix(v[1]), to_fix(v[2]))
}

#[inline]
#[must_use]
const fn from_fix(v: Fix128) -> InputFix {
    InputFix::from_raw(v.hi, v.lo)
}

#[inline]
#[must_use]
const fn from_vec3fix(v: Vec3Fix) -> [InputFix; 3] {
    [from_fix(v.x), from_fix(v.y), from_fix(v.z)]
}

/// Convert a Sync [`InputFrame`] to a Physics [`FrameInput`], bit for bit (each [`InputFix`]
/// becomes the `Fix128` with the same `hi` and `lo`).
#[inline]
#[must_use]
pub fn sync_input_to_physics(input: &InputFrame) -> FrameInput {
    // `FrameInput` は non_exhaustive なので struct literal は使えない
    FrameInput::new(input.player_id)
        .with_movement(to_vec3fix(input.movement))
        .with_actions(input.actions)
        .with_aim(to_vec3fix(input.aim))
}

/// Convert a Physics [`FrameInput`] to a Sync [`InputFrame`] for `frame`, bit for bit, so
/// [`sync_input_to_physics`] of the result is `input`.
#[inline]
#[must_use]
pub fn physics_input_to_sync(input: &FrameInput, frame: u64) -> InputFrame {
    InputFrame {
        frame,
        player_id: input.player_id,
        movement: from_vec3fix(input.movement),
        actions: input.actions,
        aim: from_vec3fix(input.aim_direction),
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
#[must_use]
pub fn physics_checksum_to_world_hash(checksum: SimulationChecksum) -> WorldHash {
    WorldHash(checksum.0)
}

/// Convert a Sync [`WorldHash`] to a Physics [`SimulationChecksum`].
#[inline]
#[must_use]
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
/// // the peer's alice-physics identifier, received in the session handshake
/// let mut session = PhysicsRollbackSession::new(
///     2, 0, 8, NetcodeConfig::default(), &peer_physics_id,
/// )?;
///
/// // Each frame:
/// let input = InputFrame::new(frame, 0).with_movement(InputFix::from_int(1), InputFix::ZERO, InputFix::ZERO);
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
    /// Create a new physics rollback session with a peer whose simulation
    /// reports `peer_physics_id` (its `alice_physics::PHYSICS_SEMANTICS_ID`).
    ///
    /// Rollback compares the peer's checksums with this simulation's, which is
    /// only meaningful when both run the same physics semantics (ALICE-Law
    /// G7): the identifier must be this build's `PHYSICS_SEMANTICS_ID` or one
    /// in `PREVIOUS_PHYSICS_SEMANTICS_IDS`.
    ///
    /// # Errors
    ///
    /// [`SemanticsMismatch`] naming the peer's identifier and this build's
    /// when they differ.
    pub fn new(
        player_count: u8,
        local_player: u8,
        max_rollback: u64,
        netcode_config: NetcodeConfig,
        peer_physics_id: &[u8; 32],
    ) -> Result<Self, SemanticsMismatch> {
        check_semantics_against(
            peer_physics_id,
            &alice_physics::PHYSICS_SEMANTICS_ID,
            alice_physics::PREVIOUS_PHYSICS_SEMANTICS_IDS,
        )?;
        Ok(Self {
            sync: RollbackSession::new(player_count, local_player, max_rollback),
            sim: DeterministicSimulation::new(netcode_config),
        })
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
    #[must_use]
    pub fn verify_checksum(&self, frame: u64, remote_checksum: u64) -> SyncResult {
        self.sync.verify_checksum(frame, remote_checksum)
    }

    /// Current simulation frame.
    #[inline]
    #[must_use]
    pub fn frame(&self) -> u64 {
        self.sim.frame()
    }

    /// Current confirmed frame (all players' inputs received).
    #[inline]
    #[must_use]
    pub fn confirmed_frame(&self) -> u64 {
        self.sync.confirmed_frame()
    }

    /// Number of frames ahead of confirmation.
    #[inline]
    #[must_use]
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

    /// `FrameInput` が non_exhaustive になった時に struct literal から
    /// `new` + `with_*` へ書き換えたので、4 field すべてが書き換え前の
    /// literal と同じ値であることを固定する
    /// (`with_*` を 1 つ落とすと `new` の初期値が silent に残るため)
    #[test]
    fn sync_input_to_physics_fills_the_same_four_fields_as_the_pre_rewrite_literal() {
        // `with_*` を 1 つ落とすと `new` の初期値が silent に残るため、4 field を固定する
        let sync_input = InputFrame::new(7, 3)
            .with_movement(
                InputFix::from_raw(1, 0),
                InputFix::from_raw(-1, 1 << 63),
                InputFix::from_raw(0, 1),
            )
            .with_actions(0x0B)
            .with_aim(
                InputFix::from_raw(-2, 0),
                InputFix::from_raw(1, 0),
                InputFix::from_raw(0, 1 << 62),
            );

        let physics = sync_input_to_physics(&sync_input);
        assert_eq!(physics.player_id, sync_input.player_id);
        assert_eq!(physics.actions, sync_input.actions);
        assert_eq!(physics.movement, to_vec3fix(sync_input.movement));
        assert_eq!(physics.aim_direction, to_vec3fix(sync_input.aim));
        assert_ne!(physics.movement, Vec3Fix::ZERO);
        assert_ne!(physics.aim_direction, Vec3Fix::ZERO);
        assert_ne!(physics.actions, 0);
    }

    #[test]
    fn test_sync_input_to_physics_roundtrip() {
        // 0.390625 = 100 / 256 and its neighbours: the fractional word is carried as is
        let sync_input = InputFrame::new(42, 1)
            .with_movement(
                InputFix::from_raw(0, 0x6400_0000_0000_0000),
                InputFix::from_raw(-1, 1),
                InputFix::from_int(200),
            )
            .with_actions(0x05)
            .with_aim(
                InputFix::ZERO,
                InputFix::from_raw(3, u64::MAX),
                InputFix::from_int(-500),
            );

        let physics = sync_input_to_physics(&sync_input);
        assert_eq!(physics.player_id, 1);
        assert_eq!(physics.movement.x.to_f64(), 100.0 / 256.0);
        assert_eq!((physics.movement.y.hi, physics.movement.y.lo), (-1, 1));
        assert_eq!(physics.movement.z.to_f64(), 200.0);
        assert_eq!(physics.actions, 0x05);
        assert_eq!(
            (physics.aim_direction.y.hi, physics.aim_direction.y.lo),
            (3, u64::MAX)
        );

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

    /// A peer with another physics identifier is refused naming both; an
    /// identifier in the previous list is accepted.
    #[test]
    fn a_peer_with_another_physics_identifier_is_refused() {
        let mut foreign = alice_physics::PHYSICS_SEMANTICS_ID;
        foreign[31] ^= 1;
        match PhysicsRollbackSession::new(2, 0, 8, NetcodeConfig::default(), &foreign) {
            Err(m) => {
                assert_eq!(m.stored, foreign);
                assert_eq!(m.expected, alice_physics::PHYSICS_SEMANTICS_ID);
            }
            Ok(_) => panic!("a foreign physics identifier was accepted"),
        }
        let earlier = alice_physics::PREVIOUS_PHYSICS_SEMANTICS_IDS[0];
        assert!(PhysicsRollbackSession::new(2, 0, 8, NetcodeConfig::default(), &earlier).is_ok());
    }

    #[test]
    fn test_physics_rollback_session_advance() {
        // `NetcodeConfig` は non_exhaustive なので struct literal も
        // `..Default::default()` も使えない 元の literal が明示していた
        // `player_count: 2` は default と同値なので、値を assert で固定する
        let netcode_config = NetcodeConfig::default();
        assert_eq!(netcode_config.player_count, 2);
        let mut session = PhysicsRollbackSession::new(
            2,
            0,
            8,
            netcode_config,
            &alice_physics::PHYSICS_SEMANTICS_ID,
        )
        .expect("this build's identifier");

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
            let input = InputFrame::new(f, 0).with_movement(
                InputFix::from_int(1),
                InputFix::from_int(0),
                InputFix::from_int(0),
            );
            let checksum = session.advance_frame(input);
            assert_ne!(checksum.0, 0);
        }

        assert_eq!(session.frame(), 3);
        assert_eq!(session.frames_ahead(), 3); // remote hasn't sent anything
    }

    #[test]
    fn test_physics_rollback_session_rollback() {
        // `NetcodeConfig` は non_exhaustive なので struct literal も
        // `..Default::default()` も使えない 元の literal が明示していた
        // `player_count: 2` / `max_snapshots: 10` は default と同値なので、
        // 値を assert で固定する
        let netcode_config = NetcodeConfig::default();
        assert_eq!(netcode_config.player_count, 2);
        assert_eq!(netcode_config.max_snapshots, 10);
        let mut session = PhysicsRollbackSession::new(
            2,
            0,
            8,
            netcode_config,
            &alice_physics::PHYSICS_SEMANTICS_ID,
        )
        .expect("this build's identifier");

        // Add player bodies
        for i in 0..2u8 {
            let body = RigidBody::new_dynamic(Vec3Fix::from_int(i as i64 * 5, 10, 0), Fix128::ONE);
            let idx = session.sim.add_body(body);
            session.sim.assign_player_body(i, idx);
        }
        session.sim.add_body(RigidBody::new_static(Vec3Fix::ZERO));

        // Advance 3 frames (remote predicted as empty)
        for f in 1..=3u64 {
            let input = InputFrame::new(f, 0).with_movement(
                InputFix::from_int(1),
                InputFix::from_int(0),
                InputFix::from_int(0),
            );
            session.advance_frame(input);
        }

        // Remote sends frame 1 with different data → rollback
        let action = session.add_remote_input(InputFrame::new(1, 1).with_movement(
            InputFix::from_int(5),
            InputFix::from_int(0),
            InputFix::from_int(-5),
        ));
        assert_eq!(action, RollbackAction::Rollback { to_frame: 1 });

        // Handle the rollback
        let ok = session.handle_rollback(1);
        assert!(ok);
        assert_eq!(session.frame(), 3); // back to current frame after re-sim
    }

    #[test]
    fn test_batch_conversion() {
        let inputs = vec![
            InputFrame::new(1, 0).with_movement(
                InputFix::from_int(10),
                InputFix::ZERO,
                InputFix::ZERO,
            ),
            InputFrame::new(1, 1).with_movement(
                InputFix::ZERO,
                InputFix::ZERO,
                InputFix::from_int(-10),
            ),
        ];

        let physics = sync_inputs_to_physics(&inputs);
        assert_eq!(physics.len(), 2);
        assert_eq!(physics[0].movement.x.to_f64(), 10.0);
        assert_eq!(physics[1].movement.z.to_f64(), -10.0);
    }

    #[test]
    fn test_extreme_values_preserved() {
        let input = InputFrame::new(1, 0)
            .with_movement(
                InputFix::from_raw(i64::MIN, 0),
                InputFix::from_raw(i64::MAX, u64::MAX),
                InputFix::from_raw(-1, u64::MAX),
            )
            .with_aim(
                InputFix::from_int(-100),
                InputFix::from_int(40_000),
                InputFix::from_raw(0, 1),
            );

        let physics = sync_input_to_physics(&input);
        assert_eq!(
            (physics.movement.x.hi, physics.movement.x.lo),
            (i64::MIN, 0)
        );
        assert_eq!(
            (physics.movement.y.hi, physics.movement.y.lo),
            (i64::MAX, u64::MAX)
        );
        assert_eq!(physics.aim_direction.y.to_f64(), 40_000.0);

        let back = physics_input_to_sync(&physics, 1);
        assert_eq!(back, input);
    }
}

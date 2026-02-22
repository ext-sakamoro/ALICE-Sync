//! Python Bindings for ALICE-Sync (PyO3 + NumPy Zero-Copy)
//!
//! # Optimization Layers
//!
//! | Layer | Technique | Effect |
//! |-------|-----------|--------|
//! | L1 | GIL Release (`py.allow_threads`) | Parallel batch processing |
//! | L2 | Zero-Copy NumPy (`into_pyarray`) | No memcpy for positions/motions |
//! | L3 | Batch API (SoA world operations) | FFI amortization |
//! | L4 | Rust backend (8-wide SIMD, Demon Mode) | Hardware-speed apply |

use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, IntoPyArray};

use crate::input_sync::{InputFrame, InputBuffer, LockstepSession, RollbackSession, RollbackAction};
use crate::event::MotionData;
use crate::world_soa::WorldSoA;

// ============================================================================
// PyInputFrame
// ============================================================================

/// A single player's input for one simulation frame (~24 bytes).
///
/// Fixed-point i16 for deterministic network transmission.
#[pyclass(name = "InputFrame")]
#[derive(Clone)]
pub struct PyInputFrame {
    inner: InputFrame,
}

#[pymethods]
impl PyInputFrame {
    /// Create a new input frame.
    #[new]
    #[pyo3(signature = (frame, player_id, move_x=0, move_y=0, move_z=0, actions=0, aim_x=0, aim_y=0, aim_z=0))]
    fn new(
        frame: u64, player_id: u8,
        move_x: i16, move_y: i16, move_z: i16,
        actions: u32,
        aim_x: i16, aim_y: i16, aim_z: i16,
    ) -> Self {
        Self {
            inner: InputFrame::new(frame, player_id)
                .with_movement(move_x, move_y, move_z)
                .with_actions(actions)
                .with_aim(aim_x, aim_y, aim_z),
        }
    }

    /// Serialize to compact bitcode bytes.
    fn to_bytes(&self) -> Vec<u8> {
        self.inner.to_bytes()
    }

    /// Deserialize from bitcode bytes. Returns None on failure.
    #[staticmethod]
    fn from_bytes(data: Vec<u8>) -> Option<PyInputFrame> {
        InputFrame::from_bytes(&data).map(|inner| PyInputFrame { inner })
    }

    #[getter]
    fn frame(&self) -> u64 { self.inner.frame }

    #[getter]
    fn player_id(&self) -> u8 { self.inner.player_id }

    #[getter]
    fn movement(&self) -> (i16, i16, i16) {
        (self.inner.movement[0], self.inner.movement[1], self.inner.movement[2])
    }

    #[getter]
    fn actions(&self) -> u32 { self.inner.actions }

    #[getter]
    fn aim(&self) -> (i16, i16, i16) {
        (self.inner.aim[0], self.inner.aim[1], self.inner.aim[2])
    }

    fn __repr__(&self) -> String {
        format!(
            "<InputFrame frame={} player={} move=({},{},{}) actions=0x{:x}>",
            self.inner.frame, self.inner.player_id,
            self.inner.movement[0], self.inner.movement[1], self.inner.movement[2],
            self.inner.actions,
        )
    }
}

// ============================================================================
// PyLockstepSession
// ============================================================================

/// Lockstep input synchronization session.
///
/// Waits for all players' inputs before advancing.
/// Best for: RTS, turn-based, < 4 players.
#[pyclass(name = "LockstepSession")]
pub struct PyLockstepSession {
    inner: LockstepSession,
}

#[pymethods]
impl PyLockstepSession {
    #[new]
    fn new(player_count: u8) -> Self {
        Self { inner: LockstepSession::new(player_count) }
    }

    /// Add a local player's input.
    fn add_local_input(&mut self, frame: &PyInputFrame) {
        self.inner.add_local_input(frame.inner);
    }

    /// Add a remote player's input.
    fn add_remote_input(&mut self, frame: &PyInputFrame) {
        self.inner.add_remote_input(frame.inner);
    }

    /// Check if all inputs for next frame are available.
    fn ready_to_advance(&self) -> bool {
        self.inner.ready_to_advance()
    }

    /// Advance and collect all inputs. Returns list of InputFrame or None.
    fn advance(&mut self) -> Option<Vec<PyInputFrame>> {
        self.inner.advance().map(|inputs| {
            inputs.into_iter().map(|inner| PyInputFrame { inner }).collect()
        })
    }

    /// Record a checksum for a frame.
    fn record_checksum(&mut self, frame: u64, checksum: u64) {
        self.inner.record_checksum(frame, checksum);
    }

    /// Verify a remote checksum. Returns "ok" or "desync".
    fn verify_checksum(&self, frame: u64, remote: u64) -> String {
        match self.inner.verify_checksum(frame, remote) {
            crate::input_sync::SyncResult::Ok => "ok".to_string(),
            crate::input_sync::SyncResult::Desync { frame, local, remote } => {
                format!("desync:frame={},local={:x},remote={:x}", frame, local, remote)
            }
        }
    }

    /// Current confirmed frame.
    fn confirmed_frame(&self) -> u64 {
        self.inner.confirmed_frame()
    }

    fn __repr__(&self) -> String {
        format!("<LockstepSession confirmed_frame={}>", self.inner.confirmed_frame())
    }
}

// ============================================================================
// PyRollbackSession
// ============================================================================

/// Rollback (GGPO-style) input synchronization session.
///
/// Predicts remote inputs and advances immediately.
/// On mismatch, returns rollback instructions.
/// Best for: Fighting games, FPS, action games.
#[pyclass(name = "RollbackSession")]
pub struct PyRollbackSession {
    inner: RollbackSession,
}

#[pymethods]
impl PyRollbackSession {
    /// Create a new rollback session.
    ///
    /// Args:
    ///     player_count: Total number of players
    ///     local_player: This client's player ID
    ///     max_rollback: Max frames that can be rolled back (default: 8)
    #[new]
    #[pyo3(signature = (player_count, local_player, max_rollback=8))]
    fn new(player_count: u8, local_player: u8, max_rollback: u64) -> Self {
        Self {
            inner: RollbackSession::new(player_count, local_player, max_rollback),
        }
    }

    /// Add local player's input. Returns list of all inputs (confirmed + predicted).
    fn add_local_input(&mut self, frame: &PyInputFrame) -> Vec<PyInputFrame> {
        let inputs = self.inner.add_local_input(frame.inner);
        inputs.into_iter().map(|inner| PyInputFrame { inner }).collect()
    }

    /// Add remote player's confirmed input.
    ///
    /// Returns:
    ///     "none" — no action needed
    ///     "rollback:N" — must rollback to frame N
    ///     "desync:N" — unrecoverable desync at frame N
    fn add_remote_input(&mut self, frame: &PyInputFrame) -> String {
        match self.inner.add_remote_input(frame.inner) {
            RollbackAction::None => "none".to_string(),
            RollbackAction::Rollback { to_frame } => format!("rollback:{}", to_frame),
            RollbackAction::Desync { frame } => format!("desync:{}", frame),
        }
    }

    /// Save a state snapshot for potential rollback.
    fn save_snapshot(&mut self, frame: u64, state: Vec<u8>, checksum: u64) {
        self.inner.save_snapshot(frame, state, checksum);
    }

    /// Get a state snapshot. Returns state_bytes or None.
    fn get_snapshot(&self, frame: u64) -> Option<Vec<u8>> {
        self.inner.get_snapshot(frame).map(|s| s.to_vec())
    }

    /// Current predicted frame.
    fn predicted_frame(&self) -> u64 {
        self.inner.predicted_frame()
    }

    fn __repr__(&self) -> String {
        format!(
            "<RollbackSession predicted_frame={}>",
            self.inner.predicted_frame(),
        )
    }
}

// ============================================================================
// PyWorldSoA — SoA world with batch SIMD operations
// ============================================================================

/// Structure-of-Arrays world for high-performance batch operations.
///
/// Entities stored in decomposed arrays: pos_x[], pos_y[], pos_z[], ...
/// Enables 8-wide vertical SIMD for position updates.
#[pyclass(name = "WorldSoA")]
pub struct PyWorldSoA {
    inner: WorldSoA,
}

#[pymethods]
impl PyWorldSoA {
    /// Create a new SoA world.
    #[new]
    #[pyo3(signature = (seed=0))]
    fn new(seed: u64) -> Self {
        Self {
            inner: WorldSoA::new(seed),
        }
    }

    /// Spawn an entity. Returns slot index.
    fn spawn(&mut self, entity_id: u32, kind: u16, x: i32, y: i32, z: i32) -> u32 {
        self.inner.spawn(entity_id, kind, x, y, z)
    }

    /// Despawn an entity by slot.
    fn despawn(&mut self, slot: u32) -> bool {
        self.inner.despawn(slot)
    }

    /// Apply batch motions using Demon Mode (sort + coalesce + 8-wide SIMD).
    ///
    /// Args:
    ///     motions: list of `(entity_id, dx, dy, dz)` tuples
    ///
    /// GIL released for the batch operation.
    fn apply_motions(&mut self, py: Python<'_>, motions: Vec<(u32, i16, i16, i16)>) {
        let motion_data: Vec<MotionData> = motions.iter().map(|&(e, dx, dy, dz)| {
            MotionData { entity: e, delta_x: dx, delta_y: dy, delta_z: dz }
        }).collect();

        py.allow_threads(|| {
            self.inner.apply_motions_demon(motion_data);
        });
    }

    /// Get all entity positions as NumPy (N, 3) int32 array.
    ///
    /// Reads from SoA arrays directly — zero intermediate copy.
    fn positions<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<i32>> {
        let n = self.inner.storage.capacity;
        let mut data = Vec::with_capacity(n * 3);
        for i in 0..n {
            data.push(self.inner.storage.pos_x[i]);
            data.push(self.inner.storage.pos_y[i]);
            data.push(self.inner.storage.pos_z[i]);
        }
        data.into_pyarray(py).reshape([n, 3]).unwrap()
    }

    /// Get world hash (XOR rolling hash, O(1) amortized).
    fn hash(&self) -> u64 {
        self.inner.hash().0
    }

    /// Number of active entities.
    fn entity_count(&self) -> usize {
        self.inner.entity_count()
    }

    /// Total capacity.
    fn capacity(&self) -> usize {
        self.inner.storage.capacity
    }

    fn __repr__(&self) -> String {
        format!(
            "<WorldSoA entities={} capacity={} hash={:016x}>",
            self.inner.entity_count(),
            self.inner.storage.capacity,
            self.inner.hash().0,
        )
    }
}

/// Module version.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

// ============================================================================
// Module registration
// ============================================================================

#[pymodule]
fn alice_sync(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyInputFrame>()?;
    m.add_class::<PyLockstepSession>()?;
    m.add_class::<PyRollbackSession>()?;
    m.add_class::<PyWorldSoA>()?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_input_frame_roundtrip() {
        let frame = InputFrame::new(1, 0).with_movement(10, -5, 3).with_actions(0x7);
        let bytes = frame.to_bytes();
        let decoded = InputFrame::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.frame, 1);
        assert_eq!(decoded.player_id, 0);
        assert_eq!(decoded.movement, [10, -5, 3]);
        assert_eq!(decoded.actions, 0x7);
    }
}

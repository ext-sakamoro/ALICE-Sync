//! Deterministic Input Synchronization for Netcode
//!
//! Provides lockstep and rollback input synchronization protocols.
//! Combined with a deterministic physics engine (ALICE-Physics), only
//! player inputs need to be synchronized — state sync is unnecessary.
//!
//! # Architecture
//!
//! ```text
//! Client A                            Client B
//!   Local Input → InputBuffer           Local Input → InputBuffer
//!        ↓                                   ↓
//!   LockstepSession / RollbackSession   LockstepSession / RollbackSession
//!        ↓              ↑                    ↓              ↑
//!        └──── Network (InputFrame) ────────┘              │
//!                       │                                   │
//!              Checksum Verification ◄──────────────────────┘
//! ```
//!
//! # Modes
//!
//! - **Lockstep**: Waits for all players' inputs before advancing.
//!   Best for: RTS, turn-based, < 4 players.
//!
//! - **Rollback**: Predicts missing inputs, corrects on receipt.
//!   Best for: Fighting games, FPS, action games.
//!
//! # Bandwidth
//!
//! An `InputFrame` is ~24 bytes. At 60 fps with 4 players:
//! ```text
//! 24 bytes × 4 players × 60 fps = 5,760 bytes/sec ≈ 5.6 KB/s
//! ```
//! Compare to state sync: 160 bytes × 100 bodies × 60 fps = 960 KB/s

use serde::{Serialize, Deserialize};
use bitcode::{Encode, Decode};
use std::collections::VecDeque;

// ============================================================================
// Input Frame
// ============================================================================

/// A single player's input for a single simulation frame.
///
/// Compact and deterministic. Serialized to ~24 bytes via bitcode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Encode, Decode)]
pub struct InputFrame {
    /// Simulation frame number this input belongs to
    pub frame: u64,
    /// Player identifier
    pub player_id: u8,
    /// Movement delta as fixed-point i16 (Q8.8 or direct integer)
    pub movement: [i16; 3],
    /// Action bitfield (jump=0x1, fire=0x2, interact=0x4, etc.)
    pub actions: u32,
    /// Aim direction as fixed-point i16
    pub aim: [i16; 3],
}

impl InputFrame {
    /// Create a new empty input for the given frame and player
    #[inline]
    pub fn new(frame: u64, player_id: u8) -> Self {
        Self {
            frame,
            player_id,
            movement: [0, 0, 0],
            actions: 0,
            aim: [0, 0, 0],
        }
    }

    /// Set movement
    #[inline]
    pub fn with_movement(mut self, x: i16, y: i16, z: i16) -> Self {
        self.movement = [x, y, z];
        self
    }

    /// Set actions
    #[inline]
    pub fn with_actions(mut self, actions: u32) -> Self {
        self.actions = actions;
        self
    }

    /// Set aim direction
    #[inline]
    pub fn with_aim(mut self, x: i16, y: i16, z: i16) -> Self {
        self.aim = [x, y, z];
        self
    }

    /// Serialize with bitcode (compact binary)
    pub fn to_bytes(&self) -> Vec<u8> {
        bitcode::encode(self)
    }

    /// Deserialize from bitcode
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        bitcode::decode(data).ok()
    }
}

// ============================================================================
// Input Buffer (per-player ring buffer)
// ============================================================================

/// Per-player input ring buffer.
///
/// Stores received inputs indexed by frame number.
/// Supports both confirmed (received) and predicted (extrapolated) inputs.
#[derive(Debug)]
pub struct InputBuffer {
    /// Player ID this buffer belongs to
    pub player_id: u8,
    /// Stored inputs, ordered by frame
    inputs: VecDeque<InputFrame>,
    /// Lowest frame number in the buffer
    base_frame: u64,
    /// Last confirmed (received from network) frame
    pub confirmed_frame: u64,
    /// Last predicted frame (local extrapolation)
    pub predicted_frame: u64,
}

impl InputBuffer {
    /// Create a new input buffer for the given player
    pub fn new(player_id: u8) -> Self {
        Self {
            player_id,
            inputs: VecDeque::with_capacity(128),
            base_frame: 1,
            confirmed_frame: 0,
            predicted_frame: 0,
        }
    }

    /// Add a confirmed input (received from network or local).
    ///
    /// Returns `true` if the input was accepted (not a duplicate or too old).
    pub fn add_confirmed(&mut self, input: InputFrame) -> bool {
        if input.frame <= self.confirmed_frame {
            return false; // duplicate or too old
        }

        // Grow buffer to accommodate the frame
        while self.base_frame + self.inputs.len() as u64 <= input.frame {
            // Fill gaps with empty inputs (will be overwritten or predicted)
            self.inputs.push_back(InputFrame::new(
                self.base_frame + self.inputs.len() as u64,
                self.player_id,
            ));
        }

        let idx = (input.frame - self.base_frame) as usize;
        if idx < self.inputs.len() {
            self.inputs[idx] = input;
        }
        self.confirmed_frame = self.confirmed_frame.max(input.frame);
        true
    }

    /// Get input for a specific frame.
    ///
    /// Returns `None` if the frame is not in the buffer.
    pub fn get(&self, frame: u64) -> Option<&InputFrame> {
        if frame < self.base_frame {
            return None;
        }
        let idx = (frame - self.base_frame) as usize;
        self.inputs.get(idx)
    }

    /// Predict input for a future frame by repeating the last confirmed input.
    ///
    /// This is the simplest prediction strategy. For better prediction,
    /// games can implement momentum-based or intent-based prediction.
    pub fn predict(&mut self, frame: u64) -> InputFrame {
        // Use last confirmed input as prediction
        let prediction = if let Some(last) = self.get(self.confirmed_frame) {
            let mut predicted = *last;
            predicted.frame = frame;
            predicted
        } else {
            InputFrame::new(frame, self.player_id)
        };

        // Store prediction in buffer so add_remote_input can compare later
        while self.base_frame + self.inputs.len() as u64 <= frame {
            self.inputs.push_back(InputFrame::new(
                self.base_frame + self.inputs.len() as u64,
                self.player_id,
            ));
        }
        let idx = (frame - self.base_frame) as usize;
        if idx < self.inputs.len() {
            self.inputs[idx] = prediction;
        }

        self.predicted_frame = self.predicted_frame.max(frame);
        prediction
    }

    /// Check if input is confirmed for a specific frame.
    #[inline]
    pub fn is_confirmed(&self, frame: u64) -> bool {
        frame <= self.confirmed_frame
    }

    /// Trim inputs older than the given frame to free memory.
    pub fn trim_before(&mut self, frame: u64) {
        while self.base_frame < frame && !self.inputs.is_empty() {
            self.inputs.pop_front();
            self.base_frame += 1;
        }
    }
}

// ============================================================================
// Sync Result Types
// ============================================================================

/// Result of a checksum verification
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SyncResult {
    /// Checksums match — clients are in sync
    Ok,
    /// Checksums differ — DESYNC detected
    Desync {
        /// Frame where desync was detected
        frame: u64,
        /// Local checksum
        local: u64,
        /// Remote checksum
        remote: u64,
    },
}

/// Action the game should take after receiving a remote input
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RollbackAction {
    /// No action needed — input was for current or future frame
    None,
    /// Must rollback to this frame and re-simulate
    Rollback {
        /// Frame to rollback to
        to_frame: u64,
    },
    /// Desync detected — checksums differ even after rollback
    Desync {
        /// Frame where desync was detected
        frame: u64,
    },
}

// ============================================================================
// Lockstep Session
// ============================================================================

/// Lockstep input synchronization.
///
/// Waits for all players' inputs before advancing any frame.
/// Simplest and most reliable, but adds input latency equal to
/// the round-trip time of the slowest player.
///
/// Best for: RTS games, turn-based games, < 4 players.
#[derive(Debug)]
pub struct LockstepSession {
    /// Number of players
    player_count: u8,
    /// Per-player input buffers
    buffers: Vec<InputBuffer>,
    /// Last frame where all inputs are confirmed
    confirmed_frame: u64,
    /// Checksum history for verification: (frame, checksum)
    checksums: VecDeque<(u64, u64)>,
    /// Maximum checksum history
    max_checksum_history: usize,
}

impl LockstepSession {
    /// Create a new lockstep session
    pub fn new(player_count: u8) -> Self {
        let buffers = (0..player_count).map(InputBuffer::new).collect();
        Self {
            player_count,
            buffers,
            confirmed_frame: 0,
            checksums: VecDeque::with_capacity(128),
            max_checksum_history: 120,
        }
    }

    /// Add a local player's input for the next frame.
    pub fn add_local_input(&mut self, input: InputFrame) {
        let pid = input.player_id as usize;
        if pid < self.buffers.len() {
            self.buffers[pid].add_confirmed(input);
        }
    }

    /// Add a remote player's input.
    pub fn add_remote_input(&mut self, input: InputFrame) {
        let pid = input.player_id as usize;
        if pid < self.buffers.len() {
            self.buffers[pid].add_confirmed(input);
        }
    }

    /// Check if the next frame is ready to advance
    /// (all players have submitted input for confirmed_frame + 1).
    pub fn ready_to_advance(&self) -> bool {
        let next = self.confirmed_frame + 1;
        self.buffers.iter().all(|b| b.is_confirmed(next))
    }

    /// Collect all inputs for the next frame and advance.
    ///
    /// Returns `None` if not all inputs are ready.
    /// Returns `Some(inputs)` — one per player, sorted by player_id.
    pub fn advance(&mut self) -> Option<Vec<InputFrame>> {
        if !self.ready_to_advance() {
            return None;
        }

        let next = self.confirmed_frame + 1;
        let mut inputs = Vec::with_capacity(self.player_count as usize);
        for buf in &self.buffers {
            if let Some(input) = buf.get(next) {
                inputs.push(*input);
            }
        }

        self.confirmed_frame = next;
        Some(inputs)
    }

    /// Record a checksum for a frame (for cross-client verification).
    pub fn record_checksum(&mut self, frame: u64, checksum: u64) {
        self.checksums.push_back((frame, checksum));
        if self.checksums.len() > self.max_checksum_history {
            self.checksums.pop_front();
        }
    }

    /// Verify a remote checksum.
    pub fn verify_checksum(&self, frame: u64, remote_checksum: u64) -> SyncResult {
        if let Some(&(_, local)) = self.checksums.iter().find(|&&(f, _)| f == frame) {
            if local == remote_checksum {
                SyncResult::Ok
            } else {
                SyncResult::Desync {
                    frame,
                    local,
                    remote: remote_checksum,
                }
            }
        } else {
            // Frame not in history — assume OK (too old or too new)
            SyncResult::Ok
        }
    }

    /// Current confirmed frame.
    #[inline]
    pub fn confirmed_frame(&self) -> u64 {
        self.confirmed_frame
    }

    /// Trim old inputs to free memory.
    pub fn trim_before(&mut self, frame: u64) {
        for buf in &mut self.buffers {
            buf.trim_before(frame);
        }
        while self.checksums.front().map(|&(f, _)| f < frame).unwrap_or(false) {
            self.checksums.pop_front();
        }
    }
}

// ============================================================================
// Rollback Session
// ============================================================================

/// Rollback (GGPO-style) input synchronization.
///
/// Predicts remote inputs and advances immediately. When the actual remote
/// input arrives and differs from prediction, the simulation is rolled back
/// and re-simulated with correct inputs.
///
/// Best for: Fighting games, FPS, action games (low perceived latency).
///
/// # Flow
///
/// ```text
/// Frame N: Local input confirmed, remote predicted
///          → Advance simulation
/// Frame N+k: Remote input for frame N arrives
///   if matches prediction → no action
///   if differs → rollback to N, re-simulate N..current with corrected inputs
/// ```
#[derive(Debug)]
pub struct RollbackSession {
    /// Number of players
    player_count: u8,
    /// Local player ID
    local_player: u8,
    /// Per-player input buffers
    buffers: Vec<InputBuffer>,
    /// Last frame confirmed by all players
    confirmed_frame: u64,
    /// Current predicted frame (local simulation is ahead of confirmed)
    predicted_frame: u64,
    /// Maximum frames of rollback allowed
    max_rollback: u64,
    /// State snapshots for rollback: (frame, serialized_state)
    snapshots: VecDeque<(u64, Vec<u8>)>,
    /// Checksum history: (frame, checksum)
    checksums: VecDeque<(u64, u64)>,
}

impl RollbackSession {
    /// Create a new rollback session.
    ///
    /// Arguments:
    ///
    /// - `player_count`: Total number of players
    /// - `local_player`: This client's player ID
    /// - `max_rollback`: Maximum frames that can be rolled back (default: 8)
    pub fn new(player_count: u8, local_player: u8, max_rollback: u64) -> Self {
        let buffers = (0..player_count).map(InputBuffer::new).collect();
        Self {
            player_count,
            local_player,
            buffers,
            confirmed_frame: 0,
            predicted_frame: 0,
            max_rollback,
            snapshots: VecDeque::with_capacity(max_rollback as usize + 2),
            checksums: VecDeque::with_capacity(128),
        }
    }

    /// Add local player's input and get the inputs for the next predicted frame.
    ///
    /// Predicts remote players' inputs (repeat last confirmed).
    /// Returns all players' inputs for the frame to simulate.
    pub fn add_local_input(&mut self, input: InputFrame) -> Vec<InputFrame> {
        let frame = input.frame;
        self.buffers[self.local_player as usize].add_confirmed(input);

        // Collect inputs for this frame: confirmed or predicted
        let mut inputs = Vec::with_capacity(self.player_count as usize);
        for (i, buf) in self.buffers.iter_mut().enumerate() {
            if i == self.local_player as usize || buf.is_confirmed(frame) {
                inputs.push(*buf.get(frame).unwrap());
            } else {
                inputs.push(buf.predict(frame));
            }
        }

        self.predicted_frame = self.predicted_frame.max(frame);
        inputs
    }

    /// Add a remote player's confirmed input.
    ///
    /// Returns a `RollbackAction` indicating what the game should do:
    /// - `None`: Input was for a future frame or matched prediction
    /// - `Rollback { to_frame }`: Must rollback and re-simulate
    /// - `Desync { frame }`: Unrecoverable desync
    pub fn add_remote_input(&mut self, input: InputFrame) -> RollbackAction {
        let frame = input.frame;
        let pid = input.player_id as usize;

        if pid >= self.buffers.len() {
            return RollbackAction::None;
        }

        // Check if this input differs from what we predicted
        let needs_rollback = if frame <= self.predicted_frame {
            // We already simulated past this frame — check if prediction was correct
            if let Some(existing) = self.buffers[pid].get(frame) {
                // We had a confirmed/predicted input — compare
                existing != &input
            } else {
                // No input was stored for this frame (never predicted) — no mismatch
                false
            }
        } else {
            false
        };

        self.buffers[pid].add_confirmed(input);

        // Update confirmed frame
        self.update_confirmed_frame();

        if needs_rollback {
            if self.predicted_frame - frame > self.max_rollback {
                return RollbackAction::Desync { frame };
            }
            RollbackAction::Rollback { to_frame: frame }
        } else {
            RollbackAction::None
        }
    }

    /// Save a state snapshot for potential rollback.
    pub fn save_snapshot(&mut self, frame: u64, state: Vec<u8>, checksum: u64) {
        self.snapshots.push_back((frame, state));
        self.checksums.push_back((frame, checksum));

        // Trim old snapshots beyond rollback window
        let min_frame = self.confirmed_frame.saturating_sub(1);
        while self.snapshots.front().map(|&(f, _)| f < min_frame).unwrap_or(false) {
            self.snapshots.pop_front();
        }
        while self.checksums.front().map(|&(f, _)| f < min_frame).unwrap_or(false) {
            self.checksums.pop_front();
        }
    }

    /// Get the snapshot for rollback to a specific frame.
    pub fn get_snapshot(&self, frame: u64) -> Option<&[u8]> {
        self.snapshots.iter()
            .find(|&&(f, _)| f == frame)
            .map(|(_, state)| state.as_slice())
    }

    /// Get all inputs for a specific frame (confirmed + predicted).
    ///
    /// Used during re-simulation after rollback.
    pub fn inputs_for_frame(&mut self, frame: u64) -> Vec<InputFrame> {
        let mut inputs = Vec::with_capacity(self.player_count as usize);
        for (i, buf) in self.buffers.iter_mut().enumerate() {
            if buf.is_confirmed(frame) {
                if let Some(input) = buf.get(frame) {
                    inputs.push(*input);
                } else {
                    inputs.push(InputFrame::new(frame, i as u8));
                }
            } else {
                inputs.push(buf.predict(frame));
            }
        }
        inputs
    }

    /// Verify a remote checksum.
    pub fn verify_checksum(&self, frame: u64, remote_checksum: u64) -> SyncResult {
        if let Some(&(_, local)) = self.checksums.iter().find(|&&(f, _)| f == frame) {
            if local == remote_checksum {
                SyncResult::Ok
            } else {
                SyncResult::Desync {
                    frame,
                    local,
                    remote: remote_checksum,
                }
            }
        } else {
            SyncResult::Ok
        }
    }

    /// Current confirmed frame (all players' inputs received).
    #[inline]
    pub fn confirmed_frame(&self) -> u64 {
        self.confirmed_frame
    }

    /// Current predicted frame (local simulation is at this frame).
    #[inline]
    pub fn predicted_frame(&self) -> u64 {
        self.predicted_frame
    }

    /// Number of frames ahead of confirmation (rollback risk).
    #[inline]
    pub fn frames_ahead(&self) -> u64 {
        self.predicted_frame.saturating_sub(self.confirmed_frame)
    }

    /// Trim old data before the given frame.
    pub fn trim_before(&mut self, frame: u64) {
        for buf in &mut self.buffers {
            buf.trim_before(frame);
        }
    }

    /// Update the confirmed frame based on all players' confirmed status.
    fn update_confirmed_frame(&mut self) {
        let min_confirmed = self.buffers.iter()
            .map(|b| b.confirmed_frame)
            .min()
            .unwrap_or(0);
        self.confirmed_frame = min_confirmed;
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- InputFrame Tests ---

    #[test]
    fn test_input_frame_serialization() {
        let input = InputFrame::new(42, 1)
            .with_movement(100, 0, -50)
            .with_actions(0x05)
            .with_aim(0, 1000, 0);

        let bytes = input.to_bytes();
        let restored = InputFrame::from_bytes(&bytes).unwrap();

        assert_eq!(restored.frame, 42);
        assert_eq!(restored.player_id, 1);
        assert_eq!(restored.movement, [100, 0, -50]);
        assert_eq!(restored.actions, 0x05);
        assert_eq!(restored.aim, [0, 1000, 0]);
    }

    #[test]
    fn test_input_frame_compact() {
        let input = InputFrame::new(1, 0).with_movement(1, 0, 0);
        let bytes = input.to_bytes();
        // bitcode should produce very compact output
        assert!(bytes.len() < 30, "InputFrame too large: {} bytes", bytes.len());
    }

    // --- InputBuffer Tests ---

    #[test]
    fn test_input_buffer_basic() {
        let mut buf = InputBuffer::new(0);

        let input1 = InputFrame::new(1, 0).with_movement(1, 0, 0);
        let input2 = InputFrame::new(2, 0).with_movement(2, 0, 0);

        assert!(buf.add_confirmed(input1));
        assert!(buf.add_confirmed(input2));

        assert_eq!(buf.get(1).unwrap().movement[0], 1);
        assert_eq!(buf.get(2).unwrap().movement[0], 2);
        assert_eq!(buf.confirmed_frame, 2);
    }

    #[test]
    fn test_input_buffer_duplicate_rejected() {
        let mut buf = InputBuffer::new(0);
        let input = InputFrame::new(1, 0);

        assert!(buf.add_confirmed(input));
        assert!(!buf.add_confirmed(input)); // duplicate
    }

    #[test]
    fn test_input_buffer_predict() {
        let mut buf = InputBuffer::new(0);
        let input = InputFrame::new(1, 0).with_movement(5, 0, 0);
        buf.add_confirmed(input);

        // Predict frame 2 → repeats frame 1's input
        let predicted = buf.predict(2);
        assert_eq!(predicted.frame, 2);
        assert_eq!(predicted.movement[0], 5);
    }

    #[test]
    fn test_input_buffer_trim() {
        let mut buf = InputBuffer::new(0);
        for f in 1..=10 {
            buf.add_confirmed(InputFrame::new(f, 0));
        }

        buf.trim_before(5);
        assert!(buf.get(4).is_none()); // trimmed
        assert!(buf.get(5).is_some()); // still here
    }

    // --- LockstepSession Tests ---

    #[test]
    fn test_lockstep_basic() {
        let mut session = LockstepSession::new(2);

        // Frame 1: only player 0 ready
        session.add_local_input(InputFrame::new(1, 0).with_movement(1, 0, 0));
        assert!(!session.ready_to_advance());

        // Frame 1: player 1 also ready
        session.add_remote_input(InputFrame::new(1, 1).with_movement(0, 0, -1));
        assert!(session.ready_to_advance());

        // Advance
        let inputs = session.advance().unwrap();
        assert_eq!(inputs.len(), 2);
        assert_eq!(inputs[0].movement[0], 1);
        assert_eq!(inputs[1].movement[2], -1);
        assert_eq!(session.confirmed_frame(), 1);
    }

    #[test]
    fn test_lockstep_waits_for_all() {
        let mut session = LockstepSession::new(3);

        session.add_local_input(InputFrame::new(1, 0));
        session.add_remote_input(InputFrame::new(1, 1));
        assert!(!session.ready_to_advance()); // waiting for player 2

        session.add_remote_input(InputFrame::new(1, 2));
        assert!(session.ready_to_advance());
    }

    #[test]
    fn test_lockstep_checksum_verify() {
        let mut session = LockstepSession::new(2);
        session.record_checksum(1, 0xABCD);

        assert_eq!(session.verify_checksum(1, 0xABCD), SyncResult::Ok);
        assert_eq!(
            session.verify_checksum(1, 0xDEAD),
            SyncResult::Desync { frame: 1, local: 0xABCD, remote: 0xDEAD },
        );
    }

    // --- RollbackSession Tests ---

    #[test]
    fn test_rollback_no_mismatch() {
        let mut session = RollbackSession::new(2, 0, 8);

        // Local input for frame 1
        let inputs = session.add_local_input(InputFrame::new(1, 0).with_movement(1, 0, 0));
        assert_eq!(inputs.len(), 2);

        // Remote input matches prediction (empty)
        let action = session.add_remote_input(InputFrame::new(1, 1));
        assert_eq!(action, RollbackAction::None);
    }

    #[test]
    fn test_rollback_mismatch_triggers_rollback() {
        let mut session = RollbackSession::new(2, 0, 8);

        // Frame 1: local input, predict remote as empty
        session.add_local_input(InputFrame::new(1, 0).with_movement(1, 0, 0));

        // Frame 2: advance further
        session.add_local_input(InputFrame::new(2, 0).with_movement(1, 0, 0));

        // Remote input for frame 1 arrives with different data than predicted
        let action = session.add_remote_input(
            InputFrame::new(1, 1).with_movement(5, 5, 5)
        );
        assert_eq!(action, RollbackAction::Rollback { to_frame: 1 });
    }

    #[test]
    fn test_rollback_snapshot_save_restore() {
        let mut session = RollbackSession::new(2, 0, 8);

        // Save snapshot for frame 1
        let fake_state = vec![1, 2, 3, 4, 5];
        session.save_snapshot(1, fake_state.clone(), 0xAAAA);

        // Retrieve
        let restored = session.get_snapshot(1).unwrap();
        assert_eq!(restored, &[1, 2, 3, 4, 5]);

        // Checksum verify
        assert_eq!(session.verify_checksum(1, 0xAAAA), SyncResult::Ok);
        assert_eq!(
            session.verify_checksum(1, 0xBBBB),
            SyncResult::Desync { frame: 1, local: 0xAAAA, remote: 0xBBBB },
        );
    }

    #[test]
    fn test_rollback_frames_ahead() {
        let mut session = RollbackSession::new(2, 0, 8);

        session.add_local_input(InputFrame::new(1, 0));
        session.add_local_input(InputFrame::new(2, 0));
        session.add_local_input(InputFrame::new(3, 0));

        // Remote hasn't sent anything yet
        assert_eq!(session.frames_ahead(), 3);

        // Remote confirms frame 1
        session.add_remote_input(InputFrame::new(1, 1));
        assert_eq!(session.confirmed_frame(), 1);
        assert_eq!(session.frames_ahead(), 2);
    }

    #[test]
    fn test_rollback_inputs_for_frame() {
        let mut session = RollbackSession::new(2, 0, 8);

        // Confirmed inputs for both players on frame 1
        session.buffers[0].add_confirmed(InputFrame::new(1, 0).with_movement(10, 0, 0));
        session.buffers[1].add_confirmed(InputFrame::new(1, 1).with_movement(0, 0, -10));

        let inputs = session.inputs_for_frame(1);
        assert_eq!(inputs.len(), 2);
        assert_eq!(inputs[0].movement[0], 10);
        assert_eq!(inputs[1].movement[2], -10);
    }

    #[test]
    fn test_rollback_desync_on_excessive_rollback() {
        let mut session = RollbackSession::new(2, 0, 2); // max_rollback = 2

        // Advance 5 frames ahead
        for f in 1..=5 {
            session.add_local_input(InputFrame::new(f, 0));
        }

        // Remote sends frame 1 with different data → needs rollback of 4 frames > max 2
        let action = session.add_remote_input(
            InputFrame::new(1, 1).with_movement(99, 99, 99)
        );
        assert_eq!(action, RollbackAction::Desync { frame: 1 });
    }
}

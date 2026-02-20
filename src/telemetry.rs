//! Sync Telemetry via ALICE-DB
//!
//! Records network synchronization metrics as time-series data in ALICE-DB.
//! ALICE-DB's model-based compression fits telemetry naturally:
//! - Stable RTT → constant model (1 coefficient)
//! - Gradually improving prediction → linear model
//! - Periodic jitter → Fourier model
//!
//! # Channel Layout
//!
//! ```text
//! Channel 0: rollback_count   — frames rolled back per event
//! Channel 1: desync_severity  — 0.0=none, 1.0=fatal
//! Channel 2: prediction_accuracy — 0.0..1.0
//! Channel 3: rtt_ms           — round-trip time in milliseconds
//! Channel 4: input_delay      — input delay in frames
//! ```
//!
//! Key encoding: `timestamp = channel * MAX_FRAMES + frame`
//!
//! # Example
//!
//! ```rust,ignore
//! use alice_sync::telemetry::SyncTelemetry;
//!
//! let telemetry = SyncTelemetry::new("./telemetry_data").unwrap();
//!
//! // Record during gameplay
//! telemetry.record_rtt(frame, 15.5).unwrap();
//! telemetry.record_prediction_accuracy(frame, 0.95).unwrap();
//!
//! // On rollback event
//! telemetry.record_rollback(frame, 3).unwrap();
//!
//! // Query for analysis
//! let rtt_data = telemetry.scan_rtt(0, 3600).unwrap();
//! let avg_rtt = telemetry.average_rtt(0, 3600).unwrap();
//! ```

use alice_db::{AliceDB, Aggregation};
use std::io;
use std::path::Path;

/// Maximum frames per channel (~46 hours at 60fps)
const MAX_FRAMES: i64 = 10_000_000;

// Channel IDs
const CH_ROLLBACK: i64 = 0;
const CH_DESYNC: i64 = 1;
const CH_PREDICTION: i64 = 2;
const CH_RTT: i64 = 3;
const CH_INPUT_DELAY: i64 = 4;

/// Telemetry channel identifiers for external queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TelemetryChannel {
    /// Frames rolled back per rollback event
    RollbackCount,
    /// Desync severity (0.0 = none, 1.0 = fatal)
    DesyncSeverity,
    /// Input prediction accuracy (0.0..1.0)
    PredictionAccuracy,
    /// Round-trip time in milliseconds
    RttMs,
    /// Input delay in frames
    InputDelay,
}

impl TelemetryChannel {
    fn id(self) -> i64 {
        match self {
            Self::RollbackCount => CH_ROLLBACK,
            Self::DesyncSeverity => CH_DESYNC,
            Self::PredictionAccuracy => CH_PREDICTION,
            Self::RttMs => CH_RTT,
            Self::InputDelay => CH_INPUT_DELAY,
        }
    }
}

/// Records sync telemetry metrics to ALICE-DB.
///
/// Each metric type is stored in a separate channel within a single DB.
/// ALICE-DB's model fitter automatically selects the best compression:
/// - Stable metrics → constant model
/// - Trending metrics → polynomial model
/// - Periodic jitter → Fourier model
pub struct SyncTelemetry {
    db: AliceDB,
}

impl SyncTelemetry {
    /// Create a new telemetry recorder.
    pub fn new<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let db = AliceDB::open(path)?;
        Ok(Self { db })
    }

    /// Internal helper: write a value to a channel.
    #[inline(always)]
    fn put_channel(&self, channel: i64, frame: u64, value: f32) -> io::Result<()> {
        self.db.put(channel * MAX_FRAMES + frame as i64, value)
    }

    // ── Recording ──────────────────────────────────────────────

    /// Record a rollback event (how many frames were rolled back).
    #[inline(always)]
    pub fn record_rollback(&self, frame: u64, rollback_frames: u32) -> io::Result<()> {
        self.put_channel(CH_ROLLBACK, frame, rollback_frames as f32)
    }

    /// Record a desync event with severity (0.0 = minor, 1.0 = fatal).
    #[inline(always)]
    pub fn record_desync(&self, frame: u64, severity: f32) -> io::Result<()> {
        self.put_channel(CH_DESYNC, frame, severity)
    }

    /// Record input prediction accuracy for this frame (0.0..1.0).
    #[inline(always)]
    pub fn record_prediction_accuracy(&self, frame: u64, accuracy: f32) -> io::Result<()> {
        self.put_channel(CH_PREDICTION, frame, accuracy)
    }

    /// Record round-trip time in milliseconds.
    #[inline(always)]
    pub fn record_rtt(&self, frame: u64, rtt_ms: f32) -> io::Result<()> {
        self.put_channel(CH_RTT, frame, rtt_ms)
    }

    /// Record input delay in frames.
    #[inline(always)]
    pub fn record_input_delay(&self, frame: u64, delay_frames: u32) -> io::Result<()> {
        self.put_channel(CH_INPUT_DELAY, frame, delay_frames as f32)
    }

    /// Record multiple metrics for a single frame in one batch insert.
    ///
    /// Single lock acquisition instead of one per metric — up to 5x fewer
    /// lock round-trips when recording all channels per frame.
    #[inline(always)]
    pub fn record_batch(&self, frame: u64, rtt_ms: Option<f32>, prediction_accuracy: Option<f32>, rollback_frames: Option<u32>, desync_severity: Option<f32>, input_delay: Option<u32>) -> io::Result<()> {
        let f = frame as i64;
        let mut batch = [(0i64, 0.0f32); 5];
        let mut len = 0usize;

        if let Some(v) = rtt_ms {
            batch[len] = (CH_RTT * MAX_FRAMES + f, v);
            len += 1;
        }
        if let Some(v) = prediction_accuracy {
            batch[len] = (CH_PREDICTION * MAX_FRAMES + f, v);
            len += 1;
        }
        if let Some(v) = rollback_frames {
            batch[len] = (CH_ROLLBACK * MAX_FRAMES + f, v as f32);
            len += 1;
        }
        if let Some(v) = desync_severity {
            batch[len] = (CH_DESYNC * MAX_FRAMES + f, v);
            len += 1;
        }
        if let Some(v) = input_delay {
            batch[len] = (CH_INPUT_DELAY * MAX_FRAMES + f, v as f32);
            len += 1;
        }

        if len > 0 {
            self.db.put_batch(&batch[..len])?;
        }
        Ok(())
    }

    // ── Point Query ────────────────────────────────────────────

    /// Get a single metric value at a frame.
    #[inline(always)]
    pub fn get(&self, channel: TelemetryChannel, frame: u64) -> io::Result<Option<f32>> {
        self.db.get(channel.id() * MAX_FRAMES + frame as i64)
    }

    // ── Range Queries ──────────────────────────────────────────

    /// Scan a channel over a frame range.
    /// Returns `(frame, value)` pairs (frame is relative to start).
    pub fn scan(
        &self,
        channel: TelemetryChannel,
        start_frame: u64,
        end_frame: u64,
    ) -> io::Result<Vec<(u64, f32)>> {
        let base = channel.id() * MAX_FRAMES;
        let raw = self.db.scan(base + start_frame as i64, base + end_frame as i64)?;
        Ok(raw.into_iter().map(|(t, v)| ((t - base) as u64, v)).collect())
    }

    /// Scan rollback events in a frame range.
    pub fn scan_rollbacks(&self, start: u64, end: u64) -> io::Result<Vec<(u64, f32)>> {
        self.scan(TelemetryChannel::RollbackCount, start, end)
    }

    /// Scan RTT measurements in a frame range.
    pub fn scan_rtt(&self, start: u64, end: u64) -> io::Result<Vec<(u64, f32)>> {
        self.scan(TelemetryChannel::RttMs, start, end)
    }

    // ── Aggregation ────────────────────────────────────────────

    /// Compute average RTT over a frame range.
    pub fn average_rtt(&self, start: u64, end: u64) -> io::Result<f64> {
        let base = CH_RTT * MAX_FRAMES;
        self.db.aggregate(base + start as i64, base + end as i64, Aggregation::Avg)
    }

    /// Compute max rollback depth over a frame range.
    pub fn max_rollback(&self, start: u64, end: u64) -> io::Result<f64> {
        let base = CH_ROLLBACK * MAX_FRAMES;
        self.db.aggregate(base + start as i64, base + end as i64, Aggregation::Max)
    }

    /// Count desync events in a frame range.
    pub fn desync_count(&self, start: u64, end: u64) -> io::Result<f64> {
        let base = CH_DESYNC * MAX_FRAMES;
        self.db.aggregate(base + start as i64, base + end as i64, Aggregation::Count)
    }

    /// Average prediction accuracy over a frame range.
    pub fn average_prediction_accuracy(&self, start: u64, end: u64) -> io::Result<f64> {
        let base = CH_PREDICTION * MAX_FRAMES;
        self.db.aggregate(base + start as i64, base + end as i64, Aggregation::Avg)
    }

    // ── Lifecycle ──────────────────────────────────────────────

    /// Flush buffered data to disk.
    pub fn flush(&self) -> io::Result<()> {
        self.db.flush()
    }

    /// Close the telemetry recorder.
    pub fn close(self) -> io::Result<()> {
        self.db.flush()?;
        self.db.close()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_telemetry_rtt_recording() {
        let dir = tempfile::tempdir().unwrap();
        // Use separate DB for RTT only — ALICE-DB fits models per flush
        let path = dir.path().join("telemetry_rtt");

        let telemetry = SyncTelemetry::new(&path).unwrap();

        // Record RTT data only (contiguous keys in one channel)
        for frame in 0..100u64 {
            telemetry.record_rtt(frame, 15.0 + (frame as f32) * 0.1).unwrap();
        }
        telemetry.flush().unwrap();

        // Scan RTT data back
        let rtt_data = telemetry.scan_rtt(0, 99).unwrap();
        assert!(!rtt_data.is_empty(), "Should have RTT data after flush");

        telemetry.close().unwrap();
    }

    #[test]
    fn test_telemetry_rollback_recording() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("telemetry_rollback");

        let telemetry = SyncTelemetry::new(&path).unwrap();

        // Record rollback events only
        for frame in 0..50u64 {
            telemetry.record_rollback(frame, (frame % 5) as u32).unwrap();
        }
        telemetry.flush().unwrap();

        // Scan rollback data
        let rollbacks = telemetry.scan_rollbacks(0, 49).unwrap();
        assert!(!rollbacks.is_empty(), "Should have rollback data after flush");

        telemetry.close().unwrap();
    }

    #[test]
    fn test_telemetry_api_completeness() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("telemetry_api");

        let telemetry = SyncTelemetry::new(&path).unwrap();

        // Verify all recording APIs work without error
        telemetry.record_rtt(0, 15.0).unwrap();
        telemetry.record_rollback(1, 3).unwrap();
        telemetry.record_desync(2, 0.5).unwrap();
        telemetry.record_prediction_accuracy(3, 0.95).unwrap();
        telemetry.record_input_delay(4, 2).unwrap();

        telemetry.close().unwrap();
    }
}

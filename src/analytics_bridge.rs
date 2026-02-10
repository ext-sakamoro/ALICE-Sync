//! ALICE-Analytics bridge: Sync metrics telemetry
//!
//! Feeds real-time synchronisation metrics (event throughput, hash
//! mismatches, rollback frequency) into ALICE-Analytics sketches for
//! monitoring and anomaly detection.

use alice_analytics::prelude::*;

/// Sync telemetry collector backed by ALICE-Analytics sketches.
pub struct SyncTelemetry {
    /// Event throughput per second (quantile estimation).
    throughput: DDSketch256,
    /// Round-trip latency in microseconds.
    latency: DDSketch256,
    /// Unique peer count (cardinality estimation).
    peers: HyperLogLog12,
    /// Event type frequency (count-min sketch).
    event_freq: CountMinSketch1024x5,
    /// Hash mismatch / divergence counter.
    divergence_count: u64,
    /// Total events processed.
    total_events: u64,
}

impl SyncTelemetry {
    /// Create a new telemetry collector.
    pub fn new() -> Self {
        Self {
            throughput: DDSketch256::new(0.01),
            latency: DDSketch256::new(0.01),
            peers: HyperLogLog12::new(),
            event_freq: CountMinSketch1024x5::new(),
            divergence_count: 0,
            total_events: 0,
        }
    }

    /// Record an event batch throughput measurement.
    pub fn record_throughput(&mut self, events_per_sec: f64) {
        self.throughput.insert(events_per_sec);
    }

    /// Record a round-trip latency measurement.
    pub fn record_latency(&mut self, latency_us: f64) {
        self.latency.insert(latency_us);
    }

    /// Record a peer observation (for unique peer counting).
    pub fn record_peer(&mut self, peer_id: &[u8]) {
        self.peers.insert_bytes(peer_id);
    }

    /// Record an event type observation.
    pub fn record_event_type(&mut self, event_type: &[u8]) {
        self.event_freq.insert_bytes(event_type);
        self.total_events += 1;
    }

    /// Record a state divergence (hash mismatch).
    pub fn record_divergence(&mut self) {
        self.divergence_count += 1;
    }

    /// Estimated p50 throughput.
    pub fn throughput_p50(&self) -> f64 {
        self.throughput.quantile(0.5)
    }

    /// Estimated p99 throughput.
    pub fn throughput_p99(&self) -> f64 {
        self.throughput.quantile(0.99)
    }

    /// Estimated p50 latency.
    pub fn latency_p50(&self) -> f64 {
        self.latency.quantile(0.5)
    }

    /// Estimated p99 latency.
    pub fn latency_p99(&self) -> f64 {
        self.latency.quantile(0.99)
    }

    /// Estimated unique peer count.
    pub fn unique_peers(&self) -> f64 {
        self.peers.cardinality()
    }

    /// Estimated frequency of an event type.
    pub fn event_type_count(&self, event_type: &[u8]) -> u64 {
        self.event_freq.estimate_bytes(event_type)
    }

    /// Total divergence count.
    pub fn divergences(&self) -> u64 {
        self.divergence_count
    }

    /// Total events processed.
    pub fn total_events(&self) -> u64 {
        self.total_events
    }
}

impl Default for SyncTelemetry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_telemetry() {
        let mut tel = SyncTelemetry::new();

        for i in 0..100 {
            tel.record_throughput(1000.0 + i as f64);
            tel.record_latency(50.0 + (i % 10) as f64);
            tel.record_peer(format!("peer-{}", i % 8).as_bytes());
            tel.record_event_type(b"motion");
        }
        tel.record_divergence();

        assert!(tel.throughput_p50() > 0.0);
        assert!(tel.latency_p50() > 0.0);
        assert!(tel.unique_peers() >= 1.0);
        assert!(tel.event_type_count(b"motion") >= 90);
        assert_eq!(tel.divergences(), 1);
        assert_eq!(tel.total_events(), 100);
    }
}

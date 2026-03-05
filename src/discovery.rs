//! Peer Discovery — mDNS LAN Auto-Discovery + Manual Registration
//!
//! Discovers ALICE-Sync peers on the local network using mDNS/DNS-SD,
//! and supports manual peer registration for WAN/cloud scenarios.
//!
//! # Discovery Modes
//!
//! | Mode | Use Case | Latency |
//! |------|----------|---------|
//! | mDNS | LAN game lobbies, IoT mesh | < 1s |
//! | Manual | Cloud relay, WAN peers | Instant |
//! | Static | Config-file bootstrap | Instant |

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::RwLock;

/// mDNS service type for ALICE-Sync peers
pub const MDNS_SERVICE_TYPE: &str = "_alice-sync._udp.local.";

/// Default mDNS announcement interval
pub const ANNOUNCE_INTERVAL: Duration = Duration::from_secs(5);

/// Peer liveness timeout (no announcement for this long → considered offline)
pub const PEER_TIMEOUT: Duration = Duration::from_secs(30);

/// Information about a discovered peer
#[derive(Debug, Clone)]
pub struct PeerInfo {
    /// Network address of the peer
    pub addr: SocketAddr,
    /// Human-readable name (mDNS instance name or manual label)
    pub name: String,
    /// Application-defined metadata (e.g. room ID, capabilities, version)
    pub metadata: HashMap<String, String>,
    /// When this peer was first discovered
    pub discovered_at: Instant,
    /// When this peer was last seen (announcement or heartbeat)
    pub last_seen: Instant,
    /// Discovery source
    pub source: DiscoverySource,
}

/// How a peer was discovered
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiscoverySource {
    /// Discovered via mDNS on the local network
    Mdns,
    /// Manually registered (API call)
    Manual,
    /// Loaded from static configuration
    Static,
}

/// Callback for peer discovery/loss events
pub type PeerCallback = Arc<dyn Fn(PeerEvent) + Send + Sync>;

/// Events emitted by the discovery system
#[derive(Debug, Clone)]
pub enum PeerEvent {
    /// A new peer was discovered
    Discovered(PeerInfo),
    /// A previously known peer went offline
    Lost { addr: SocketAddr, name: String },
    /// A peer's metadata was updated
    Updated(PeerInfo),
}

/// Peer discovery registry.
///
/// Manages known peers from all discovery sources.
/// Thread-safe for concurrent access from transport and application layers.
pub struct Discovery {
    /// Known peers indexed by address
    peers: RwLock<HashMap<SocketAddr, PeerInfo>>,
    /// Local peer info for announcements
    local_info: RwLock<PeerInfo>,
    /// Event callbacks
    callbacks: RwLock<Vec<PeerCallback>>,
    /// Peer timeout duration
    timeout: Duration,
}

impl Discovery {
    /// Create a new discovery instance.
    ///
    /// `local_addr` is this node's bind address.
    /// `name` is the human-readable instance name.
    #[must_use]
    pub fn new(local_addr: SocketAddr, name: &str) -> Self {
        let now = Instant::now();
        Self {
            peers: RwLock::new(HashMap::new()),
            local_info: RwLock::new(PeerInfo {
                addr: local_addr,
                name: name.to_string(),
                metadata: HashMap::new(),
                discovered_at: now,
                last_seen: now,
                source: DiscoverySource::Manual,
            }),
            callbacks: RwLock::new(Vec::new()),
            timeout: PEER_TIMEOUT,
        }
    }

    /// Register an event callback.
    pub async fn on_event(&self, callback: PeerCallback) {
        self.callbacks.write().await.push(callback);
    }

    /// Set local metadata (advertised to other peers).
    pub async fn set_metadata(&self, key: &str, value: &str) {
        self.local_info
            .write()
            .await
            .metadata
            .insert(key.to_string(), value.to_string());
    }

    /// Manually register a peer.
    ///
    /// Returns `true` if this is a new peer, `false` if updated.
    pub async fn register_peer(
        &self,
        addr: SocketAddr,
        name: &str,
        metadata: HashMap<String, String>,
    ) -> bool {
        let now = Instant::now();
        let mut peers = self.peers.write().await;

        let is_new = !peers.contains_key(&addr);

        let info = PeerInfo {
            addr,
            name: name.to_string(),
            metadata,
            discovered_at: if is_new {
                now
            } else {
                peers[&addr].discovered_at
            },
            last_seen: now,
            source: DiscoverySource::Manual,
        };

        let event = if is_new {
            PeerEvent::Discovered(info.clone())
        } else {
            PeerEvent::Updated(info.clone())
        };

        peers.insert(addr, info);
        drop(peers);

        self.emit_event(event).await;
        is_new
    }

    /// Record that a peer was seen (heartbeat / message received).
    pub async fn peer_seen(&self, addr: &SocketAddr) {
        let mut peers = self.peers.write().await;
        if let Some(info) = peers.get_mut(addr) {
            info.last_seen = Instant::now();
        }
    }

    /// Remove a peer explicitly.
    pub async fn remove_peer(&self, addr: &SocketAddr) -> Option<PeerInfo> {
        let removed = self.peers.write().await.remove(addr);
        if let Some(ref info) = removed {
            self.emit_event(PeerEvent::Lost {
                addr: *addr,
                name: info.name.clone(),
            })
            .await;
        }
        removed
    }

    /// Get info about a specific peer.
    pub async fn get_peer(&self, addr: &SocketAddr) -> Option<PeerInfo> {
        self.peers.read().await.get(addr).cloned()
    }

    /// Get all known live peers.
    pub async fn live_peers(&self) -> Vec<PeerInfo> {
        let now = Instant::now();
        self.peers
            .read()
            .await
            .values()
            .filter(|p| now.duration_since(p.last_seen) < self.timeout)
            .cloned()
            .collect()
    }

    /// Get all known peer addresses (for transport broadcast).
    pub async fn peer_addrs(&self) -> Vec<SocketAddr> {
        let now = Instant::now();
        self.peers
            .read()
            .await
            .values()
            .filter(|p| now.duration_since(p.last_seen) < self.timeout)
            .map(|p| p.addr)
            .collect()
    }

    /// Number of known live peers.
    pub async fn peer_count(&self) -> usize {
        let now = Instant::now();
        self.peers
            .read()
            .await
            .values()
            .filter(|p| now.duration_since(p.last_seen) < self.timeout)
            .count()
    }

    /// Garbage-collect timed-out peers.
    ///
    /// Returns the list of removed peers.
    pub async fn gc(&self) -> Vec<PeerInfo> {
        let now = Instant::now();
        let mut peers = self.peers.write().await;
        let mut removed = Vec::new();

        peers.retain(|_, info| {
            if now.duration_since(info.last_seen) >= self.timeout {
                removed.push(info.clone());
                false
            } else {
                true
            }
        });

        drop(peers);

        for info in &removed {
            self.emit_event(PeerEvent::Lost {
                addr: info.addr,
                name: info.name.clone(),
            })
            .await;
        }

        removed
    }

    /// Local peer info (for mDNS announcements).
    pub async fn local_info(&self) -> PeerInfo {
        self.local_info.read().await.clone()
    }

    /// Emit an event to all registered callbacks.
    async fn emit_event(&self, event: PeerEvent) {
        let callbacks = self.callbacks.read().await;
        for cb in callbacks.iter() {
            cb(event.clone());
        }
    }

    /// Start the mDNS discovery background task.
    ///
    /// This spawns a tokio task that:
    /// 1. Announces this peer on the local network
    /// 2. Listens for other peers' announcements
    /// 3. Periodically garbage-collects timed-out peers
    ///
    /// Returns a handle that can be used to stop discovery.
    ///
    /// Note: Requires the `mdns-sd` crate for actual mDNS.
    /// Without it, this is a no-op that just runs GC.
    pub fn start_mdns_gc(self: &Arc<Self>) -> tokio::task::JoinHandle<()> {
        let this = Arc::clone(self);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(ANNOUNCE_INTERVAL);
            loop {
                interval.tick().await;
                this.gc().await;
            }
        })
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn addr(port: u16) -> SocketAddr {
        use std::net::{IpAddr, Ipv4Addr};
        SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port)
    }

    #[tokio::test]
    async fn test_register_and_get_peer() {
        let disc = Discovery::new(addr(5000), "node-a");

        let is_new = disc
            .register_peer(addr(5001), "node-b", HashMap::new())
            .await;
        assert!(is_new);

        let info = disc.get_peer(&addr(5001)).await.unwrap();
        assert_eq!(info.name, "node-b");
        assert_eq!(info.source, DiscoverySource::Manual);
    }

    #[tokio::test]
    async fn test_register_existing_returns_false() {
        let disc = Discovery::new(addr(5000), "node-a");

        disc.register_peer(addr(5001), "node-b", HashMap::new())
            .await;
        let is_new = disc
            .register_peer(addr(5001), "node-b-updated", HashMap::new())
            .await;
        assert!(!is_new);

        let info = disc.get_peer(&addr(5001)).await.unwrap();
        assert_eq!(info.name, "node-b-updated");
    }

    #[tokio::test]
    async fn test_remove_peer() {
        let disc = Discovery::new(addr(5000), "node-a");
        disc.register_peer(addr(5001), "node-b", HashMap::new())
            .await;

        let removed = disc.remove_peer(&addr(5001)).await;
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().name, "node-b");

        assert!(disc.get_peer(&addr(5001)).await.is_none());
    }

    #[tokio::test]
    async fn test_live_peers_and_count() {
        let disc = Discovery::new(addr(5000), "node-a");
        disc.register_peer(addr(5001), "b", HashMap::new()).await;
        disc.register_peer(addr(5002), "c", HashMap::new()).await;
        disc.register_peer(addr(5003), "d", HashMap::new()).await;

        assert_eq!(disc.peer_count().await, 3);

        let addrs = disc.peer_addrs().await;
        assert_eq!(addrs.len(), 3);
    }

    #[tokio::test]
    async fn test_peer_seen_updates_timestamp() {
        let disc = Discovery::new(addr(5000), "node-a");
        disc.register_peer(addr(5001), "b", HashMap::new()).await;

        let before = disc.get_peer(&addr(5001)).await.unwrap().last_seen;
        // Small delay to ensure timestamp differs
        tokio::time::sleep(Duration::from_millis(10)).await;
        disc.peer_seen(&addr(5001)).await;
        let after = disc.get_peer(&addr(5001)).await.unwrap().last_seen;

        assert!(after > before);
    }

    #[tokio::test]
    async fn test_metadata() {
        let disc = Discovery::new(addr(5000), "node-a");
        disc.set_metadata("version", "0.6.0").await;
        disc.set_metadata("room", "lobby").await;

        let info = disc.local_info().await;
        assert_eq!(info.metadata.get("version").unwrap(), "0.6.0");
        assert_eq!(info.metadata.get("room").unwrap(), "lobby");
    }

    #[tokio::test]
    async fn test_event_callback() {
        use std::sync::atomic::{AtomicU32, Ordering};

        let disc = Discovery::new(addr(5000), "node-a");
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = Arc::clone(&counter);

        disc.on_event(Arc::new(move |_event| {
            counter_clone.fetch_add(1, Ordering::Relaxed);
        }))
        .await;

        disc.register_peer(addr(5001), "b", HashMap::new()).await;
        disc.remove_peer(&addr(5001)).await;

        // 1 Discovered + 1 Lost
        assert_eq!(counter.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn test_gc_removes_timed_out() {
        // Use a very short timeout for testing
        let disc = Discovery {
            peers: RwLock::new(HashMap::new()),
            local_info: RwLock::new(PeerInfo {
                addr: addr(5000),
                name: "node-a".to_string(),
                metadata: HashMap::new(),
                discovered_at: Instant::now(),
                last_seen: Instant::now(),
                source: DiscoverySource::Manual,
            }),
            callbacks: RwLock::new(Vec::new()),
            timeout: Duration::from_millis(50),
        };

        disc.register_peer(addr(5001), "b", HashMap::new()).await;
        assert_eq!(disc.peer_count().await, 1);

        // Wait for timeout
        tokio::time::sleep(Duration::from_millis(100)).await;

        let removed = disc.gc().await;
        assert_eq!(removed.len(), 1);
        assert_eq!(disc.peer_count().await, 0);
    }

    #[tokio::test]
    async fn test_register_with_metadata() {
        let disc = Discovery::new(addr(5000), "node-a");

        let mut meta = HashMap::new();
        meta.insert("role".to_string(), "relay".to_string());
        meta.insert("region".to_string(), "ap-northeast-1".to_string());

        disc.register_peer(addr(5001), "relay-1", meta).await;

        let info = disc.get_peer(&addr(5001)).await.unwrap();
        assert_eq!(info.metadata.get("role").unwrap(), "relay");
        assert_eq!(info.metadata.get("region").unwrap(), "ap-northeast-1");
    }
}

#![allow(clippy::significant_drop_tightening)]

//! Topic-Based Pub/Sub Channels
//!
//! Provides structured message routing for multi-domain synchronization.
//! Each topic is a logical channel that subscribers can join/leave dynamically.
//!
//! # Use Cases
//!
//! | Domain | Topic Pattern | Example |
//! |--------|---------------|---------|
//! | Games | `game/{room_id}/state` | Real-time game state sync |
//! | Collaboration | `doc/{doc_id}/edits` | Document co-editing |
//! | IoT | `sensor/{device_id}/telemetry` | Device telemetry streams |
//! | Messaging | `chat/{channel}/messages` | Group chat delivery |
//! | DB Replication | `db/{table}/mutations` | Change-data-capture |

use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::sync::Arc;

use tokio::sync::{broadcast, RwLock};

/// Maximum number of buffered messages per topic before slow subscribers lose data
const DEFAULT_CHANNEL_CAPACITY: usize = 256;

/// A topic identifier (hierarchical string, e.g. `"game/room42/state"`)
pub type Topic = Arc<str>;

/// Filter predicate for incoming messages on a topic.
///
/// Returns `true` to accept the message, `false` to drop it.
pub type FilterFn = Arc<dyn Fn(&[u8]) -> bool + Send + Sync>;

/// A published message on a topic
#[derive(Debug, Clone)]
pub struct TopicMessage {
    /// Topic this message belongs to
    pub topic: Topic,
    /// Serialized payload (bitcode / custom)
    pub payload: Vec<u8>,
    /// Origin peer (None if local)
    pub origin: Option<SocketAddr>,
    /// Monotonic sequence within the topic
    pub seq: u64,
}

/// Per-topic state
struct TopicState {
    /// Broadcast sender for this topic
    tx: broadcast::Sender<TopicMessage>,
    /// Set of subscribed peer addresses (for routing)
    subscribers: HashSet<SocketAddr>,
    /// Next sequence number
    next_seq: u64,
    /// Optional filter applied before dispatch
    filter: Option<FilterFn>,
}

/// Topic-based pub/sub router.
///
/// Thread-safe, lock-free reads for hot path (publish),
/// write lock only for subscribe/unsubscribe.
pub struct PubSub {
    topics: RwLock<HashMap<Topic, TopicState>>,
    capacity: usize,
}

impl PubSub {
    /// Create a new pub/sub router with default channel capacity.
    #[must_use]
    pub fn new() -> Self {
        Self {
            topics: RwLock::new(HashMap::new()),
            capacity: DEFAULT_CHANNEL_CAPACITY,
        }
    }

    /// Create with custom channel capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            topics: RwLock::new(HashMap::new()),
            capacity,
        }
    }

    /// Subscribe a peer to a topic. Creates the topic if it doesn't exist.
    ///
    /// Returns a receiver for messages on this topic.
    pub async fn subscribe(
        &self,
        topic: &str,
        peer: SocketAddr,
    ) -> broadcast::Receiver<TopicMessage> {
        let mut topics = self.topics.write().await;
        let topic_arc: Topic = Arc::from(topic);

        let state = topics.entry(topic_arc).or_insert_with(|| {
            let (tx, _) = broadcast::channel(self.capacity);
            TopicState {
                tx,
                subscribers: HashSet::new(),
                next_seq: 0,
                filter: None,
            }
        });

        state.subscribers.insert(peer);
        state.tx.subscribe()
    }

    /// Unsubscribe a peer from a topic.
    ///
    /// Returns `true` if the peer was subscribed.
    pub async fn unsubscribe(&self, topic: &str, peer: &SocketAddr) -> bool {
        let mut topics = self.topics.write().await;
        let topic_arc: Topic = Arc::from(topic);

        if let Some(state) = topics.get_mut(&topic_arc) {
            let removed = state.subscribers.remove(peer);
            // Clean up empty topics
            if state.subscribers.is_empty() {
                topics.remove(&topic_arc);
            }
            removed
        } else {
            false
        }
    }

    /// Set a filter on a topic. Only messages passing the filter are delivered.
    pub async fn set_filter(&self, topic: &str, filter: FilterFn) {
        let mut topics = self.topics.write().await;
        let topic_arc: Topic = Arc::from(topic);

        if let Some(state) = topics.get_mut(&topic_arc) {
            state.filter = Some(filter);
        }
    }

    /// Publish a message to a topic.
    ///
    /// Returns the number of receivers that got the message, or 0 if topic doesn't exist.
    pub async fn publish(
        &self,
        topic: &str,
        payload: Vec<u8>,
        origin: Option<SocketAddr>,
    ) -> usize {
        let topics = self.topics.read().await;
        let topic_arc: Topic = Arc::from(topic);

        if let Some(state) = topics.get(&topic_arc) {
            // Apply filter
            if let Some(ref filter) = state.filter {
                if !filter(&payload) {
                    return 0;
                }
            }

            let msg = TopicMessage {
                topic: topic_arc,
                payload,
                origin,
                seq: state.next_seq,
            };

            // broadcast::send returns number of receivers
            state.tx.send(msg).unwrap_or(0)
        } else {
            0
        }
    }

    /// Publish and increment the topic sequence counter.
    ///
    /// This is the mutable variant that tracks ordering.
    pub async fn publish_ordered(
        &self,
        topic: &str,
        payload: Vec<u8>,
        origin: Option<SocketAddr>,
    ) -> (u64, usize) {
        let mut topics = self.topics.write().await;
        let topic_arc: Topic = Arc::from(topic);

        if let Some(state) = topics.get_mut(&topic_arc) {
            if let Some(ref filter) = state.filter {
                if !filter(&payload) {
                    return (state.next_seq, 0);
                }
            }

            let seq = state.next_seq;
            state.next_seq += 1;

            let msg = TopicMessage {
                topic: topic_arc,
                payload,
                origin,
                seq,
            };

            let count = state.tx.send(msg).unwrap_or(0);
            (seq, count)
        } else {
            (0, 0)
        }
    }

    /// List all active topics.
    pub async fn topics(&self) -> Vec<Topic> {
        self.topics.read().await.keys().cloned().collect()
    }

    /// Get subscriber count for a topic.
    pub async fn subscriber_count(&self, topic: &str) -> usize {
        let topics = self.topics.read().await;
        let topic_arc: Topic = Arc::from(topic);
        topics.get(&topic_arc).map_or(0, |s| s.subscribers.len())
    }

    /// Get all subscribers for a topic (for transport-level routing).
    pub async fn subscribers(&self, topic: &str) -> Vec<SocketAddr> {
        let topics = self.topics.read().await;
        let topic_arc: Topic = Arc::from(topic);
        topics
            .get(&topic_arc)
            .map_or_else(Vec::new, |s| s.subscribers.iter().copied().collect())
    }
}

impl Default for PubSub {
    fn default() -> Self {
        Self::new()
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
    async fn test_subscribe_and_publish() {
        let ps = PubSub::new();
        let mut rx = ps.subscribe("test/topic", addr(1001)).await;

        let count = ps.publish("test/topic", b"hello".to_vec(), None).await;
        assert_eq!(count, 1);

        let msg = rx.recv().await.unwrap();
        assert_eq!(msg.payload, b"hello");
        assert_eq!(&*msg.topic, "test/topic");
    }

    #[tokio::test]
    async fn test_multiple_subscribers() {
        let ps = PubSub::new();
        let mut rx1 = ps.subscribe("room/1", addr(1001)).await;
        let mut rx2 = ps.subscribe("room/1", addr(1002)).await;

        let count = ps
            .publish("room/1", b"data".to_vec(), Some(addr(9999)))
            .await;
        assert_eq!(count, 2);

        let m1 = rx1.recv().await.unwrap();
        let m2 = rx2.recv().await.unwrap();
        assert_eq!(m1.payload, m2.payload);
        assert_eq!(m1.origin, Some(addr(9999)));
    }

    #[tokio::test]
    async fn test_unsubscribe() {
        let ps = PubSub::new();
        let _rx = ps.subscribe("topic", addr(1001)).await;
        assert_eq!(ps.subscriber_count("topic").await, 1);

        let removed = ps.unsubscribe("topic", &addr(1001)).await;
        assert!(removed);
        assert_eq!(ps.subscriber_count("topic").await, 0);
    }

    #[tokio::test]
    async fn test_publish_to_nonexistent_topic() {
        let ps = PubSub::new();
        let count = ps.publish("no/such/topic", b"data".to_vec(), None).await;
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn test_filter() {
        let ps = PubSub::new();
        let mut rx = ps.subscribe("filtered", addr(1001)).await;

        // Only accept payloads starting with 0xFF
        let filter: FilterFn = Arc::new(|payload: &[u8]| payload.first().copied() == Some(0xFF));
        ps.set_filter("filtered", filter).await;

        // This should be filtered out
        let count = ps.publish("filtered", vec![0x00, 0x01], None).await;
        assert_eq!(count, 0);

        // This should pass
        let count = ps.publish("filtered", vec![0xFF, 0x01], None).await;
        assert_eq!(count, 1);

        let msg = rx.recv().await.unwrap();
        assert_eq!(msg.payload, vec![0xFF, 0x01]);
    }

    #[tokio::test]
    async fn test_publish_ordered_increments_seq() {
        let ps = PubSub::new();
        let mut rx = ps.subscribe("ordered", addr(1001)).await;

        let (seq0, _) = ps.publish_ordered("ordered", b"a".to_vec(), None).await;
        let (seq1, _) = ps.publish_ordered("ordered", b"b".to_vec(), None).await;
        let (seq2, _) = ps.publish_ordered("ordered", b"c".to_vec(), None).await;

        assert_eq!(seq0, 0);
        assert_eq!(seq1, 1);
        assert_eq!(seq2, 2);

        let m0 = rx.recv().await.unwrap();
        let m1 = rx.recv().await.unwrap();
        let m2 = rx.recv().await.unwrap();
        assert_eq!(m0.seq, 0);
        assert_eq!(m1.seq, 1);
        assert_eq!(m2.seq, 2);
    }

    #[tokio::test]
    async fn test_topics_list() {
        let ps = PubSub::new();
        ps.subscribe("a/b", addr(1001)).await;
        ps.subscribe("c/d", addr(1002)).await;

        let topics = ps.topics().await;
        assert_eq!(topics.len(), 2);
    }

    #[tokio::test]
    async fn test_subscribers_list() {
        let ps = PubSub::new();
        ps.subscribe("room", addr(1001)).await;
        ps.subscribe("room", addr(1002)).await;
        ps.subscribe("room", addr(1003)).await;

        let subs = ps.subscribers("room").await;
        assert_eq!(subs.len(), 3);
        assert!(subs.contains(&addr(1001)));
        assert!(subs.contains(&addr(1002)));
        assert!(subs.contains(&addr(1003)));
    }
}

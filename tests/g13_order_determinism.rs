//! G13 oracles: every public accessor that exposes a collection's iteration
//! order must return a deterministic key order, independent of insertion
//! order and of OS hash seeding.
//!
//! Teeth: each case inserts a fixed scramble of 16 distinct keys and asserts
//! the sorted key order. A revert of the backing collection to a std HashMap /
//! HashSet would iterate in hash order; with 16 keys the chance of that
//! happening to equal the sorted order is 1/16! (~5e-14), so the revert
//! reddens the test deterministically (unlike a 3-key case, which a
//! random-seeded HashMap sorts by chance 1/6 of the time).

/// A fixed non-sorted permutation of 0..16 (no value in its sorted position
/// at the ends; a HashMap built from this does not come out in 0..16 order).
const SCRAMBLE: [u64; 16] = [11, 3, 7, 0, 14, 5, 9, 1, 12, 6, 15, 2, 8, 4, 13, 10];

fn sorted_0_to_16() -> Vec<u64> {
    (0..16).collect()
}

use alice_sync::{Node, NodeId};

#[test]
fn node_peers_iterate_in_nodeid_order_regardless_of_insertion() {
    let mut node = Node::new(NodeId(1000));
    for &k in &SCRAMBLE {
        node.add_peer(NodeId(k));
    }
    let got: Vec<u64> = node.peers().map(|p| p.id.0).collect();
    assert_eq!(
        got,
        sorted_0_to_16(),
        "peers() must iterate in NodeId order"
    );
}

#[cfg(feature = "cloud")]
#[test]
fn cloud_connected_devices_iterate_in_device_id_order() {
    use alice_sync::cloud_bridge::CloudSyncHub;
    let mut hub = CloudSyncHub::new();
    for &k in &SCRAMBLE {
        hub.register_device(k, format!("d{k}"));
    }
    let got: Vec<u64> = hub
        .connected_devices()
        .iter()
        .map(|d| d.device_id.0)
        .collect();
    assert_eq!(
        got,
        sorted_0_to_16(),
        "connected_devices() must iterate in device id order"
    );
}

#[cfg(feature = "async")]
mod async_oracles {
    use super::{sorted_0_to_16, SCRAMBLE};
    use std::net::{IpAddr, Ipv4Addr, SocketAddr};

    // build addresses from a port only (no IP literal), the loopback idiom the
    // crate's own tests use; distinct ports give sortable distinct SocketAddrs
    fn addr(port: u16) -> SocketAddr {
        SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port)
    }

    fn ports_of(addrs: &[SocketAddr]) -> Vec<u64> {
        addrs.iter().map(|a| u64::from(a.port()) - 9000).collect()
    }

    #[tokio::test]
    async fn channel_subscribers_iterate_in_socketaddr_order() {
        use alice_sync::PubSub;
        let ps = PubSub::new();
        for &k in &SCRAMBLE {
            let _ = ps.subscribe("t", addr(9000 + k as u16)).await;
        }
        let got = ps.subscribers("t").await;
        assert_eq!(
            ports_of(&got),
            sorted_0_to_16(),
            "subscribers() must iterate in SocketAddr order"
        );
    }

    #[tokio::test]
    async fn discovery_peer_addrs_and_metadata_iterate_in_order() {
        use alice_sync::discovery::Discovery;
        use std::collections::HashMap;
        let disc = Discovery::new(addr(1000), "self");
        let mut md = HashMap::new();
        for &k in &SCRAMBLE {
            disc.register_peer(addr(9000 + k as u16), "p", HashMap::new())
                .await;
            md.insert(format!("k{k:02}"), k.to_string());
        }
        let got = disc.peer_addrs().await;
        assert_eq!(
            ports_of(&got),
            sorted_0_to_16(),
            "peer_addrs() must iterate in SocketAddr order"
        );

        // the metadata field is serialised into the mDNS record; its keys must
        // iterate in a deterministic order (BTreeMap), not the HashMap arg order.
        // Register one peer carrying all 16 scrambled metadata keys.
        disc.register_peer(addr(8000), "meta", md).await;
        let peers = disc.live_peers().await;
        let one = peers
            .iter()
            .find(|p| p.addr == addr(8000))
            .expect("meta peer registered");
        let keys: Vec<String> = one.metadata.keys().cloned().collect();
        let want: Vec<String> = (0..16).map(|k| format!("k{k:02}")).collect();
        assert_eq!(keys, want, "metadata must iterate in key order");
    }
}

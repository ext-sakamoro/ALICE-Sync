//! Oracle for the UDP transport over a network that loses, repeats and reorders packets: a burst of
//! messages arrives once each (late retransmissions included), a peer that never answers is reported as a
//! delivery failure instead of being forgotten, and a session turns that failure into an event.
#![cfg(feature = "async")]

use std::collections::BTreeSet;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use alice_sync::protocol::Message;
use alice_sync::session::{SessionBuilder, SessionEvent};
use alice_sync::transport::{SyncTransport, TransportError, UdpTransport};
use tokio::net::UdpSocket;
use tokio::time::timeout;

const WAIT: Duration = Duration::from_secs(15);

/// A relay between the side that sends to it and `b`: packets each way pass, are dropped, sent twice or
/// held back until the next one, by `fate(to_b, n)`
async fn relay(b: SocketAddr, fate: fn(bool, usize) -> u8) -> SocketAddr {
    let socket = Arc::new(UdpSocket::bind("127.0.0.1:0").await.unwrap());
    let addr = socket.local_addr().unwrap();
    tokio::spawn(async move {
        let mut a: Option<SocketAddr> = None;
        let counts = [AtomicUsize::new(0), AtomicUsize::new(0)];
        let mut held: [Option<Vec<u8>>; 2] = [None, None];
        let mut buf = vec![0u8; 65_535];
        loop {
            let Ok((n, from)) = socket.recv_from(&mut buf).await else {
                return;
            };
            let to_b = from != b;
            if to_b {
                a = Some(from);
            }
            let Some(dest) = (if to_b { Some(b) } else { a }) else {
                continue;
            };
            let d = usize::from(!to_b);
            let k = counts[d].fetch_add(1, Ordering::SeqCst);
            let pkt = buf[..n].to_vec();
            match fate(to_b, k) {
                0 => {
                    let _ = socket.send_to(&pkt, dest).await;
                }
                1 => {}
                2 => {
                    let _ = socket.send_to(&pkt, dest).await;
                    let _ = socket.send_to(&pkt, dest).await;
                }
                _ => {
                    held[d] = Some(pkt);
                    continue;
                }
            }
            if let Some(h) = held[d].take() {
                let _ = socket.send_to(&h, dest).await;
            }
        }
    });
    addr
}

/// lose every third packet, repeat every fifth, hold back every seventh, in both directions
fn faulty(_: bool, k: usize) -> u8 {
    match k {
        k if k % 3 == 0 => 1,
        k if k % 5 == 0 => 2,
        k if k % 7 == 0 => 3,
        _ => 0,
    }
}

/// Keep calling `recv` on `t` (the transport's timers run there) and hand back what it returns
fn pump(t: Arc<UdpTransport>) -> tokio::sync::mpsc::UnboundedReceiver<Result<Message, String>> {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    tokio::spawn(async move {
        loop {
            let r = t.recv().await;
            let item = match r {
                Ok(e) => Ok(e.message),
                Err(TransportError::Closed) => return,
                Err(e) => Err(format!("{e}")),
            };
            if tx.send(item).is_err() {
                return;
            }
        }
    });
    rx
}

#[tokio::test]
async fn a_burst_arrives_once_each_through_loss_repeats_and_reordering() {
    let a = Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap());
    let b = Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap());
    let r = relay(b.local_addr().unwrap(), faulty).await;
    let mut from_a = pump(a.clone());
    let mut at_b = pump(b.clone());
    for seq in 0..40 {
        a.send_to(&Message::Ack { seq }, r).await.unwrap();
    }
    let mut got = BTreeSet::new();
    while got.len() < 40 {
        match timeout(WAIT, at_b.recv())
            .await
            .expect("all 40 arrive")
            .unwrap()
        {
            Ok(Message::Ack { seq }) => assert!(got.insert(seq), "{seq} came twice"),
            other => panic!("{other:?}"),
        }
    }
    // nothing more, and the sender reports no failure
    assert!(timeout(Duration::from_secs(4), at_b.recv()).await.is_err());
    assert!(from_a.try_recv().is_err(), "the sender reported a failure");
}

#[tokio::test]
async fn a_peer_that_never_answers_is_reported() {
    let a = Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap());
    // a socket that receives and never sends anything back
    let silent = UdpSocket::bind("127.0.0.1:0").await.unwrap();
    let dead = silent.local_addr().unwrap();
    let mut from_a = pump(a.clone());
    a.send_to(&Message::Ack { seq: 1 }, dead).await.unwrap();
    let r = timeout(WAIT, from_a.recv())
        .await
        .expect("a failure is reported")
        .unwrap();
    let err = r.expect_err("a failure, not a message");
    assert!(err.contains(&dead.to_string()), "{err}");
    drop(silent);
}

#[tokio::test]
async fn a_session_turns_an_undelivered_message_into_an_event() {
    let silent = UdpSocket::bind("127.0.0.1:0").await.unwrap();
    let dead = silent.local_addr().unwrap();
    let transport = UdpTransport::bind("127.0.0.1:0").await.unwrap();
    let session = SessionBuilder::new().name("a").build(transport);
    let mut events = session.subscribe_events();
    session.start().await;
    session.add_peer(dead).await;
    session.broadcast(&Message::Ack { seq: 9 }).await.unwrap();
    let failed = timeout(WAIT, async {
        loop {
            if let SessionEvent::DeliveryFailed { peer, .. } = events.recv().await.unwrap() {
                return peer;
            }
        }
    })
    .await
    .expect("the session reports the failure");
    assert_eq!(failed, dead);
    drop(silent);
}

#[tokio::test]
async fn two_peers_keep_their_own_reliability_state() {
    let a = Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap());
    let b = Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap());
    let c = Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap());
    let (ab, ac) = (b.local_addr().unwrap(), c.local_addr().unwrap());
    let _from_a = pump(a.clone());
    let mut at_b = pump(b.clone());
    let mut at_c = pump(c.clone());
    // interleaved sends: with one shared sequence space each receiver would see gaps it cannot fill
    for seq in 0..50 {
        a.send_to(&Message::Ack { seq }, if seq % 2 == 0 { ab } else { ac })
            .await
            .unwrap();
    }
    for (rx, want) in [(&mut at_b, 25usize), (&mut at_c, 25usize)] {
        let mut got = BTreeSet::new();
        while got.len() < want {
            if let Ok(Message::Ack { seq }) = timeout(WAIT, rx.recv()).await.unwrap().unwrap() {
                got.insert(seq);
            }
        }
    }
}

/// hold back the third packet towards `b` until the one after it
fn third_late(to_b: bool, k: usize) -> u8 {
    if to_b && k == 2 {
        3
    } else {
        0
    }
}

fn big(tag: i16) -> Message {
    use alice_sync::{Event, EventKind};
    Message::Events {
        events: (0..230u32)
            .map(|i| {
                Event::new(EventKind::Motion {
                    entity: 0x0100_0000 + i * 7919,
                    delta: [tag, (i * 97) as i16, -(i as i16) * 31],
                })
            })
            .collect(),
        checkpoints: Vec::new(),
    }
}

/// Two fragmented messages whose fragments interleave each arrive whole: each message has its own id
#[tokio::test]
async fn interleaved_fragments_of_two_messages_stay_apart() {
    let a = Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap());
    let b = Arc::new(UdpTransport::bind("127.0.0.1:0").await.unwrap());
    let r = relay(b.local_addr().unwrap(), third_late).await;
    let _from_a = pump(a.clone());
    let mut at_b = pump(b.clone());
    // three fragments each: the held-back third is the last of the first message
    assert_eq!(
        alice_sync::reliability::fragment_payload(0, &big(1).to_compact_bytes()).len(),
        3
    );
    a.send_to(&big(1), r).await.unwrap();
    a.send_to(&big(2), r).await.unwrap();
    let mut tags = BTreeSet::new();
    while tags.len() < 2 {
        match timeout(WAIT, at_b.recv())
            .await
            .expect("both arrive")
            .unwrap()
        {
            Ok(m @ Message::Events { .. }) => {
                let bytes = m.to_compact_bytes();
                let tag = [1, 2]
                    .into_iter()
                    .find(|t| big(*t).to_compact_bytes() == bytes)
                    .expect("one of the two messages, whole");
                assert!(tags.insert(tag), "{tag} came twice");
            }
            other => panic!("{other:?}"),
        }
    }
}

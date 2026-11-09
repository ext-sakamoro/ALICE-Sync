//! End to end: two peers play one match through the relay.
//!
//! Every piece runs as a match would use it: each peer's id is the device id of a certificate it was
//! issued (the same 16 bytes as its event origin), the peers agree on the input delay with
//! `match_sync`, every frame crosses the wire as `relay_wire` bytes in both directions, the room
//! relay (`RelayCore`) orders and forwards the payloads, each peer runs a `RollbackSession` and rolls
//! back when a late remote input differs from its prediction, and each peer sends the digests of its
//! confirmed frames to the other, compared by `DigestLog`. The network delays packets by different
//! amounts (so the relay sees them out of order) and loses one, which the relay asks for again.
//!
//! The oracle is independent of the crate: a counter game simulated once with every input known in
//! advance, with the agreed delay applied by hand. Both peers must end on that state for every
//! confirmed frame, and no digest comparison may report a mismatch.

use std::collections::BTreeMap;

use alice_identity::device_cert::DeviceCert;
use alice_identity::did::Did;
use alice_sign::KeyPair;
use alice_sync::event::Origin;
use alice_sync::input_sync::{InputFrame, RollbackAction, RollbackSession};
use alice_sync::match_sync::{agree, propose_delay, DigestLog};
use alice_sync::relay::{Limits, Out, PeerId, RelayCore, RoomId};
use alice_sync::relay_wire::Frame;

const FRAMES: u64 = 180;
const FRAME_US: u64 = 16_667;
const ROOM: RoomId = RoomId::from_raw(7);
const TAG_INPUT: u8 = 1;
const TAG_DIGEST: u8 = 2;

/// Reference game: the state folds every applied input, weighted by player.
fn step(state: (u64, u64), inputs: &[InputFrame]) -> (u64, u64) {
    let mut s = state.0.wrapping_mul(31);
    for i in inputs {
        s = s.wrapping_add(u64::from(i.actions) * (u64::from(i.player_id) + 1));
    }
    (s, state.1 + 1)
}

fn enc(s: (u64, u64)) -> Vec<u8> {
    let mut v = s.0.to_le_bytes().to_vec();
    v.extend_from_slice(&s.1.to_le_bytes());
    v
}

fn dec(b: &[u8]) -> (u64, u64) {
    (
        u64::from_le_bytes(b[..8].try_into().unwrap()),
        u64::from_le_bytes(b[8..16].try_into().unwrap()),
    )
}

fn digest(s: (u64, u64)) -> u64 {
    s.0 ^ s.1.rotate_left(17)
}

/// The buttons each player presses on frame `f` (changes often, so predictions fail)
fn pressed(player: u8, f: u64) -> u32 {
    let k = f / (3 + u64::from(player) * 2);
    u32::try_from((k * 7 + u64::from(player) * 3) % 5).unwrap()
}

fn device(seed: u8, name: &str) -> DeviceCert {
    let master = KeyPair::from_seed([seed; 32]);
    let device = KeyPair::from_seed([seed + 100; 32]);
    DeviceCert::issue(
        &master,
        &Did::parse(&format!("did:alice:{name}")).unwrap(),
        device.public().0,
        [0x4e; 32],
        1_800_000_000,
        1_800_086_400,
    )
    .unwrap()
}

struct Peer {
    id: PeerId,
    player: u8,
    session: RollbackSession,
    state: (u64, u64),
    /// state after each simulated frame (index = frame)
    history: Vec<(u64, u64)>,
    next_seq: u64,
    /// everything published, by seq, for the relay's resend requests
    sent: BTreeMap<u64, Vec<u8>>,
    digests: DigestLog,
    /// the highest frame whose digest was sent
    digest_sent: Option<u64>,
    rollbacks: u32,
    reports: u32,
}

impl Peer {
    fn new(id: PeerId, player: u8, delay: u64) -> Self {
        Self {
            id,
            player,
            session: RollbackSession::with_input_delay(2, player, 32, delay),
            state: (0, 0),
            history: Vec::new(),
            next_seq: 0,
            sent: BTreeMap::new(),
            digests: DigestLog::new(),
            digest_sent: None,
            rollbacks: 0,
            reports: 0,
        }
    }

    fn publish(&mut self, payload: Vec<u8>) -> Vec<u8> {
        let seq = self.next_seq;
        self.next_seq += 1;
        let bytes = Frame::Publish {
            room: ROOM,
            seq,
            payload,
        }
        .encode();
        self.sent.insert(seq, bytes.clone());
        bytes
    }

    /// Handle a frame from the relay; returns frames to send back (resends)
    fn receive(&mut self, bytes: &[u8], earliest: &mut Option<u64>) -> Vec<Vec<u8>> {
        match Frame::decode(bytes).expect("a whole frame from the relay") {
            Frame::Forward { payload, .. } => {
                match payload.split_first() {
                    Some((&TAG_INPUT, rest)) => {
                        let input = InputFrame::from_bytes(rest).expect("an input");
                        if let RollbackAction::Rollback { to_frame } =
                            self.session.add_remote_input(input)
                        {
                            *earliest = Some(earliest.map_or(to_frame, |e| e.min(to_frame)));
                        }
                    }
                    Some((&TAG_DIGEST, rest)) => {
                        let frame = u64::from_be_bytes(rest[..8].try_into().unwrap());
                        let d = u64::from_be_bytes(rest[8..16].try_into().unwrap());
                        if self.digests.remote(frame, d).is_some() {
                            self.reports += 1;
                        }
                    }
                    _ => panic!("unknown payload"),
                }
                Vec::new()
            }
            Frame::Resend { from, to_seq, .. } => {
                (from..=to_seq).map(|s| self.sent[&s].clone()).collect()
            }
            other => panic!("unexpected frame {other:?}"),
        }
    }

    /// Roll back to `to` (if any) and replay up to `f - 1`, then simulate `f`; returns frames to send
    fn tick(&mut self, f: u64, earliest: Option<u64>) -> Vec<Vec<u8>> {
        if let Some(to) = earliest {
            self.rollbacks += 1;
            self.state = dec(self.session.get_snapshot(to).expect("snapshot"));
            self.history.truncate(usize::try_from(to).unwrap());
            for g in to..f {
                let ins = self.session.inputs_for_frame(g);
                self.session
                    .save_snapshot(g, enc(self.state), digest(self.state));
                self.state = step(self.state, &ins);
                self.history.push(self.state);
            }
        }
        let mut out = Vec::new();
        let given = {
            let mut i = InputFrame::new(f, self.player);
            i.actions = pressed(self.player, f);
            i
        };
        let (ins, stamped) = self.session.add_local_input_delayed(given);
        let mut payload = vec![TAG_INPUT];
        payload.extend_from_slice(&stamped.to_bytes());
        out.push(self.publish(payload));
        self.session
            .save_snapshot(f, enc(self.state), digest(self.state));
        self.state = step(self.state, &ins);
        self.history.push(self.state);
        // the digests of frames both inputs are confirmed for, each sent once
        let confirmed = self.session.confirmed_frame();
        let start = self.digest_sent.map_or(0, |d| d + 1);
        for g in start..=confirmed.min(f) {
            // only a confirmed frame's digest is final: a predicted one may still be rolled back
            let d = digest(self.history[usize::try_from(g).unwrap()]);
            if self.digests.local(g, d).is_some() {
                self.reports += 1;
            }
            let mut payload = vec![TAG_DIGEST];
            payload.extend_from_slice(&g.to_be_bytes());
            payload.extend_from_slice(&d.to_be_bytes());
            out.push(self.publish(payload));
            self.digest_sent = Some(g);
        }
        out
    }
}

/// The reference: every input known, the local input given on frame `f` applied on `f + delay`
fn reference(delay: u64) -> Vec<(u64, u64)> {
    let applied = |p: u8, f: u64| if f < delay { 0 } else { pressed(p, f - delay) };
    let mut state = (0, 0);
    (0..FRAMES)
        .map(|f| {
            let mut a = InputFrame::new(f, 0);
            a.actions = applied(0, f);
            let mut b = InputFrame::new(f, 1);
            b.actions = applied(1, f);
            state = step(state, &[a, b]);
            state
        })
        .collect()
}

/// One packet in flight: when it arrives, and where (None = the relay, Some(i) = peer i)
struct Packet {
    at: u64,
    order: u64,
    to: Option<usize>,
    from: usize,
    bytes: Vec<u8>,
}

#[test]
fn two_peers_play_one_match_through_the_relay() {
    let certs = [device(1, "alice"), device(2, "bob")];
    let ids = [certs[0].device_id(), certs[1].device_id()];
    // one identifier: the relay peer is the certificate's device id and the event origin
    for c in &certs {
        assert_eq!(Origin::of(c).0, *c.device_id().as_bytes());
    }
    assert_ne!(ids[0], ids[1]);

    // each peer proposes from the round trip it measured; both take the larger
    let rtt_us = [3 * 2 * FRAME_US, 4 * 2 * FRAME_US];
    let proposals = [
        propose_delay(rtt_us[0], FRAME_US, 1),
        propose_delay(rtt_us[1], FRAME_US, 1),
    ];
    let delay = agree(proposals[0], proposals[1]);
    assert_eq!((proposals, delay), ([4, 5], 5));

    let mut relay = RelayCore::new();
    relay
        .open_room(
            ROOM,
            &ids,
            Limits {
                max_payload: 256,
                max_log: 4096,
                max_pending: 64,
            },
        )
        .unwrap();
    let mut peers = [Peer::new(ids[0], 0, delay), Peer::new(ids[1], 1, delay)];

    // latency in frames: peer 0's uplink alternates 1 and 4 (the relay sees its seqs out of order),
    // peer 1's is 2 except every 7th publish at 5 (late past the agreed delay, so peer 0 rolls back
    // too); downlinks are 1 and 3; peer 0's 30th publish is lost once
    let mut sent_by = [0u64; 2];
    let mut lost = false;
    let mut order = 0u64;
    let mut wire: Vec<Packet> = Vec::new();
    let mut relayed = 0u64;
    let mut resent = 0u64;
    for f in 0..FRAMES + 40 {
        // deliver everything due by this frame, in arrival order
        wire.sort_by_key(|p| (p.at, p.order));
        let due: Vec<Packet> = {
            let split = wire.iter().position(|p| p.at > f).unwrap_or(wire.len());
            wire.drain(..split).collect()
        };
        let mut earliest = [None, None];
        for p in due {
            match p.to {
                None => {
                    let frame = Frame::decode(&p.bytes).expect("a whole frame at the relay");
                    let Frame::Publish { room, seq, payload } = frame else {
                        panic!("peers only publish");
                    };
                    for out in relay.publish(peers[p.from].id, room, seq, payload) {
                        let (to, bytes) = match out {
                            Out::Forward {
                                to,
                                room,
                                from,
                                seq,
                                payload,
                            } => {
                                relayed += 1;
                                (
                                    to,
                                    Frame::Forward {
                                        room,
                                        from,
                                        seq,
                                        payload,
                                    }
                                    .encode(),
                                )
                            }
                            Out::Resend {
                                to,
                                room,
                                from,
                                to_seq,
                            } => (to, Frame::Resend { room, from, to_seq }.encode()),
                            Out::Refused { reason, .. } => panic!("refused: {reason:?}"),
                        };
                        let i = ids.iter().position(|d| *d == to).unwrap();
                        order += 1;
                        wire.push(Packet {
                            at: f + [1, 3][i],
                            order,
                            to: Some(i),
                            from: 2,
                            bytes,
                        });
                    }
                }
                Some(i) => {
                    for back in peers[i].receive(&p.bytes, &mut earliest[i]) {
                        resent += 1;
                        order += 1;
                        wire.push(Packet {
                            at: f + 1,
                            order,
                            to: None,
                            from: i,
                            bytes: back,
                        });
                    }
                }
            }
        }
        if f >= FRAMES {
            // the match is over: only deliver what is still in flight
            continue;
        }
        for i in 0..2 {
            for bytes in peers[i].tick(f, earliest[i]) {
                sent_by[i] += 1;
                if i == 0 && sent_by[0] == 30 && !lost {
                    lost = true;
                    continue;
                }
                let up = if i == 0 {
                    [1, 4][usize::try_from(sent_by[0] % 2).unwrap()]
                } else if sent_by[1] % 7 == 0 {
                    5
                } else {
                    2
                };
                order += 1;
                wire.push(Packet {
                    at: f + up,
                    order,
                    to: None,
                    from: i,
                    bytes,
                });
            }
        }
    }
    assert!(wire.is_empty(), "packets still in flight");

    // the network did what the scene needs: a loss resent on request, rollbacks on both sides
    assert!(lost && resent >= 1, "resent {resent}");
    assert!(relayed > 2 * FRAMES, "relayed {relayed}");
    assert!(
        peers[0].rollbacks > 0 && peers[1].rollbacks > 0,
        "{} {}",
        peers[0].rollbacks,
        peers[1].rollbacks
    );

    // both peers confirmed the whole match and match the reference on every frame
    let want = reference(delay);
    for p in &peers {
        assert!(
            p.session.confirmed_frame() + 1 >= FRAMES - delay,
            "{}",
            p.session.confirmed_frame()
        );
        let n = usize::try_from(FRAMES - delay).unwrap();
        assert_eq!(&p.history[..n], &want[..n], "peer {}", p.player);
        assert_eq!(p.reports, 0, "peer {} reported a desync", p.player);
    }
    // the digests both sides exchanged cover the confirmed frames
    assert!(peers[0].digest_sent.unwrap() + 1 >= FRAMES - delay);
    assert!(peers[1].digest_sent.unwrap() + 1 >= FRAMES - delay);
}

//! `converge::Key::to_bytes` / `from_bytes`: a store ordered by bytes keeps keys in their order.
//!
//! oracle: the derived `Ord` of `Key` (timestamp, then origin, then sequence) is the reference. For every pair of
//! keys drawn below, the lexicographic order of their bytes equals that `Ord`; every key comes back from its bytes;
//! bytes of any other length are refused. The keys are drawn from small ranges so that equal timestamps and equal
//! origins are common (the later fields then decide), with the extremes 0 and u64::MAX mixed in.

use alice_sync::converge::{Key, KEY_BYTES};
use alice_sync::event::Origin;

/// SplitMix64: a fixed sequence, so the test is the same on every run.
struct Mix(u64);

impl Mix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn small_u64(&mut self) -> u64 {
        match self.next() % 6 {
            0 => 0,
            1 => u64::MAX,
            2 => u64::MAX - 1,
            _ => self.next() % 4,
        }
    }
    fn key(&mut self) -> Key {
        let mut origin = [0u8; 16];
        // origins differing in the first byte, in the last byte, or not at all
        origin[0] = (self.next() % 3) as u8;
        origin[15] = (self.next() % 3) as u8 * 0x7f;
        Key {
            timestamp: self.small_u64(),
            origin: Origin(origin),
            seq: self.small_u64(),
        }
    }
}

fn keys() -> Vec<Key> {
    let mut m = Mix(0x5eed);
    (0..400).map(|_| m.key()).collect()
}

#[test]
fn the_bytes_of_two_keys_compare_as_the_keys_do() {
    let ks = keys();
    let mut compared = 0usize;
    let mut equal_ts = 0usize;
    for a in &ks {
        for b in &ks {
            assert_eq!(a.to_bytes().cmp(&b.to_bytes()), a.cmp(b), "{a:?} vs {b:?}");
            compared += 1;
            if a.timestamp == b.timestamp && a.origin != b.origin {
                equal_ts += 1;
            }
        }
    }
    assert_eq!(compared, 400 * 400);
    // the later fields were exercised: many pairs share a timestamp and differ in origin
    assert!(
        equal_ts > 10_000,
        "{equal_ts} pairs with one timestamp and two origins"
    );
}

#[test]
fn every_key_comes_back_from_its_bytes() {
    for k in keys() {
        let b = k.to_bytes();
        assert_eq!(b.len(), KEY_BYTES);
        assert_eq!(Key::from_bytes(&b), Some(k));
    }
    // the layout: timestamp, origin, sequence, big-endian
    let k = Key {
        timestamp: 0x0102_0304_0506_0708,
        origin: Origin([0xaa; 16]),
        seq: 0x1112_1314_1516_1718,
    };
    let mut want = vec![1, 2, 3, 4, 5, 6, 7, 8];
    want.extend([0xaa; 16]);
    want.extend([0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18]);
    assert_eq!(k.to_bytes().to_vec(), want);
}

#[test]
fn bytes_of_another_length_are_refused() {
    let b = keys()[0].to_bytes();
    for n in [0, 1, KEY_BYTES - 1] {
        assert_eq!(Key::from_bytes(&b[..n]), None, "{n} bytes");
    }
    let mut long = b.to_vec();
    long.push(0);
    assert_eq!(Key::from_bytes(&long), None, "{} bytes", long.len());
}

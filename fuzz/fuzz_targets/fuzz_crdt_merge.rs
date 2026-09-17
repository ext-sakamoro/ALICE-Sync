//! Fuzz: arbitrary operation sequences on the CRDTs — merges must never panic
//! and must be commutative (a ⊔ b == b ⊔ a) and idempotent (a ⊔ a == a).
#![no_main]

use alice_sync::crdt::{CrdtMergeable, GCounter, LwwMap, LwwRegister, OrSet, PnCounter};
use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
enum Op {
    Inc { replica: u8, by: u8 },
    Dec { replica: u8 },
    Set { value: u16, ts: u32 },
    Add { replica: u8, value: u8 },
    Remove { value: u8 },
    MapInsert { key: u8, value: u16 },
    MapRemove { key: u8 },
}

#[derive(Arbitrary, Debug)]
struct Input {
    a: Vec<Op>,
    b: Vec<Op>,
}

struct Replica {
    g: GCounter,
    pn: PnCounter,
    reg: LwwRegister<u16>,
    set: OrSet<u8>,
    map: LwwMap<u8, u16>,
}

impl Replica {
    fn new(id: u64) -> Self {
        Self {
            g: GCounter::new(),
            pn: PnCounter::new(),
            reg: LwwRegister::new(0, id),
            set: OrSet::new(),
            map: LwwMap::new(id),
        }
    }
    fn apply(&mut self, id: u64, ops: &[Op]) {
        for op in ops.iter().take(64) {
            match *op {
                Op::Inc { replica, by } => self.g.increment_by(u64::from(replica), u64::from(by)),
                Op::Dec { replica } => self.pn.decrement(u64::from(replica)),
                Op::Set { value, ts } => self.reg.set_at(value, u64::from(ts)),
                Op::Add { replica, value } => self.set.add(u64::from(replica) * 4 + id, value),
                Op::Remove { value } => self.set.remove(&value),
                Op::MapInsert { key, value } => self.map.insert(key, value),
                Op::MapRemove { key } => self.map.remove(&key),
            }
        }
    }
    fn merge(&mut self, o: &Self) {
        self.g.merge(&o.g);
        self.pn.merge(&o.pn);
        self.reg.merge(&o.reg);
        self.set.merge(&o.set);
        self.map.merge(&o.map);
    }
    fn observe(&self) -> (u64, i64, u16, Vec<u8>, Vec<(u8, u16)>) {
        (
            self.g.value(),
            self.pn.value(),
            self.reg.value,
            self.set.values().into_iter().copied().collect(),
            self.map.entries().into_iter().map(|(k, v)| (*k, *v)).collect(),
        )
    }
}

fuzz_target!(|input: Input| {
    let mut a = Replica::new(1);
    let mut b = Replica::new(2);
    a.apply(1, &input.a);
    b.apply(2, &input.b);
    let mut ab = Replica::new(1);
    ab.apply(1, &input.a);
    ab.merge(&b);
    let mut ba = Replica::new(2);
    ba.apply(2, &input.b);
    ba.merge(&a);
    assert_eq!(ab.observe(), ba.observe(), "merge is commutative");
    let before = ab.observe();
    let snapshot = Replica {
        g: ab.g.clone(),
        pn: ab.pn.clone(),
        reg: ab.reg.clone(),
        set: ab.set.clone(),
        map: ab.map.clone(),
    };
    ab.merge(&snapshot);
    assert_eq!(ab.observe(), before, "merge is idempotent");
});

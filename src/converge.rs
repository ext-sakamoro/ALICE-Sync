/*
    ALICE-Sync
    Copyright (C) 2026 Moroya Sakamoto

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

//! Order-independent application of events from any number of origins (ALICE-Law G13).
//!
//! [`World::apply`] applies events in the order it is given them, which is right for one writer
//! replaying its own log. Events from several peers arrive in an order the network and the async
//! scheduler choose, so [`crate::Node`] applies them through [`Converge`] instead: the world it
//! produces is a function of the *set* of events received, the same for every arrival order and
//! unchanged by a duplicate delivery.
//!
//! An event is identified by its origin and sequence number (an origin never reuses one; a second
//! event with the same pair is a duplicate delivery and is not applied again) and ordered by its
//! [`Key`] (Lamport timestamp, origin, sequence). Per entity:
//!
//! - the incarnation is the Spawn with the greatest key; a Despawn with a greater key ends it
//!   (a tombstone, kept so a Spawn delivered late cannot revive the entity);
//! - its position is that Spawn's position plus every Motion whose key is greater than the Spawn's
//!   (a Motion that arrives before its Spawn is kept and applied when the Spawn arrives; one with a
//!   smaller key belonged to an earlier incarnation and is not applied);
//! - each property is the value of the Property write with the greatest key after the Spawn.
//!
//! The frame is the value of the Tick with the greatest key. Input and Custom do not change the
//! world.
//!
//! # Memory
//!
//! Per origin: the sequence numbers below which every event was received, plus those received
//! above a gap. Per entity: its greatest Spawn and Despawn, its latest Property writes, and the
//! Motions keyed after its greatest Spawn (a Spawn with a greater key drops the Motions keyed before
//! it, which can never count again). The Motions of one incarnation are not bounded: dropping them
//! needs a point every peer has passed (causal stability), which this module does not track.

use crate::{Event, EventKind, Result, World};
use std::collections::{BTreeMap, BTreeSet};

/// The order of events: Lamport timestamp, origin, sequence number
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Key {
    /// Lamport timestamp of the event
    pub timestamp: u64,
    /// Node that originated the event
    pub origin: u64,
    /// The origin's sequence number
    pub seq: u64,
}

impl Key {
    /// The key of an event
    #[must_use]
    pub const fn of(event: &Event) -> Self {
        Self {
            timestamp: event.timestamp,
            origin: event.origin,
            seq: event.seq.0,
        }
    }
}

/// The sequence numbers received from one origin
#[derive(Debug, Default)]
struct Received {
    /// every sequence number below this one was received
    below: u64,
    /// received sequence numbers above a gap (all greater than `below`)
    above: BTreeSet<u64>,
}

impl Received {
    /// Record `seq`; false when it was already received
    fn insert(&mut self, seq: u64) -> bool {
        if seq < self.below || !self.above.insert(seq) {
            return false;
        }
        while self.above.remove(&self.below) {
            self.below += 1;
        }
        true
    }
}

#[derive(Debug, Default)]
struct EntityRecord {
    /// (key, kind, position) of the Spawn with the greatest key
    spawn: Option<(Key, u16, [i16; 3])>,
    /// key of the Despawn with the greatest key
    despawn: Option<Key>,
    /// the Motions keyed after the greatest Spawn (or all of them, before any Spawn)
    motions: Vec<(Key, [i16; 3])>,
    /// per property, (key, value) of the write with the greatest key
    props: BTreeMap<u16, (Key, i32)>,
}

impl EntityRecord {
    /// The live incarnation's Spawn, if the entity is live
    fn live_spawn(&self) -> Option<(Key, u16, [i16; 3])> {
        let spawn = self.spawn?;
        match self.despawn {
            Some(d) if d > spawn.0 => None,
            _ => Some(spawn),
        }
    }
}

/// The records behind a node's world; see the module documentation
#[derive(Debug, Default)]
pub struct Converge {
    received: BTreeMap<u64, Received>,
    entities: BTreeMap<u32, EntityRecord>,
    /// (key, frame) of the Tick with the greatest key
    frame: Option<(Key, u64)>,
}

impl Converge {
    /// An empty record set
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record `event` and bring `world` to the state of the recorded set; an event whose origin
    /// and sequence number were already received changes nothing.
    ///
    /// # Errors
    ///
    /// Returns the error of [`World::apply`] (currently infallible).
    #[allow(
        clippy::disallowed_methods,
        reason = "the order-independent sink of ALICE-Law G13: drives the world from the keyed records"
    )]
    pub fn apply(&mut self, world: &mut World, event: &Event) -> Result<()> {
        if !self
            .received
            .entry(event.origin)
            .or_default()
            .insert(event.seq.0)
        {
            return Ok(());
        }
        let key = Key::of(event);
        match event.kind {
            EventKind::Spawn { entity, kind, pos } => {
                let rec = self.entities.entry(entity).or_default();
                if rec.spawn.is_some_and(|s| s.0 >= key) {
                    return Ok(());
                }
                rec.spawn = Some((key, kind, pos));
                rec.motions.retain(|&(k, _)| k > key);
                Self::rebuild(world, entity, rec)
            }
            EventKind::Despawn { entity } => {
                let rec = self.entities.entry(entity).or_default();
                if rec.despawn.is_some_and(|d| d >= key) {
                    return Ok(());
                }
                rec.despawn = Some(key);
                Self::rebuild(world, entity, rec)
            }
            EventKind::Motion { entity, delta } => {
                let rec = self.entities.entry(entity).or_default();
                if rec.spawn.is_some_and(|s| key < s.0) {
                    return Ok(()); // an earlier incarnation's
                }
                rec.motions.push((key, delta));
                if rec.live_spawn().is_some() {
                    // motion is a wrapping sum, so adding it now equals replaying in key order
                    world.apply(event)?;
                }
                Ok(())
            }
            EventKind::Property {
                entity,
                prop,
                value,
            } => {
                let rec = self.entities.entry(entity).or_default();
                if rec.props.get(&prop).is_some_and(|&(k, _)| k >= key) {
                    return Ok(());
                }
                rec.props.insert(prop, (key, value));
                match rec.live_spawn() {
                    Some(s) if key > s.0 => world.apply(event),
                    _ => Ok(()),
                }
            }
            EventKind::Tick { frame } => {
                if self.frame.is_some_and(|f| f.0 >= key) {
                    return Ok(());
                }
                self.frame = Some((key, frame));
                world.apply(event)
            }
            EventKind::Input { .. } | EventKind::Custom { .. } => Ok(()),
        }
    }

    /// Replace `entity` in `world` with what its record says: removed, or its live Spawn followed by
    /// its Motions and the latest Property writes keyed after that Spawn
    #[allow(
        clippy::disallowed_methods,
        reason = "the order-independent sink of ALICE-Law G13: drives the world from the keyed records"
    )]
    fn rebuild(world: &mut World, entity: u32, rec: &EntityRecord) -> Result<()> {
        world.apply(&Event::new(EventKind::Despawn { entity }))?;
        let Some((spawn_key, kind, pos)) = rec.live_spawn() else {
            return Ok(());
        };
        world.apply(&Event::new(EventKind::Spawn { entity, kind, pos }))?;
        for &(_, delta) in &rec.motions {
            world.apply(&Event::new(EventKind::Motion { entity, delta }))?;
        }
        for (&prop, &(key, value)) in &rec.props {
            if key > spawn_key {
                world.apply(&Event::new(EventKind::Property {
                    entity,
                    prop,
                    value,
                }))?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::SeqNum;

    fn stamped(kind: EventKind, timestamp: u64, origin: u64, seq: u64) -> Event {
        let mut e = Event::new(kind);
        e.timestamp = timestamp;
        e.origin = origin;
        e.seq = SeqNum(seq);
        e
    }

    #[test]
    fn the_watermark_holds_a_gap_open_until_it_is_filled() {
        let mut r = Received::default();
        for seq in [0, 2, 3, 5] {
            assert!(r.insert(seq));
        }
        assert_eq!(
            (r.below, r.above.iter().copied().collect::<Vec<_>>()),
            (1, vec![2, 3, 5])
        );
        assert!(!r.insert(2), "a duplicate above the gap");
        assert!(!r.insert(0), "a duplicate below the watermark");
        assert!(r.insert(1), "the late event that fills the gap is new");
        assert_eq!(
            (r.below, r.above.iter().copied().collect::<Vec<_>>()),
            (4, vec![5])
        );
        assert!(r.insert(4));
        assert_eq!((r.below, r.above.len()), (6, 0));
    }

    #[test]
    fn a_greater_spawn_drops_the_motions_keyed_before_it() {
        let mut c = Converge::new();
        let mut w = World::new(0);
        let motion = EventKind::Motion {
            entity: 1,
            delta: [1, 0, 0],
        };
        let spawn = EventKind::Spawn {
            entity: 1,
            kind: 0,
            pos: [0, 0, 0],
        };
        c.apply(&mut w, &stamped(spawn.clone(), 0, 1, 0)).unwrap();
        for seq in 1..=4 {
            c.apply(&mut w, &stamped(motion.clone(), seq, 1, seq))
                .unwrap();
        }
        assert_eq!(c.entities[&1].motions.len(), 4);
        // keyed (3, 2, 0): after the motions at times 1 and 2 and (3, 1, 3), before (4, 1, 4)
        c.apply(&mut w, &stamped(spawn, 3, 2, 0)).unwrap();
        assert_eq!(c.entities[&1].motions.len(), 1);
        c.apply(&mut w, &stamped(motion, 2, 3, 0)).unwrap(); // keyed before the live Spawn
        assert_eq!(c.entities[&1].motions.len(), 1, "never stored");
        let unit = crate::fixed_point::Fixed::from_i16(1).0;
        assert_eq!(w.get_entity(1).unwrap().position.x.0, unit);
    }
}

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

//! Generational Arena - cache-friendly entity storage
//!
//! HashMap has poor cache locality. Arena allocates entities
//! in contiguous memory for maximum CPU cache efficiency.

use serde::{Deserialize, Serialize};

/// Generation counter to detect stale handles
pub type Generation = u32;

/// Handle to an entity in the arena
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Handle {
    pub index: u32,
    pub generation: Generation,
}

impl Handle {
    pub const INVALID: Self = Self {
        index: u32::MAX,
        generation: 0,
    };

    #[inline]
    pub const fn new(index: u32, generation: Generation) -> Self {
        Self { index, generation }
    }

    #[inline]
    pub const fn is_valid(&self) -> bool {
        self.index != u32::MAX
    }
}

/// Slot in the arena
#[derive(Debug, Clone, Serialize, Deserialize)]
enum Slot<T> {
    /// Occupied slot with data and generation
    Occupied { data: T, generation: Generation },
    /// Free slot pointing to next free index
    Free {
        next_free: u32,
        generation: Generation,
    },
}

/// Generational arena for cache-friendly storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Arena<T> {
    slots: Vec<Slot<T>>,
    free_head: u32,
    len: usize,
}

impl<T> Default for Arena<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Arena<T> {
    /// Create empty arena
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            free_head: u32::MAX,
            len: 0,
        }
    }

    /// Create with capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            slots: Vec::with_capacity(capacity),
            free_head: u32::MAX,
            len: 0,
        }
    }

    /// Number of active elements
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Insert element, return handle
    pub fn insert(&mut self, data: T) -> Handle {
        self.len += 1;

        if self.free_head != u32::MAX {
            // Reuse free slot
            let index = self.free_head as usize;
            match &self.slots[index] {
                Slot::Free {
                    next_free,
                    generation,
                } => {
                    let gen = *generation;
                    self.free_head = *next_free;
                    self.slots[index] = Slot::Occupied {
                        data,
                        generation: gen,
                    };
                    Handle::new(index as u32, gen)
                }
                _ => unreachable!(),
            }
        } else {
            // Allocate new slot
            let index = self.slots.len();
            self.slots.push(Slot::Occupied {
                data,
                generation: 0,
            });
            Handle::new(index as u32, 0)
        }
    }

    /// Remove element by handle
    pub fn remove(&mut self, handle: Handle) -> Option<T> {
        let index = handle.index as usize;
        if index >= self.slots.len() {
            return None;
        }

        match &self.slots[index] {
            Slot::Occupied { generation, .. } if *generation == handle.generation => {
                let new_gen = generation.wrapping_add(1);
                let old = std::mem::replace(
                    &mut self.slots[index],
                    Slot::Free {
                        next_free: self.free_head,
                        generation: new_gen,
                    },
                );
                self.free_head = handle.index;
                self.len -= 1;

                match old {
                    Slot::Occupied { data, .. } => Some(data),
                    _ => unreachable!(),
                }
            }
            _ => None,
        }
    }

    /// Get reference by handle
    #[inline]
    pub fn get(&self, handle: Handle) -> Option<&T> {
        let index = handle.index as usize;
        if index >= self.slots.len() {
            return None;
        }

        match &self.slots[index] {
            Slot::Occupied { data, generation } if *generation == handle.generation => Some(data),
            _ => None,
        }
    }

    /// Get mutable reference by handle
    #[inline]
    pub fn get_mut(&mut self, handle: Handle) -> Option<&mut T> {
        let index = handle.index as usize;
        if index >= self.slots.len() {
            return None;
        }

        match &mut self.slots[index] {
            Slot::Occupied { data, generation } if *generation == handle.generation => Some(data),
            _ => None,
        }
    }

    /// Get reference without bounds check (unsafe)
    ///
    /// # Safety
    /// Caller must ensure handle is valid and points to an occupied slot
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, handle: Handle) -> &T {
        let slot = self.slots.get_unchecked(handle.index as usize);
        match slot {
            Slot::Occupied { data, .. } => data,
            _ => std::hint::unreachable_unchecked(),
        }
    }

    /// Get mutable reference without bounds check (unsafe)
    ///
    /// # Safety
    /// Caller must ensure handle is valid and points to an occupied slot
    #[inline(always)]
    pub unsafe fn get_unchecked_mut(&mut self, handle: Handle) -> &mut T {
        let slot = self.slots.get_unchecked_mut(handle.index as usize);
        match slot {
            Slot::Occupied { data, .. } => data,
            _ => std::hint::unreachable_unchecked(),
        }
    }

    /// Iterate over all occupied slots
    pub fn iter(&self) -> impl Iterator<Item = (Handle, &T)> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(i, slot)| match slot {
                Slot::Occupied { data, generation } => {
                    Some((Handle::new(i as u32, *generation), data))
                }
                _ => None,
            })
    }

    /// Iterate mutably over all occupied slots
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (Handle, &mut T)> {
        self.slots
            .iter_mut()
            .enumerate()
            .filter_map(|(i, slot)| match slot {
                Slot::Occupied { data, generation } => {
                    Some((Handle::new(i as u32, *generation), data))
                }
                _ => None,
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_basic() {
        let mut arena: Arena<i32> = Arena::new();

        let h1 = arena.insert(10);
        let h2 = arena.insert(20);
        let h3 = arena.insert(30);

        assert_eq!(arena.len(), 3);
        assert_eq!(arena.get(h1), Some(&10));
        assert_eq!(arena.get(h2), Some(&20));
        assert_eq!(arena.get(h3), Some(&30));
    }

    #[test]
    fn test_arena_remove_reuse() {
        let mut arena: Arena<i32> = Arena::new();

        let h1 = arena.insert(10);
        let _h2 = arena.insert(20);

        // Remove h1
        assert_eq!(arena.remove(h1), Some(10));
        assert_eq!(arena.len(), 1);

        // Old handle should be invalid
        assert_eq!(arena.get(h1), None);

        // Insert reuses slot
        let h3 = arena.insert(30);
        assert_eq!(h3.index, h1.index);
        assert_ne!(h3.generation, h1.generation);
    }

    #[test]
    fn test_arena_iter() {
        let mut arena: Arena<i32> = Arena::new();
        arena.insert(1);
        arena.insert(2);
        arena.insert(3);

        let sum: i32 = arena.iter().map(|(_, v)| *v).sum();
        assert_eq!(sum, 6);
    }
}

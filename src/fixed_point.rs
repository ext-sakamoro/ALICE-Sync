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

//! Fixed-point arithmetic for deterministic simulation
//!
//! v0.3: Added SIMD support via `wide` crate for vectorized operations.
//!
//! Floating-point numbers (f32/f64) can produce different results
//! across CPU architectures and compiler optimizations.
//! Fixed-point ensures bit-exact results on all platforms.

use serde::{Deserialize, Serialize};
use std::ops::{Add, AddAssign, Mul, Sub};
use wide::i32x4;

/// Q16.16 fixed-point number (32-bit, 16 fractional bits)
/// Range: -32768.0 to 32767.99998 with precision of 0.0000153
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[repr(transparent)]
pub struct Fixed(pub i32);

impl Fixed {
    pub const ONE: Self = Self(1 << 16);
    pub const ZERO: Self = Self(0);
    pub const FRAC_BITS: u32 = 16;
    pub const SCALE: i32 = 1 << 16;

    /// Precomputed reciprocal of SCALE for f32 conversion
    const RCP_SCALE: f32 = 1.0 / (1u32 << 16) as f32;

    #[inline(always)]
    pub const fn from_int(n: i32) -> Self {
        Self(n << 16)
    }

    #[inline(always)]
    pub const fn from_bits(bits: i32) -> Self {
        Self(bits)
    }

    #[inline(always)]
    pub const fn to_bits(self) -> i32 {
        self.0
    }

    #[inline(always)]
    pub fn from_f32(f: f32) -> Self {
        Self((f * Self::SCALE as f32) as i32)
    }

    #[inline(always)]
    pub fn to_f32(self) -> f32 {
        self.0 as f32 * Self::RCP_SCALE
    }

    #[inline(always)]
    pub const fn from_i16(n: i16) -> Self {
        Self((n as i32) << 6)
    }

    #[inline(always)]
    pub const fn to_i16(self) -> i16 {
        (self.0 >> 6) as i16
    }

    #[inline(always)]
    pub const fn saturating_add(self, rhs: Self) -> Self {
        Self(self.0.saturating_add(rhs.0))
    }

    #[inline(always)]
    pub const fn saturating_sub(self, rhs: Self) -> Self {
        Self(self.0.saturating_sub(rhs.0))
    }

    #[inline(always)]
    pub fn saturating_mul(self, rhs: Self) -> Self {
        let result = (self.0 as i64 * rhs.0 as i64) >> 16;
        Self(result.clamp(i32::MIN as i64, i32::MAX as i64) as i32)
    }
}

impl Add for Fixed {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(self.0.wrapping_add(rhs.0))
    }
}

impl AddAssign for Fixed {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.0 = self.0.wrapping_add(rhs.0);
    }
}

impl Sub for Fixed {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0.wrapping_sub(rhs.0))
    }
}

impl Mul for Fixed {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        let result = (self.0 as i64 * rhs.0 as i64) >> 16;
        Self(result as i32)
    }
}

/// 3D vector with fixed-point components (scalar version)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct Vec3Fixed {
    pub x: Fixed,
    pub y: Fixed,
    pub z: Fixed,
}

impl Vec3Fixed {
    pub const ZERO: Self = Self {
        x: Fixed::ZERO,
        y: Fixed::ZERO,
        z: Fixed::ZERO,
    };

    #[inline(always)]
    pub const fn new(x: Fixed, y: Fixed, z: Fixed) -> Self {
        Self { x, y, z }
    }

    #[inline(always)]
    pub fn from_f32(x: f32, y: f32, z: f32) -> Self {
        Self {
            x: Fixed::from_f32(x),
            y: Fixed::from_f32(y),
            z: Fixed::from_f32(z),
        }
    }

    #[inline(always)]
    pub fn to_f32_array(self) -> [f32; 3] {
        [self.x.to_f32(), self.y.to_f32(), self.z.to_f32()]
    }

    #[inline(always)]
    pub const fn to_i16_array(self) -> [i16; 3] {
        [self.x.to_i16(), self.y.to_i16(), self.z.to_i16()]
    }

    #[inline(always)]
    pub const fn from_i16_array(arr: [i16; 3]) -> Self {
        Self {
            x: Fixed::from_i16(arr[0]),
            y: Fixed::from_i16(arr[1]),
            z: Fixed::from_i16(arr[2]),
        }
    }

    /// Convert to SIMD vector for batch operations
    #[inline(always)]
    pub fn to_simd(self) -> Vec3Simd {
        Vec3Simd::from_vec3(self)
    }

    /// Hash for XOR rolling hash (optimized: no branching)
    #[inline(always)]
    pub fn hash_bits(self) -> u64 {
        let x = self.x.0 as u64;
        let y = self.y.0 as u64;
        let z = self.z.0 as u64;
        x ^ (x << 32) ^ y.rotate_left(21) ^ z.rotate_left(42)
    }
}

impl Add for Vec3Fixed {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl AddAssign for Vec3Fixed {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

// ============================================================================
// SIMD Vec3 - Processes x, y, z in a single CPU instruction
// ============================================================================

/// SIMD-accelerated 3D vector (128-bit, processes x,y,z,w in parallel)
#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub struct Vec3Simd {
    /// [x, y, z, 0] packed into a 128-bit SIMD register
    data: i32x4,
}

impl Vec3Simd {
    pub const ZERO: Self = Self { data: i32x4::ZERO };

    /// Create from individual components
    #[inline(always)]
    pub fn new(x: i32, y: i32, z: i32) -> Self {
        Self {
            data: i32x4::new([x, y, z, 0]),
        }
    }

    /// Create from Vec3Fixed
    #[inline(always)]
    pub fn from_vec3(v: Vec3Fixed) -> Self {
        Self {
            data: i32x4::new([v.x.0, v.y.0, v.z.0, 0]),
        }
    }

    /// Create from i16 array (network format)
    #[inline(always)]
    pub fn from_i16_array(arr: [i16; 3]) -> Self {
        Self {
            data: i32x4::new([
                (arr[0] as i32) << 6,
                (arr[1] as i32) << 6,
                (arr[2] as i32) << 6,
                0,
            ]),
        }
    }

    /// Convert to Vec3Fixed
    #[inline(always)]
    pub fn to_vec3(self) -> Vec3Fixed {
        let arr = self.data.to_array();
        Vec3Fixed {
            x: Fixed(arr[0]),
            y: Fixed(arr[1]),
            z: Fixed(arr[2]),
        }
    }

    /// SIMD add (single instruction for x+y+z)
    #[inline(always)]
    #[allow(clippy::should_implement_trait)]
    pub fn add(self, rhs: Self) -> Self {
        Self {
            data: self.data + rhs.data,
        }
    }

    /// SIMD sub
    #[inline(always)]
    #[allow(clippy::should_implement_trait)]
    pub fn sub(self, rhs: Self) -> Self {
        Self {
            data: self.data - rhs.data,
        }
    }

    /// Add and store back to Vec3Fixed (common pattern)
    #[inline(always)]
    pub fn add_to_vec3(self, target: &mut Vec3Fixed) {
        let current = Vec3Simd::from_vec3(*target);
        *target = current.add(self).to_vec3();
    }

    /// Hash for XOR rolling (SIMD horizontal XOR)
    #[inline(always)]
    pub fn hash_bits(self) -> u64 {
        let arr = self.data.to_array();
        let x = arr[0] as u64;
        let y = arr[1] as u64;
        let z = arr[2] as u64;
        x ^ (x << 32) ^ y.rotate_left(21) ^ z.rotate_left(42)
    }
}

impl Add for Vec3Simd {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self::add(self, rhs)
    }
}

impl AddAssign for Vec3Simd {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = Self::add(*self, rhs);
    }
}

// ============================================================================
// Batch operations for multiple vectors
// ============================================================================

/// Process multiple Vec3 additions in parallel
#[inline(always)]
pub fn batch_add_vec3(positions: &mut [Vec3Fixed], deltas: &[Vec3Fixed]) {
    debug_assert_eq!(positions.len(), deltas.len());
    for (pos, delta) in positions.iter_mut().zip(deltas.iter()) {
        let p = Vec3Simd::from_vec3(*pos);
        let d = Vec3Simd::from_vec3(*delta);
        *pos = (p + d).to_vec3();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_arithmetic() {
        let a = Fixed::from_f32(1.5);
        let b = Fixed::from_f32(2.5);
        let sum = a + b;
        assert!((sum.to_f32() - 4.0).abs() < 0.001);
    }

    #[test]
    fn test_fixed_determinism() {
        let a = Fixed::from_bits(0x18000);
        let b = Fixed::from_bits(0x28000);
        let result1 = a + b;
        let result2 = a + b;
        assert_eq!(result1.0, result2.0);
    }

    #[test]
    fn test_simd_add() {
        let a = Vec3Simd::new(100, 200, 300);
        let b = Vec3Simd::new(10, 20, 30);
        let result = a + b;
        let v = result.to_vec3();
        assert_eq!(v.x.0, 110);
        assert_eq!(v.y.0, 220);
        assert_eq!(v.z.0, 330);
    }

    #[test]
    fn test_simd_roundtrip() {
        let original = Vec3Fixed::from_f32(1.5, -2.0, 0.5);
        let simd = original.to_simd();
        let back = simd.to_vec3();
        assert_eq!(original, back);
    }

    #[test]
    fn test_i16_roundtrip() {
        let v = Vec3Fixed::from_f32(1.5, -2.0, 0.5);
        let packed = v.to_i16_array();
        let unpacked = Vec3Fixed::from_i16_array(packed);
        assert!((v.x.to_f32() - unpacked.x.to_f32()).abs() < 0.1);
    }
}

//! This module converts the validity bits of a SIMD register of data into a SIMD mask
//! (see the `SIMDValidity` trait), for each instruction set and lane size.
//!
//! AVX512 uses the validity bits as mask directly. The other instruction sets broadcast
//! the bits to all lanes and set lane `i` when it has bit `i` set.
//!
#[cfg(any(
    all(target_arch = "arm", feature = "nightly_simd"),
    target_arch = "aarch64"
))]
use super::config::NEON;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use super::config::{AVX2, AVX512, SSE};
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(all(target_arch = "arm", feature = "nightly_simd"))]
use std::arch::arm::*;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Converts validity bits into a SIMD mask.
/// This is implemented for each instruction set, for each SIMD mask type and lane size
/// that the `SIMDOps` implementations use.
#[doc(hidden)]
pub trait SIMDValidity<SIMDMaskDtype, const LANE_SIZE: usize> {
    /// Returns the SIMD mask that selects lane `i` iff bit `i` is set (the bits from
    /// `LANE_SIZE` onwards are ignored).
    unsafe fn _mm_validity_mask(bits: u64) -> SIMDMaskDtype;
}

// -------------------------------------- x86 / x86_64 ---------------------------------------

/// Byte `i` has bit `i % 8` set
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const BYTE_BITS: i64 = 0x8040_2010_0804_0201_u64 as i64;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod sse {
    use super::*;

    const BITS_16: __m128i = unsafe { std::mem::transmute([1i16, 2, 4, 8, 16, 32, 64, 128]) };
    const BITS_32: __m128i = unsafe { std::mem::transmute([1i32, 2, 4, 8]) };
    const BITS_64: __m128i = unsafe { std::mem::transmute([1i64, 2]) };

    impl<DTypeStrategy> SIMDValidity<__m128i, 16> for SSE<DTypeStrategy> {
        #[inline(always)]
        unsafe fn _mm_validity_mask(bits: u64) -> __m128i {
            // Copy the 1st byte of the bits to lanes 0-7, and the 2nd byte to lanes 8-15
            let bytes = _mm_shuffle_epi8(
                _mm_cvtsi32_si128(bits as i32),
                _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1),
            );
            let byte_bits = _mm_set1_epi64x(BYTE_BITS);
            _mm_cmpeq_epi8(_mm_and_si128(bytes, byte_bits), byte_bits)
        }
    }

    impl<DTypeStrategy> SIMDValidity<__m128i, 8> for SSE<DTypeStrategy> {
        #[inline(always)]
        unsafe fn _mm_validity_mask(bits: u64) -> __m128i {
            _mm_cmpeq_epi16(_mm_and_si128(_mm_set1_epi16(bits as i16), BITS_16), BITS_16)
        }
    }

    impl<DTypeStrategy> SIMDValidity<__m128i, 4> for SSE<DTypeStrategy> {
        #[inline(always)]
        unsafe fn _mm_validity_mask(bits: u64) -> __m128i {
            _mm_cmpeq_epi32(_mm_and_si128(_mm_set1_epi32(bits as i32), BITS_32), BITS_32)
        }
    }

    impl<DTypeStrategy> SIMDValidity<__m128i, 2> for SSE<DTypeStrategy> {
        #[inline(always)]
        unsafe fn _mm_validity_mask(bits: u64) -> __m128i {
            _mm_cmpeq_epi64(
                _mm_and_si128(_mm_set1_epi64x(bits as i64), BITS_64),
                BITS_64,
            )
        }
    }

    impl<DTypeStrategy> SIMDValidity<__m128, 4> for SSE<DTypeStrategy> {
        #[inline(always)]
        unsafe fn _mm_validity_mask(bits: u64) -> __m128 {
            _mm_castsi128_ps(<Self as SIMDValidity<__m128i, 4>>::_mm_validity_mask(bits))
        }
    }

    impl<DTypeStrategy> SIMDValidity<__m128d, 2> for SSE<DTypeStrategy> {
        #[inline(always)]
        unsafe fn _mm_validity_mask(bits: u64) -> __m128d {
            _mm_castsi128_pd(<Self as SIMDValidity<__m128i, 2>>::_mm_validity_mask(bits))
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx2 {
    use super::*;

    const BITS_16: __m256i = unsafe {
        std::mem::transmute([
            1u16, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
        ])
    };
    const BITS_32: __m256i = unsafe { std::mem::transmute([1i32, 2, 4, 8, 16, 32, 64, 128]) };
    const BITS_64: __m256i = unsafe { std::mem::transmute([1i64, 2, 4, 8]) };

    impl<DTypeStrategy> SIMDValidity<__m256i, 32> for AVX2<DTypeStrategy> {
        #[inline(always)]
        unsafe fn _mm_validity_mask(bits: u64) -> __m256i {
            // Copy byte j of the bits to lanes 8j-8j+7 (the shuffle is per 128-bit lane)
            let bytes = _mm256_shuffle_epi8(
                _mm256_set1_epi32(bits as i32),
                _mm256_setr_epi8(
                    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3,
                    3, 3, 3, 3, 3, 3,
                ),
            );
            let byte_bits = _mm256_set1_epi64x(BYTE_BITS);
            _mm256_cmpeq_epi8(_mm256_and_si256(bytes, byte_bits), byte_bits)
        }
    }

    impl<DTypeStrategy> SIMDValidity<__m256i, 16> for AVX2<DTypeStrategy> {
        #[inline(always)]
        unsafe fn _mm_validity_mask(bits: u64) -> __m256i {
            _mm256_cmpeq_epi16(
                _mm256_and_si256(_mm256_set1_epi16(bits as i16), BITS_16),
                BITS_16,
            )
        }
    }

    impl<DTypeStrategy> SIMDValidity<__m256i, 8> for AVX2<DTypeStrategy> {
        #[inline(always)]
        unsafe fn _mm_validity_mask(bits: u64) -> __m256i {
            _mm256_cmpeq_epi32(
                _mm256_and_si256(_mm256_set1_epi32(bits as i32), BITS_32),
                BITS_32,
            )
        }
    }

    impl<DTypeStrategy> SIMDValidity<__m256i, 4> for AVX2<DTypeStrategy> {
        #[inline(always)]
        unsafe fn _mm_validity_mask(bits: u64) -> __m256i {
            _mm256_cmpeq_epi64(
                _mm256_and_si256(_mm256_set1_epi64x(bits as i64), BITS_64),
                BITS_64,
            )
        }
    }

    // The float masks are used for f32 and f64 with only AVX (and thus no AVX2 integer
    // instructions): the lanes are converted to floats (0.0 or a power of 2) and compared
    // with 0.0.

    impl<DTypeStrategy> SIMDValidity<__m256, 8> for AVX2<DTypeStrategy> {
        #[inline(always)]
        unsafe fn _mm_validity_mask(bits: u64) -> __m256 {
            let lane_bits = _mm256_and_ps(
                _mm256_castsi256_ps(_mm256_set1_epi32(bits as i32)),
                _mm256_castsi256_ps(BITS_32),
            );
            let lane_bits = _mm256_cvtepi32_ps(_mm256_castps_si256(lane_bits));
            _mm256_cmp_ps(lane_bits, _mm256_setzero_ps(), _CMP_NEQ_OQ)
        }
    }

    impl<DTypeStrategy> SIMDValidity<__m256d, 4> for AVX2<DTypeStrategy> {
        #[inline(always)]
        unsafe fn _mm_validity_mask(bits: u64) -> __m256d {
            let lane_bits = _mm_and_si128(_mm_set1_epi32(bits as i32), _mm_setr_epi32(1, 2, 4, 8));
            _mm256_cmp_pd(
                _mm256_cvtepi32_pd(lane_bits),
                _mm256_setzero_pd(),
                _CMP_NEQ_OQ,
            )
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx512 {
    use super::*;

    impl<DTypeStrategy> SIMDValidity<u64, 64> for AVX512<DTypeStrategy> {
        #[inline(always)]
        unsafe fn _mm_validity_mask(bits: u64) -> u64 {
            bits
        }
    }

    impl<DTypeStrategy> SIMDValidity<u32, 32> for AVX512<DTypeStrategy> {
        #[inline(always)]
        unsafe fn _mm_validity_mask(bits: u64) -> u32 {
            bits as u32
        }
    }

    impl<DTypeStrategy> SIMDValidity<u16, 16> for AVX512<DTypeStrategy> {
        #[inline(always)]
        unsafe fn _mm_validity_mask(bits: u64) -> u16 {
            bits as u16
        }
    }

    impl<DTypeStrategy> SIMDValidity<u8, 8> for AVX512<DTypeStrategy> {
        #[inline(always)]
        unsafe fn _mm_validity_mask(bits: u64) -> u8 {
            bits as u8
        }
    }
}

// -------------------------------------- arm / aarch64 --------------------------------------

#[cfg(any(
    all(target_arch = "arm", feature = "nightly_simd"),
    target_arch = "aarch64"
))]
mod neon {
    use super::*;

    const BYTE_BITS: uint8x16_t = unsafe {
        std::mem::transmute([1u8, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128])
    };
    const BITS_16: uint16x8_t = unsafe { std::mem::transmute([1u16, 2, 4, 8, 16, 32, 64, 128]) };
    const BITS_32: uint32x4_t = unsafe { std::mem::transmute([1u32, 2, 4, 8]) };
    #[cfg(target_arch = "aarch64")] // 64-bit data uses the scalar implementation on arm
    const BITS_64: uint64x2_t = unsafe { std::mem::transmute([1u64, 2]) };

    impl<DTypeStrategy> SIMDValidity<uint8x16_t, 16> for NEON<DTypeStrategy> {
        #[inline(always)]
        unsafe fn _mm_validity_mask(bits: u64) -> uint8x16_t {
            // Copy the 1st byte of the bits to lanes 0-7, and the 2nd byte to lanes 8-15
            let bytes = vcombine_u8(vdup_n_u8(bits as u8), vdup_n_u8((bits >> 8) as u8));
            vtstq_u8(bytes, BYTE_BITS)
        }
    }

    impl<DTypeStrategy> SIMDValidity<uint16x8_t, 8> for NEON<DTypeStrategy> {
        #[inline(always)]
        unsafe fn _mm_validity_mask(bits: u64) -> uint16x8_t {
            vtstq_u16(vdupq_n_u16(bits as u16), BITS_16)
        }
    }

    impl<DTypeStrategy> SIMDValidity<uint32x4_t, 4> for NEON<DTypeStrategy> {
        #[inline(always)]
        unsafe fn _mm_validity_mask(bits: u64) -> uint32x4_t {
            vtstq_u32(vdupq_n_u32(bits as u32), BITS_32)
        }
    }

    #[cfg(target_arch = "aarch64")]
    impl<DTypeStrategy> SIMDValidity<uint64x2_t, 2> for NEON<DTypeStrategy> {
        #[inline(always)]
        unsafe fn _mm_validity_mask(bits: u64) -> uint64x2_t {
            vtstq_u64(vdupq_n_u64(bits), BITS_64)
        }
    }
}

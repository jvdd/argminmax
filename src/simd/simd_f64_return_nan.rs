/// Implementation of the argminmax operations for f64 where NaN values take precedence.
/// This implementation returns the index of the first* NaN value if any are present,
/// otherwise it returns the index of the minimum and maximum values.
///
/// To serve this functionality we transform the f64 values to ordinal i64 values:
///     ord_i64 = ((v >> 63) & 0x7FFFFFFFFFFFFFFF) ^ v
///
/// This transformation is a bijection, i.e. it is reversible:
///     v = ((ord_i64 >> 63) & 0x7FFFFFFFFFFFFFFF) ^ ord_i64
///
/// Through this transformation we can perform the argminmax operations on the ordinal
/// integer values and then transform the result back to the original f64 values.
/// This transformation is necessary because comparisons with NaN values are always false.
/// So unless we perform ! <=  as gt and ! >=  as lt the argminmax operations will not
/// add NaN values to the accumulating SIMD register. And as le and ge are significantly
/// more expensive than lt and gt we use this efficient bitwise transformation.
///
/// Also comparing integers is faster than comparing floats:
///   - https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_cmp_pd&ig_expand=886
///   - https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_cmpgt_epi64&ig_expand=1094
///
///
/// ---
///
/// *Note: the first NaN value is only returned iff all NaN values have the same bit
/// representation. When NaN values have different bit representations then the index of
/// the highest / lowest ord_i64 is returned for the
/// SIMDOps::_get_overflow_lane_size_limit() chunk of the data - which is not
/// necessarily the index of the first NaN value.
///

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use super::config::SIMDInstructionSet;
use super::generic::{SIMDArgMinMax, SIMDOps};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use super::task::{max_index_value, min_index_value};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const BIT_SHIFT: i32 = 63;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const MASK_VALUE: i64 = 0x7FFFFFFFFFFFFFFF; // i64::MAX - masks everything but the sign bit

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
fn _i64ord_to_f64(ord_i64: i64) -> f64 {
    let v = ((ord_i64 >> BIT_SHIFT) & MASK_VALUE) ^ ord_i64;
    f64::from_bits(v as u64)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const MAX_INDEX: usize = i64::MAX as usize;

// --------------------------------------- AVX2 ----------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx2 {
    use super::super::config::AVX2;
    use super::*;

    const LANE_SIZE: usize = AVX2::LANE_SIZE_64;
    const LOWER_63_MASK: __m256i = unsafe { std::mem::transmute([MASK_VALUE; LANE_SIZE]) };

    #[inline(always)]
    unsafe fn _f64_as_m256i_to_i64ord(f64_as_m256i: __m256i) -> __m256i {
        // on a scalar: ((v >> 63) & 0x7FFFFFFFFFFFFFFF) ^ v
        // Note: _mm256_srai_epi64 is not available on AVX2.. (only AVX512F)
        //  -> As we only want to shift the sign bit to the first position, we can use
        //     _mm256_srai_epi32 instead, which is available on AVX2, and then copy the
        // sign bit to the next 32 bits (per 64 bit lane).
        let sign_bit_shifted =
            _mm256_shuffle_epi32(_mm256_srai_epi32(f64_as_m256i, BIT_SHIFT), 0b11110101);
        let sign_bit_masked = _mm256_and_si256(sign_bit_shifted, LOWER_63_MASK);
        _mm256_xor_si256(sign_bit_masked, f64_as_m256i)
    }

    #[inline(always)]
    unsafe fn _reg_to_i64_arr(reg: __m256i) -> [i64; LANE_SIZE] {
        std::mem::transmute::<__m256i, [i64; LANE_SIZE]>(reg)
    }

    impl SIMDOps<f64, __m256i, __m256i, LANE_SIZE> for AVX2 {
        const INITIAL_INDEX: __m256i = unsafe { std::mem::transmute([0i64, 1i64, 2i64, 3i64]) };
        const INDEX_INCREMENT: __m256i =
            unsafe { std::mem::transmute([LANE_SIZE as i64; LANE_SIZE]) };
        const MAX_INDEX: usize = MAX_INDEX;

        #[inline(always)]
        unsafe fn _reg_to_arr(_: __m256i) -> [f64; LANE_SIZE] {
            // Not implemented because we will perform the horizontal operations on the
            // signed integer values instead of trying to retransform **only** the values
            // (and thus not the indices) to floats.
            unimplemented!()
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const f64) -> __m256i {
            _f64_as_m256i_to_i64ord(_mm256_loadu_si256(data as *const __m256i))
        }

        #[inline(always)]
        unsafe fn _mm_add(a: __m256i, b: __m256i) -> __m256i {
            _mm256_add_epi64(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: __m256i, b: __m256i) -> __m256i {
            _mm256_cmpgt_epi64(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: __m256i, b: __m256i) -> __m256i {
            _mm256_cmpgt_epi64(b, a)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: __m256i, b: __m256i, mask: __m256i) -> __m256i {
            _mm256_blendv_epi8(a, b, mask)
        }

        #[inline(always)]
        unsafe fn _horiz_min(index: __m256i, value: __m256i) -> (usize, f64) {
            let index_arr: [i64; LANE_SIZE] = _reg_to_i64_arr(index);
            let value_arr: [i64; LANE_SIZE] = _reg_to_i64_arr(value);
            let (min_index, min_value) = min_index_value(&index_arr, &value_arr);
            (min_index as usize, _i64ord_to_f64(min_value))
        }

        #[inline(always)]
        unsafe fn _horiz_max(index: __m256i, value: __m256i) -> (usize, f64) {
            let index_arr: [i64; LANE_SIZE] = _reg_to_i64_arr(index);
            let value_arr: [i64; LANE_SIZE] = _reg_to_i64_arr(value);
            let (max_index, max_value) = max_index_value(&index_arr, &value_arr);
            (max_index as usize, _i64ord_to_f64(max_value))
        }
    }

    impl SIMDArgMinMax<f64, __m256i, __m256i, LANE_SIZE> for AVX2 {
        #[target_feature(enable = "avx2")]
        unsafe fn argminmax(data: &[f64]) -> (usize, usize) {
            Self::_argminmax(data)
        }
    }
}

// ---------------------------------------- SSE ----------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod sse {
    use super::super::config::SSE;
    use super::*;

    const LANE_SIZE: usize = SSE::LANE_SIZE_64;
    const LOWER_63_MASK: __m128i = unsafe { std::mem::transmute([MASK_VALUE; LANE_SIZE]) };

    #[inline(always)]
    unsafe fn _f64_as_m128i_to_i64ord(f64_as_m128i: __m128i) -> __m128i {
        // on a scalar: ((v >> 63) & 0x7FFFFFFFFFFFFFFF) ^ v
        // Note: _mm_srai_epi64 is not available on AVX2.. (only on AVX512F)
        //  -> As we only want to shift the sign bit to the first position, we can use
        //     _mm_srai_epi32 instead, which is available on AVX2, and then copy the
        // sign bit to the next 32 bits (per 64 bit lane).
        let sign_bit_shifted =
            _mm_shuffle_epi32(_mm_srai_epi32(f64_as_m128i, BIT_SHIFT), 0b11110101);
        let sign_bit_masked = _mm_and_si128(sign_bit_shifted, LOWER_63_MASK);
        _mm_xor_si128(sign_bit_masked, f64_as_m128i)
    }

    #[inline(always)]
    unsafe fn _reg_to_i64_arr(reg: __m128i) -> [i64; LANE_SIZE] {
        std::mem::transmute::<__m128i, [i64; LANE_SIZE]>(reg)
    }

    impl SIMDOps<f64, __m128i, __m128i, LANE_SIZE> for SSE {
        const INITIAL_INDEX: __m128i = unsafe { std::mem::transmute([0i64, 1i64]) };
        const INDEX_INCREMENT: __m128i =
            unsafe { std::mem::transmute([LANE_SIZE as i64; LANE_SIZE]) };
        const MAX_INDEX: usize = MAX_INDEX;

        #[inline(always)]
        unsafe fn _reg_to_arr(_: __m128i) -> [f64; LANE_SIZE] {
            // Not implemented because we will perform the horizontal operations on the
            // signed integer values instead of trying to retransform **only** the values
            // (and thus not the indices) to floats.
            unimplemented!()
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const f64) -> __m128i {
            _f64_as_m128i_to_i64ord(_mm_loadu_si128(data as *const __m128i))
        }

        #[inline(always)]
        unsafe fn _mm_add(a: __m128i, b: __m128i) -> __m128i {
            _mm_add_epi64(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: __m128i, b: __m128i) -> __m128i {
            _mm_cmpgt_epi64(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: __m128i, b: __m128i) -> __m128i {
            _mm_cmpgt_epi64(b, a)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: __m128i, b: __m128i, mask: __m128i) -> __m128i {
            _mm_blendv_epi8(a, b, mask)
        }

        #[inline(always)]
        unsafe fn _horiz_min(index: __m128i, value: __m128i) -> (usize, f64) {
            let index_arr: [i64; LANE_SIZE] = _reg_to_i64_arr(index);
            let value_arr: [i64; LANE_SIZE] = _reg_to_i64_arr(value);
            let (min_index, min_value) = min_index_value(&index_arr, &value_arr);
            (min_index as usize, _i64ord_to_f64(min_value))
        }

        #[inline(always)]
        unsafe fn _horiz_max(index: __m128i, value: __m128i) -> (usize, f64) {
            let index_arr: [i64; LANE_SIZE] = _reg_to_i64_arr(index);
            let value_arr: [i64; LANE_SIZE] = _reg_to_i64_arr(value);
            let (max_index, max_value) = max_index_value(&index_arr, &value_arr);
            (max_index as usize, _i64ord_to_f64(max_value))
        }
    }

    impl SIMDArgMinMax<f64, __m128i, __m128i, LANE_SIZE> for SSE {
        #[target_feature(enable = "sse4.2")]
        unsafe fn argminmax(data: &[f64]) -> (usize, usize) {
            Self::_argminmax(data)
        }
    }
}

// -------------------------------------- AVX512 ---------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx512 {
    use super::super::config::AVX512;
    use super::*;

    const LANE_SIZE: usize = AVX512::LANE_SIZE_64;
    const LOWER_63_MASK: __m512i = unsafe { std::mem::transmute([MASK_VALUE; LANE_SIZE]) };

    #[inline(always)]
    unsafe fn _f64_as_m512i_to_i64ord(f64_as_m512i: __m512i) -> __m512i {
        // on a scalar: ((v >> 63) & 0x7FFFFFFFFFFFFFFF) ^ v
        let sign_bit_shifted = _mm512_srai_epi64(f64_as_m512i, BIT_SHIFT as u32);
        let sign_bit_masked = _mm512_and_si512(sign_bit_shifted, LOWER_63_MASK);
        _mm512_xor_si512(sign_bit_masked, f64_as_m512i)
    }

    #[inline(always)]
    unsafe fn _reg_to_i64_arr(reg: __m512i) -> [i64; LANE_SIZE] {
        std::mem::transmute::<__m512i, [i64; LANE_SIZE]>(reg)
    }

    impl SIMDOps<f64, __m512i, u8, LANE_SIZE> for AVX512 {
        const INITIAL_INDEX: __m512i =
            unsafe { std::mem::transmute([0i64, 1i64, 2i64, 3i64, 4i64, 5i64, 6i64, 7i64]) };
        const INDEX_INCREMENT: __m512i =
            unsafe { std::mem::transmute([LANE_SIZE as i64; LANE_SIZE]) };
        const MAX_INDEX: usize = MAX_INDEX;

        #[inline(always)]
        unsafe fn _reg_to_arr(_: __m512i) -> [f64; LANE_SIZE] {
            // Not implemented because we will perform the horizontal operations on the
            // signed integer values instead of trying to retransform **only** the values
            // (and thus not the indices) to floats.
            unimplemented!()
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const f64) -> __m512i {
            _f64_as_m512i_to_i64ord(_mm512_loadu_epi64(data as *const i64))
        }

        #[inline(always)]
        unsafe fn _mm_add(a: __m512i, b: __m512i) -> __m512i {
            _mm512_add_epi64(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: __m512i, b: __m512i) -> u8 {
            _mm512_cmpgt_epi64_mask(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: __m512i, b: __m512i) -> u8 {
            _mm512_cmpgt_epi64_mask(b, a)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: __m512i, b: __m512i, mask: u8) -> __m512i {
            _mm512_mask_blend_epi64(mask, a, b)
        }

        #[inline(always)]
        unsafe fn _horiz_min(index: __m512i, value: __m512i) -> (usize, f64) {
            let index_arr: [i64; LANE_SIZE] = _reg_to_i64_arr(index);
            let value_arr: [i64; LANE_SIZE] = _reg_to_i64_arr(value);
            let (min_index, min_value) = min_index_value(&index_arr, &value_arr);
            (min_index as usize, _i64ord_to_f64(min_value))
        }

        #[inline(always)]
        unsafe fn _horiz_max(index: __m512i, value: __m512i) -> (usize, f64) {
            let index_arr: [i64; LANE_SIZE] = _reg_to_i64_arr(index);
            let value_arr: [i64; LANE_SIZE] = _reg_to_i64_arr(value);
            let (max_index, max_value) = max_index_value(&index_arr, &value_arr);
            (max_index as usize, _i64ord_to_f64(max_value))
        }
    }

    impl SIMDArgMinMax<f64, __m512i, u8, LANE_SIZE> for AVX512 {
        #[target_feature(enable = "avx512f")]
        unsafe fn argminmax(data: &[f64]) -> (usize, usize) {
            Self::_argminmax(data)
        }
    }
}

// --------------------------------------- NEON ----------------------------------------

// There are no NEON intrinsics for f64, so we need to use the scalar version.
//   although NEON intrinsics exist for i64 and u64, we cannot use them as
//   they there is no 64-bit variant (of any data type) for the following three
//   intrinsics: vadd_, vcgt_, vclt_

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
mod neon {
    use super::super::config::NEON;
    use super::super::generic::{unimpl_SIMDArgMinMax, unimpl_SIMDOps};
    use super::*;

    // We need to (un)implement the SIMD trait for the NEON struct as otherwise the
    // compiler will complain that the trait is not implemented for the struct -
    // even though we are not using the trait for the NEON struct when dealing with
    // > 64 bit data types.
    unimpl_SIMDOps!(f64, usize, NEON);
    unimpl_SIMDArgMinMax!(f64, usize, NEON);
}

// ======================================= TESTS =======================================

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(test)]
mod tests {
    use rstest::rstest;
    use rstest_reuse::{self, *};

    use crate::scalar::generic::scalar_argminmax;
    use crate::simd::config::{AVX2, AVX512, SSE};
    use crate::SIMDArgMinMax;

    use super::super::test_utils::{test_long_array_argminmax, test_random_runs_argminmax};
    // Float specific tests
    use super::super::test_utils::{test_return_infs_argminmax, test_return_nans_argminmax};

    use dev_utils::utils;

    fn get_array_f64(n: usize) -> Vec<f64> {
        utils::get_random_array(n, f64::MIN, f64::MAX)
    }

    // ------------ Template for x86 / x86_64 -------------

    #[template]
    #[rstest]
    #[case::sse(SSE, is_x86_feature_detected!("sse4.2"))]
    #[case::avx2(AVX2, is_x86_feature_detected!("avx2"))]
    #[case::avx512(AVX512, is_x86_feature_detected!("avx512f"))]
    fn simd_implementations<T, SIMDV, SIMDM, const LANE_SIZE: usize>(
        #[case] _simd: T,
        #[case] simd_available: bool,
    ) {
    }

    // ----------------- The actual tests -----------------

    #[apply(simd_implementations)]
    fn test_first_index_is_returned_when_identical_values_found<
        T,
        SIMDV,
        SIMDM,
        const LANE_SIZE: usize,
    >(
        #[case] _simd: T, // This is just to make sure the template is applied
        #[case] simd_available: bool,
    ) where
        T: SIMDArgMinMax<f64, SIMDV, SIMDM, LANE_SIZE>,
        SIMDV: Copy,
        SIMDM: Copy,
    {
        if !simd_available {
            return;
        }

        let data = [
            10.,
            f64::MAX,
            6.,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
            f64::MAX,
            10_000.0,
        ];
        let data: &[f64] = &data;

        let (argmin_index, argmax_index) = scalar_argminmax(data);
        assert_eq!(argmin_index, 3);
        assert_eq!(argmax_index, 1);

        let (argmin_simd_index, argmax_simd_index) = unsafe { T::argminmax(data) };
        assert_eq!(argmin_simd_index, 3);
        assert_eq!(argmax_simd_index, 1);
    }

    #[apply(simd_implementations)]
    fn test_return_same_result<T, SIMDV, SIMDM, const LANE_SIZE: usize>(
        #[case] _simd: T, // This is just to make sure the template is applied
        #[case] simd_available: bool,
    ) where
        T: SIMDArgMinMax<f64, SIMDV, SIMDM, LANE_SIZE>,
        SIMDV: Copy,
        SIMDM: Copy,
    {
        if !simd_available {
            return;
        }
        test_long_array_argminmax(get_array_f64, scalar_argminmax, T::argminmax);
        test_random_runs_argminmax(get_array_f64, scalar_argminmax, T::argminmax);
    }

    #[apply(simd_implementations)]
    fn test_return_infs<T, SIMDV, SIMDM, const LANE_SIZE: usize>(
        #[case] _simd: T, // This is just to make sure the template is applied
        #[case] simd_available: bool,
    ) where
        T: SIMDArgMinMax<f64, SIMDV, SIMDM, LANE_SIZE>,
        SIMDV: Copy,
        SIMDM: Copy,
    {
        if !simd_available {
            return;
        }
        test_return_infs_argminmax(get_array_f64, scalar_argminmax, T::argminmax);
    }

    #[apply(simd_implementations)]
    fn test_return_nans<T, SIMDV, SIMDM, const LANE_SIZE: usize>(
        #[case] _simd: T, // This is just to make sure the template is applied
        #[case] simd_available: bool,
    ) where
        T: SIMDArgMinMax<f64, SIMDV, SIMDM, LANE_SIZE>,
        SIMDV: Copy,
        SIMDM: Copy,
    {
        if !simd_available {
            return;
        }
        test_return_nans_argminmax(get_array_f64, scalar_argminmax, T::argminmax);
    }
}
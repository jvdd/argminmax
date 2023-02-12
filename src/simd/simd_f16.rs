/// Default implementation of the argminmax operations for f16.
/// This implementation returns the index of the first NaN value if any are present,
/// otherwise it returns the index of the minimum and maximum values.
///
/// To serve this functionality we transform the f16 values to ordinal i32 values:
///     ord_i16 = ((v >> 15) & 0x7FFFFFFF) ^ v
///
/// This transformation is a bijection, i.e. it is reversible:
///     v = ((ord_i16 >> 15) & 0x7FFFFFFF) ^ ord_i16
///
/// Through this transformation we can perform the argminmax operations on the ordinal
/// integer values and then transform the result back to the original f16 values.
/// This transformation is necessary because comparisons with NaN values are always false.
/// So unless we perform ! <=  as gt and ! >=  as lt the argminmax operations will not
/// add NaN values to the accumulating SIMD register. And as le and ge are significantly
/// more expensive than lt and gt we use this efficient bitwise transformation.
///
/// Note that most x86 CPUs do not support f16 instructions - making this implementation
/// multitudes (up to 300x) faster than trying to use a scalar implementation.
///

// TODO: this file should implement the SIMDInstructionSet instead of the FloatIgnoreNans struct
//  => this code returns the nans and thus not ignores them

#[cfg(feature = "half")]
use super::config::SIMDInstructionSet;
#[cfg(feature = "half")]
use super::generic::{SIMDArgMinMaxIgnoreNaN, SIMDOps, SIMDSetOps};

#[cfg(feature = "half")]
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(feature = "half")]
#[cfg(target_arch = "arm")]
use std::arch::arm::*;
#[cfg(feature = "half")]
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(feature = "half")]
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(feature = "half")]
use half::f16;

#[cfg(feature = "half")]
const BIT_SHIFT: i32 = 15;
#[cfg(feature = "half")]
const MASK_VALUE: i16 = 0x7FFF; // i16::MAX - MASKS everything but the sign bit

#[cfg(feature = "half")]
#[inline(always)]
fn _i16ord_to_f16(ord_i16: i16) -> f16 {
    let v = ((ord_i16 >> BIT_SHIFT) & MASK_VALUE) ^ ord_i16;
    unsafe { std::mem::transmute::<i16, f16>(v) }
}

#[cfg(feature = "half")]
const MAX_INDEX: usize = i16::MAX as usize;

// ------------------------------------------ AVX2 ------------------------------------------

#[cfg(feature = "half")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx2 {
    use super::super::config::{AVX2IgnoreNaN, AVX2};
    use super::*;

    const LANE_SIZE: usize = AVX2::LANE_SIZE_16;
    const LOWER_15_MASK: __m256i = unsafe { std::mem::transmute([MASK_VALUE; LANE_SIZE]) };

    #[inline(always)]
    unsafe fn _f16_as_m256i_to_i16ord(f16_as_m256i: __m256i) -> __m256i {
        // on a scalar: ((v >> 15) & 0x7FFF) ^ v
        let sign_bit_shifted = _mm256_srai_epi16(f16_as_m256i, BIT_SHIFT);
        let sign_bit_masked = _mm256_and_si256(sign_bit_shifted, LOWER_15_MASK);
        _mm256_xor_si256(sign_bit_masked, f16_as_m256i)
        // TODO: investigate if this is faster
        // _mm256_xor_si256(
        //     _mm256_srai_epi16(f16_as_m256i, 15),
        //     _mm256_and_si256(f16_as_m256i, LOWER_15_MASK),
        // )
    }

    #[inline(always)]
    unsafe fn _reg_to_i16_arr(reg: __m256i) -> [i16; LANE_SIZE] {
        std::mem::transmute::<__m256i, [i16; LANE_SIZE]>(reg)
    }

    impl SIMDOps<f16, __m256i, __m256i, LANE_SIZE> for AVX2IgnoreNaN {
        const INITIAL_INDEX: __m256i = unsafe {
            std::mem::transmute([
                0i16, 1i16, 2i16, 3i16, 4i16, 5i16, 6i16, 7i16, 8i16, 9i16, 10i16, 11i16, 12i16,
                13i16, 14i16, 15i16,
            ])
        };
        const INDEX_INCREMENT: __m256i =
            unsafe { std::mem::transmute([LANE_SIZE as i16; LANE_SIZE]) };
        const MAX_INDEX: usize = MAX_INDEX;

        #[inline(always)]
        unsafe fn _reg_to_arr(_: __m256i) -> [f16; LANE_SIZE] {
            // Not implemented because we will perform the horizontal operations on the
            // signed integer values instead of trying to retransform **only** the values
            // (and thus not the indices) to floats.
            unimplemented!()
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const f16) -> __m256i {
            _f16_as_m256i_to_i16ord(_mm256_loadu_si256(data as *const __m256i))
        }

        #[inline(always)]
        unsafe fn _mm_add(a: __m256i, b: __m256i) -> __m256i {
            _mm256_add_epi16(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: __m256i, b: __m256i) -> __m256i {
            _mm256_cmpgt_epi16(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: __m256i, b: __m256i) -> __m256i {
            _mm256_cmpgt_epi16(b, a)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: __m256i, b: __m256i, mask: __m256i) -> __m256i {
            _mm256_blendv_epi8(a, b, mask)
        }

        #[inline(always)]
        unsafe fn _horiz_min(index: __m256i, value: __m256i) -> (usize, f16) {
            // 0. Find the minimum value
            let mut vmin: __m256i = value;
            vmin = _mm256_min_epi16(vmin, _mm256_permute2x128_si256(vmin, vmin, 1));
            vmin = _mm256_min_epi16(vmin, _mm256_alignr_epi8(vmin, vmin, 8));
            vmin = _mm256_min_epi16(vmin, _mm256_alignr_epi8(vmin, vmin, 4));
            vmin = _mm256_min_epi16(vmin, _mm256_alignr_epi8(vmin, vmin, 2));
            let min_value: i16 = _mm256_extract_epi16(vmin, 0) as i16;

            // Extract the index of the minimum value
            // 1. Create a mask with the index of the minimum value
            let mask = _mm256_cmpeq_epi16(value, vmin);
            // 2. Blend the mask with the index
            let search_index = _mm256_blendv_epi8(
                _mm256_set1_epi16(i16::MAX), // if mask is 0, use i16::MAX
                index,                       // if mask is 1, use index
                mask,
            );
            // 3. Find the minimum index
            let mut imin: __m256i = search_index;
            imin = _mm256_min_epi16(imin, _mm256_permute2x128_si256(imin, imin, 1));
            imin = _mm256_min_epi16(imin, _mm256_alignr_epi8(imin, imin, 8));
            imin = _mm256_min_epi16(imin, _mm256_alignr_epi8(imin, imin, 4));
            imin = _mm256_min_epi16(imin, _mm256_alignr_epi8(imin, imin, 2));
            let min_index: usize = _mm256_extract_epi16(imin, 0) as usize;

            (min_index, _i16ord_to_f16(min_value))
        }

        #[inline(always)]
        unsafe fn _horiz_max(index: __m256i, value: __m256i) -> (usize, f16) {
            // 0. Find the maximum value
            let mut vmax: __m256i = value;
            vmax = _mm256_max_epi16(vmax, _mm256_permute2x128_si256(vmax, vmax, 1));
            vmax = _mm256_max_epi16(vmax, _mm256_alignr_epi8(vmax, vmax, 8));
            vmax = _mm256_max_epi16(vmax, _mm256_alignr_epi8(vmax, vmax, 4));
            vmax = _mm256_max_epi16(vmax, _mm256_alignr_epi8(vmax, vmax, 2));
            let max_value: i16 = _mm256_extract_epi16(vmax, 0) as i16;

            // Extract the index of the maximum value
            // 1. Create a mask with the index of the maximum value
            let mask = _mm256_cmpeq_epi16(value, vmax);
            // 2. Blend the mask with the index
            let search_index = _mm256_blendv_epi8(
                _mm256_set1_epi16(i16::MAX), // if mask is 0, use i16::MAX
                index,                       // if mask is 1, use index
                mask,
            );
            // 3. Find the maximum index
            let mut imin: __m256i = search_index;
            imin = _mm256_min_epi16(imin, _mm256_permute2x128_si256(imin, imin, 1));
            imin = _mm256_min_epi16(imin, _mm256_alignr_epi8(imin, imin, 8));
            imin = _mm256_min_epi16(imin, _mm256_alignr_epi8(imin, imin, 4));
            imin = _mm256_min_epi16(imin, _mm256_alignr_epi8(imin, imin, 2));
            let max_index: usize = _mm256_extract_epi16(imin, 0) as usize;

            (max_index, _i16ord_to_f16(max_value))
        }
    }

    impl SIMDSetOps<f16, __m256i> for AVX2IgnoreNaN {
        #[inline(always)]
        unsafe fn _mm_set1(a: f16) -> __m256i {
            _mm256_set1_epi16(a.to_bits() as i16)
        }
    }

    impl SIMDArgMinMaxIgnoreNaN<f16, __m256i, __m256i, LANE_SIZE> for AVX2IgnoreNaN {
        #[target_feature(enable = "avx2")]
        unsafe fn argminmax(data: &[f16]) -> (usize, usize) {
            Self::_argminmax(data)
        }
    }

    //----- TESTS -----

    #[cfg(test)]
    mod tests {
        use super::AVX2IgnoreNaN as AVX2;
        use super::SIMDArgMinMaxIgnoreNaN;
        use crate::scalar::generic::scalar_argminmax;

        use half::f16;

        extern crate dev_utils;
        use dev_utils::utils;

        fn get_array_f16(n: usize) -> Vec<f16> {
            let arr = utils::get_random_array(n, i16::MIN, i16::MAX);
            arr.iter().map(|x| f16::from_f32(*x as f32)).collect()
        }

        #[test]
        fn test_both_versions_return_the_same_results() {
            if !is_x86_feature_detected!("avx2") {
                return;
            }

            let data: &[f16] = &get_array_f16(1025);
            assert_eq!(data.len() % 8, 1);

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX2::argminmax(data) };
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }

        #[test]
        fn test_first_index_is_returned_when_identical_values_found() {
            if !is_x86_feature_detected!("avx2") {
                return;
            }

            let data = [
                f16::from_f32(10.),
                f16::MAX,
                f16::from_f32(6.),
                f16::NEG_INFINITY,
                f16::NEG_INFINITY,
                f16::MAX,
                f16::from_f32(5_000.0),
            ];
            let data: Vec<f16> = data.iter().map(|x| *x).collect();
            let data: &[f16] = &data;

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            assert_eq!(argmin_index, 3);
            assert_eq!(argmax_index, 1);

            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX2::argminmax(data) };
            assert_eq!(argmin_simd_index, 3);
            assert_eq!(argmax_simd_index, 1);
        }

        #[test]
        fn test_no_overflow() {
            if !is_x86_feature_detected!("avx2") {
                return;
            }

            let n: usize = 1 << 18;
            let data: &[f16] = &get_array_f16(n);

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX2::argminmax(data) };
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }

        #[test]
        fn test_many_random_runs() {
            if !is_x86_feature_detected!("avx2") {
                return;
            }

            for _ in 0..10_000 {
                let data: &[f16] = &get_array_f16(32 * 8 + 1);
                let (argmin_index, argmax_index) = scalar_argminmax(data);
                let (argmin_simd_index, argmax_simd_index) = unsafe { AVX2::argminmax(data) };
                assert_eq!(argmin_index, argmin_simd_index);
                assert_eq!(argmax_index, argmax_simd_index);
            }
        }
    }
}

// ----------------------------------------- SSE -----------------------------------------

#[cfg(feature = "half")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod sse {
    use super::super::config::{SSEIgnoreNaN, SSE};
    use super::*;

    const LANE_SIZE: usize = SSE::LANE_SIZE_16;
    const LOWER_15_MASK: __m128i = unsafe { std::mem::transmute([MASK_VALUE; LANE_SIZE]) };

    #[inline(always)]
    unsafe fn _f16_as_m128i_to_i16ord(f16_as_m128i: __m128i) -> __m128i {
        // on a scalar: ((v >> 15) & 0x7FFF) ^ v
        let sign_bit_shifted = _mm_srai_epi16(f16_as_m128i, BIT_SHIFT);
        let sign_bit_masked = _mm_and_si128(sign_bit_shifted, LOWER_15_MASK);
        _mm_xor_si128(sign_bit_masked, f16_as_m128i)
    }

    #[inline(always)]
    unsafe fn _reg_to_i16_arr(reg: __m128i) -> [i16; LANE_SIZE] {
        std::mem::transmute::<__m128i, [i16; LANE_SIZE]>(reg)
    }

    impl SIMDOps<f16, __m128i, __m128i, LANE_SIZE> for SSEIgnoreNaN {
        const INITIAL_INDEX: __m128i =
            unsafe { std::mem::transmute([0i16, 1i16, 2i16, 3i16, 4i16, 5i16, 6i16, 7i16]) };
        const INDEX_INCREMENT: __m128i =
            unsafe { std::mem::transmute([LANE_SIZE as i16; LANE_SIZE]) };
        const MAX_INDEX: usize = MAX_INDEX;

        #[inline(always)]
        unsafe fn _reg_to_arr(_: __m128i) -> [f16; LANE_SIZE] {
            // Not implemented because we will perform the horizontal operations on the
            // signed integer values instead of trying to retransform **only** the values
            // (and thus not the indices) to floats.
            unimplemented!()
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const f16) -> __m128i {
            _f16_as_m128i_to_i16ord(_mm_loadu_si128(data as *const __m128i))
        }

        #[inline(always)]
        unsafe fn _mm_add(a: __m128i, b: __m128i) -> __m128i {
            _mm_add_epi16(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: __m128i, b: __m128i) -> __m128i {
            _mm_cmpgt_epi16(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: __m128i, b: __m128i) -> __m128i {
            _mm_cmplt_epi16(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: __m128i, b: __m128i, mask: __m128i) -> __m128i {
            _mm_blendv_epi8(a, b, mask)
        }

        #[inline(always)]
        unsafe fn _horiz_min(index: __m128i, value: __m128i) -> (usize, f16) {
            // 0. Find the minimum value
            let mut vmin: __m128i = value;
            vmin = _mm_min_epi16(vmin, _mm_alignr_epi8(vmin, vmin, 8));
            vmin = _mm_min_epi16(vmin, _mm_alignr_epi8(vmin, vmin, 4));
            vmin = _mm_min_epi16(vmin, _mm_alignr_epi8(vmin, vmin, 2));
            let min_value: i16 = _mm_extract_epi16(vmin, 0) as i16;

            // Extract the index of the minimum value
            // 1. Create a mask with the index of the minimum value
            let mask = _mm_cmpeq_epi16(value, vmin);
            // 2. Blend the mask with the index
            let search_index = _mm_blendv_epi8(
                _mm_set1_epi16(i16::MAX), // if mask is 0, use i16::MAX
                index,                    // if mask is 1, use index
                mask,
            );
            // 3. Find the minimum index
            let mut imin: __m128i = search_index;
            imin = _mm_min_epi16(imin, _mm_alignr_epi8(imin, imin, 8));
            imin = _mm_min_epi16(imin, _mm_alignr_epi8(imin, imin, 4));
            imin = _mm_min_epi16(imin, _mm_alignr_epi8(imin, imin, 2));
            let min_index: usize = _mm_extract_epi16(imin, 0) as usize;

            (min_index, _i16ord_to_f16(min_value))
        }

        #[inline(always)]
        unsafe fn _horiz_max(index: __m128i, value: __m128i) -> (usize, f16) {
            // 0. Find the maximum value
            let mut vmax: __m128i = value;
            vmax = _mm_max_epi16(vmax, _mm_alignr_epi8(vmax, vmax, 8));
            vmax = _mm_max_epi16(vmax, _mm_alignr_epi8(vmax, vmax, 4));
            vmax = _mm_max_epi16(vmax, _mm_alignr_epi8(vmax, vmax, 2));
            let max_value: i16 = _mm_extract_epi16(vmax, 0) as i16;

            // Extract the index of the maximum value
            // 1. Create a mask with the index of the maximum value
            let mask = _mm_cmpeq_epi16(value, vmax);
            // 2. Blend the mask with the index
            let search_index = _mm_blendv_epi8(
                _mm_set1_epi16(i16::MAX), // if mask is 0, use i8::MAX
                index,                    // if mask is 1, use index
                mask,
            );
            // 3. Find the maximum index
            let mut imin: __m128i = search_index;
            imin = _mm_min_epi16(imin, _mm_alignr_epi8(imin, imin, 8));
            imin = _mm_min_epi16(imin, _mm_alignr_epi8(imin, imin, 4));
            imin = _mm_min_epi16(imin, _mm_alignr_epi8(imin, imin, 2));
            let max_index: usize = _mm_extract_epi16(imin, 0) as usize;

            (max_index, _i16ord_to_f16(max_value))
        }
    }

    impl SIMDSetOps<f16, __m128i> for SSEIgnoreNaN {
        #[inline(always)]
        unsafe fn _mm_set1(a: f16) -> __m128i {
            _mm_set1_epi16(a.to_bits() as i16)
        }
    }

    impl SIMDArgMinMaxIgnoreNaN<f16, __m128i, __m128i, LANE_SIZE> for SSEIgnoreNaN {
        #[target_feature(enable = "sse4.1")]
        unsafe fn argminmax(data: &[f16]) -> (usize, usize) {
            Self::_argminmax(data)
        }
    }

    // ----------------------------------------- TESTS -----------------------------------------

    #[cfg(test)]
    mod tests {
        use super::SIMDArgMinMaxIgnoreNaN;
        use super::SSEIgnoreNaN as SSE;
        use crate::scalar::generic::scalar_argminmax;

        use half::f16;

        extern crate dev_utils;
        use dev_utils::utils;

        fn get_array_f16(n: usize) -> Vec<f16> {
            let arr = utils::get_random_array(n, i16::MIN, i16::MAX);
            arr.iter().map(|x| f16::from_f32(*x as f32)).collect()
        }

        #[test]
        fn test_both_versions_return_the_same_results() {
            let data: &[f16] = &get_array_f16(1025);
            assert_eq!(data.len() % 8, 1);

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            let (argmin_simd_index, argmax_simd_index) = unsafe { SSE::argminmax(data) };
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }

        #[test]
        fn test_first_index_is_returned_when_identical_values_found() {
            let data = [
                f16::from_f32(10.),
                f16::MAX,
                f16::from_f32(6.),
                f16::NEG_INFINITY,
                f16::NEG_INFINITY,
                f16::MAX,
                f16::from_f32(5_000.0),
            ];
            let data: Vec<f16> = data.iter().map(|x| *x).collect();
            let data: &[f16] = &data;

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            assert_eq!(argmin_index, 3);
            assert_eq!(argmax_index, 1);

            let (argmin_simd_index, argmax_simd_index) = unsafe { SSE::argminmax(data) };
            assert_eq!(argmin_simd_index, 3);
            assert_eq!(argmax_simd_index, 1);
        }

        #[test]
        fn test_no_overflow() {
            let n: usize = 1 << 18;
            let data: &[f16] = &get_array_f16(n);

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            let (argmin_simd_index, argmax_simd_index) = unsafe { SSE::argminmax(data) };
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }

        #[test]
        fn test_many_random_runs() {
            for _ in 0..10_000 {
                let data: &[f16] = &get_array_f16(32 * 8 + 1);
                let (argmin_index, argmax_index) = scalar_argminmax(data);
                let (argmin_simd_index, argmax_simd_index) = unsafe { SSE::argminmax(data) };
                assert_eq!(argmin_index, argmin_simd_index);
                assert_eq!(argmax_index, argmax_simd_index);
            }
        }
    }
}

// --------------------------------------- AVX512 ----------------------------------------

#[cfg(feature = "half")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx512 {
    use super::super::config::{AVX512IgnoreNaN, AVX512};
    use super::*;

    const LANE_SIZE: usize = AVX512::LANE_SIZE_16;
    const LOWER_15_MASK: __m512i = unsafe { std::mem::transmute([MASK_VALUE; LANE_SIZE]) };

    #[inline(always)]
    unsafe fn _f16_as_m521i_to_i16ord(f16_as_m512i: __m512i) -> __m512i {
        // on a scalar: ((v >> 15) & 0x7FFF) ^ v
        let sign_bit_shifted = _mm512_srai_epi16(f16_as_m512i, BIT_SHIFT as u32);
        let sign_bit_masked = _mm512_and_si512(sign_bit_shifted, LOWER_15_MASK);
        _mm512_xor_si512(f16_as_m512i, sign_bit_masked)
    }

    #[inline(always)]
    unsafe fn _reg_to_i16_arr(reg: __m512i) -> [i16; LANE_SIZE] {
        std::mem::transmute::<__m512i, [i16; LANE_SIZE]>(reg)
    }

    impl SIMDOps<f16, __m512i, u32, LANE_SIZE> for AVX512IgnoreNaN {
        const INITIAL_INDEX: __m512i = unsafe {
            std::mem::transmute([
                0i16, 1i16, 2i16, 3i16, 4i16, 5i16, 6i16, 7i16, 8i16, 9i16, 10i16, 11i16, 12i16,
                13i16, 14i16, 15i16, 16i16, 17i16, 18i16, 19i16, 20i16, 21i16, 22i16, 23i16, 24i16,
                25i16, 26i16, 27i16, 28i16, 29i16, 30i16, 31i16,
            ])
        };
        const INDEX_INCREMENT: __m512i =
            unsafe { std::mem::transmute([LANE_SIZE as i16; LANE_SIZE]) };
        const MAX_INDEX: usize = MAX_INDEX;

        #[inline(always)]
        unsafe fn _reg_to_arr(_: __m512i) -> [f16; LANE_SIZE] {
            // Not implemented because we will perform the horizontal operations on the
            // signed integer values instead of trying to retransform **only** the values
            // (and thus not the indices) to floats.
            unimplemented!()
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const f16) -> __m512i {
            _f16_as_m521i_to_i16ord(_mm512_loadu_epi16(data as *const i16))
        }

        #[inline(always)]
        unsafe fn _mm_add(a: __m512i, b: __m512i) -> __m512i {
            _mm512_add_epi16(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: __m512i, b: __m512i) -> u32 {
            _mm512_cmpgt_epi16_mask(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: __m512i, b: __m512i) -> u32 {
            _mm512_cmplt_epi16_mask(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: __m512i, b: __m512i, mask: u32) -> __m512i {
            _mm512_mask_blend_epi16(mask, a, b)
        }

        #[inline(always)]
        unsafe fn _horiz_min(index: __m512i, value: __m512i) -> (usize, f16) {
            // 0. Find the minimum value
            let mut vmin: __m512i = value;
            vmin = _mm512_min_epi16(vmin, _mm512_alignr_epi32(vmin, vmin, 8));
            vmin = _mm512_min_epi16(vmin, _mm512_alignr_epi32(vmin, vmin, 4));
            vmin = _mm512_min_epi16(vmin, _mm512_alignr_epi8(vmin, vmin, 8));
            vmin = _mm512_min_epi16(vmin, _mm512_alignr_epi8(vmin, vmin, 4));
            vmin = _mm512_min_epi16(vmin, _mm512_alignr_epi8(vmin, vmin, 2));
            let min_value: i16 = _mm_extract_epi16(_mm512_castsi512_si128(vmin), 0) as i16;

            // Extract the index of the minimum value
            // 1. Create a mask with the index of the minimum value
            let mask = _mm512_cmpeq_epi16_mask(value, vmin);
            // 2. Blend the mask with the index
            let search_index = _mm512_mask_blend_epi16(
                mask,
                _mm512_set1_epi16(i16::MAX), // if mask is 0, use i16::MAX
                index,                       // if mask is 1, use index
            );
            // 3. Find the minimum index
            let mut imin: __m512i = search_index;
            imin = _mm512_min_epi16(imin, _mm512_alignr_epi32(imin, imin, 8));
            imin = _mm512_min_epi16(imin, _mm512_alignr_epi32(imin, imin, 4));
            imin = _mm512_min_epi16(imin, _mm512_alignr_epi8(imin, imin, 8));
            imin = _mm512_min_epi16(imin, _mm512_alignr_epi8(imin, imin, 4));
            imin = _mm512_min_epi16(imin, _mm512_alignr_epi8(imin, imin, 2));
            let min_index: usize = _mm_extract_epi16(_mm512_castsi512_si128(imin), 0) as usize;

            (min_index, _i16ord_to_f16(min_value))
        }

        #[inline(always)]
        unsafe fn _horiz_max(index: __m512i, value: __m512i) -> (usize, f16) {
            // 0. Find the maximum value
            let mut vmax: __m512i = value;
            vmax = _mm512_max_epi16(vmax, _mm512_alignr_epi32(vmax, vmax, 8));
            vmax = _mm512_max_epi16(vmax, _mm512_alignr_epi32(vmax, vmax, 4));
            vmax = _mm512_max_epi16(vmax, _mm512_alignr_epi8(vmax, vmax, 8));
            vmax = _mm512_max_epi16(vmax, _mm512_alignr_epi8(vmax, vmax, 4));
            vmax = _mm512_max_epi16(vmax, _mm512_alignr_epi8(vmax, vmax, 2));
            let max_value: i16 = _mm_extract_epi16(_mm512_castsi512_si128(vmax), 0) as i16;

            // Extract the index of the maximum value
            // 1. Create a mask with the index of the maximum value
            let mask = _mm512_cmpeq_epi16_mask(value, vmax);
            // 2. Blend the mask with the index
            let search_index = _mm512_mask_blend_epi16(
                mask,
                _mm512_set1_epi16(i16::MAX), // if mask is 0, use i16::MAX
                index,                       // if mask is 1, use index
            );
            // 3. Find the maximum index
            let mut imin: __m512i = search_index;
            imin = _mm512_min_epi16(imin, _mm512_alignr_epi32(imin, imin, 8));
            imin = _mm512_min_epi16(imin, _mm512_alignr_epi32(imin, imin, 4));
            imin = _mm512_min_epi16(imin, _mm512_alignr_epi8(imin, imin, 8));
            imin = _mm512_min_epi16(imin, _mm512_alignr_epi8(imin, imin, 4));
            imin = _mm512_min_epi16(imin, _mm512_alignr_epi8(imin, imin, 2));
            let max_index: usize = _mm_extract_epi16(_mm512_castsi512_si128(imin), 0) as usize;

            (max_index, _i16ord_to_f16(max_value))
        }
    }

    impl SIMDSetOps<f16, __m512i> for AVX512IgnoreNaN {
        #[inline(always)]
        unsafe fn _mm_set1(a: f16) -> __m512i {
            _mm512_set1_epi16(a.to_bits() as i16)
        }
    }

    impl SIMDArgMinMaxIgnoreNaN<f16, __m512i, u32, LANE_SIZE> for AVX512IgnoreNaN {
        #[target_feature(enable = "avx512bw")]
        unsafe fn argminmax(data: &[f16]) -> (usize, usize) {
            Self::_argminmax(data)
        }
    }

    // ----------------------------------------- TESTS -----------------------------------------

    #[cfg(test)]
    mod tests {
        use super::AVX512IgnoreNaN as AVX512;
        use super::SIMDArgMinMaxIgnoreNaN;
        use crate::scalar::generic::scalar_argminmax;

        use half::f16;

        extern crate dev_utils;
        use dev_utils::utils;

        fn get_array_f16(n: usize) -> Vec<f16> {
            let arr = utils::get_random_array(n, i16::MIN, i16::MAX);
            arr.iter().map(|x| f16::from_f32(*x as f32)).collect()
        }

        #[test]
        fn test_both_versions_return_the_same_results() {
            if !is_x86_feature_detected!("avx512bw") {
                return;
            }

            let data: &[f16] = &get_array_f16(1025);
            assert_eq!(data.len() % 8, 1);

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX512::argminmax(data) };
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }

        #[test]
        fn test_first_index_is_returned_when_identical_values_found() {
            if !is_x86_feature_detected!("avx512bw") {
                return;
            }

            let data = [
                f16::from_f32(10.),
                f16::MAX,
                f16::from_f32(6.),
                f16::NEG_INFINITY,
                f16::NEG_INFINITY,
                f16::MAX,
                f16::from_f32(5_000.0),
            ];
            let data: Vec<f16> = data.iter().map(|x| *x).collect();
            let data: &[f16] = &data;

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            assert_eq!(argmin_index, 3);
            assert_eq!(argmax_index, 1);

            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX512::argminmax(data) };
            assert_eq!(argmin_simd_index, 3);
            assert_eq!(argmax_simd_index, 1);
        }

        #[test]
        fn test_no_overflow() {
            if !is_x86_feature_detected!("avx512bw") {
                return;
            }

            let n: usize = 1 << 18;
            let data: &[f16] = &get_array_f16(n);

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX512::argminmax(data) };
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }

        #[test]
        fn test_many_random_runs() {
            if !is_x86_feature_detected!("avx512bw") {
                return;
            }

            for _ in 0..10_000 {
                let data: &[f16] = &get_array_f16(32 * 8 + 1);
                let (argmin_index, argmax_index) = scalar_argminmax(data);
                let (argmin_simd_index, argmax_simd_index) = unsafe { AVX512::argminmax(data) };
                assert_eq!(argmin_index, argmin_simd_index);
                assert_eq!(argmax_index, argmax_simd_index);
            }
        }
    }
}

// ---------------------------------------- NEON -----------------------------------------

#[cfg(feature = "half")]
#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
mod neon {
    use super::super::config::{NEONIgnoreNaN, NEON};
    use super::*;

    const LANE_SIZE: usize = NEON::LANE_SIZE_16;
    const LOWER_15_MASK: int16x8_t = unsafe { std::mem::transmute([MASK_VALUE; LANE_SIZE]) };

    #[inline(always)]
    unsafe fn _f16_as_int16x8_to_i16ord(f16_as_int16x8: int16x8_t) -> int16x8_t {
        // on a scalar: ((v >> 15) & 0x7FFF) ^ v
        let sign_bit_shifted = vshrq_n_s16(f16_as_int16x8, BIT_SHIFT);
        let sign_bit_masked = vandq_s16(sign_bit_shifted, LOWER_15_MASK);
        veorq_s16(f16_as_int16x8, sign_bit_masked)
    }

    #[inline(always)]
    unsafe fn _reg_to_i16_arr(reg: int16x8_t) -> [i16; LANE_SIZE] {
        std::mem::transmute::<int16x8_t, [i16; LANE_SIZE]>(reg)
    }

    impl SIMDOps<f16, int16x8_t, uint16x8_t, LANE_SIZE> for NEONIgnoreNaN {
        const INITIAL_INDEX: int16x8_t =
            unsafe { std::mem::transmute([0i16, 1i16, 2i16, 3i16, 4i16, 5i16, 6i16, 7i16]) };
        const INDEX_INCREMENT: int16x8_t =
            unsafe { std::mem::transmute([LANE_SIZE as i16; LANE_SIZE]) };
        const MAX_INDEX: usize = MAX_INDEX;

        #[inline(always)]
        unsafe fn _reg_to_arr(_: int16x8_t) -> [f16; LANE_SIZE] {
            // Not implemented because we will perform the horizontal operations on the
            // signed integer values instead of trying to retransform **only** the values
            // (and thus not the indices) to floats.
            unimplemented!()
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const f16) -> int16x8_t {
            _f16_as_int16x8_to_i16ord(vld1q_s16(unsafe {
                std::mem::transmute::<*const f16, *const i16>(data)
            }))
        }

        #[inline(always)]
        unsafe fn _mm_add(a: int16x8_t, b: int16x8_t) -> int16x8_t {
            vaddq_s16(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: int16x8_t, b: int16x8_t) -> uint16x8_t {
            vcgtq_s16(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: int16x8_t, b: int16x8_t) -> uint16x8_t {
            vcltq_s16(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: int16x8_t, b: int16x8_t, mask: uint16x8_t) -> int16x8_t {
            vbslq_s16(mask, b, a)
        }

        #[inline(always)]
        unsafe fn _horiz_min(index: int16x8_t, value: int16x8_t) -> (usize, f16) {
            // 0. Find the minimum value
            let mut vmin: int16x8_t = value;
            vmin = vminq_s16(vmin, vextq_s16(vmin, vmin, 4));
            vmin = vminq_s16(vmin, vextq_s16(vmin, vmin, 2));
            vmin = vminq_s16(vmin, vextq_s16(vmin, vmin, 1));
            let min_value: i16 = vgetq_lane_s16(vmin, 0);

            // Extract the index of the minimum value
            // 1. Create a mask with the index of the minimum value
            let mask = vceqq_s16(value, vmin);
            // 2. Blend the mask with the index
            let search_index = vbslq_s16(
                mask,
                index,                 // if mask is 1, use index
                vdupq_n_s16(i16::MAX), // if mask is 0, use i16::MAX
            );
            // 3. Find the minimum index
            let mut imin: int16x8_t = search_index;
            imin = vminq_s16(imin, vextq_s16(imin, imin, 4));
            imin = vminq_s16(imin, vextq_s16(imin, imin, 2));
            imin = vminq_s16(imin, vextq_s16(imin, imin, 1));
            let min_index: usize = vgetq_lane_s16(imin, 0) as usize;

            (min_index, _i16ord_to_f16(min_value))
        }

        #[inline(always)]
        unsafe fn _horiz_max(index: int16x8_t, value: int16x8_t) -> (usize, f16) {
            // 0. Find the maximum value
            let mut vmax: int16x8_t = value;
            vmax = vmaxq_s16(vmax, vextq_s16(vmax, vmax, 4));
            vmax = vmaxq_s16(vmax, vextq_s16(vmax, vmax, 2));
            vmax = vmaxq_s16(vmax, vextq_s16(vmax, vmax, 1));
            let max_value: i16 = vgetq_lane_s16(vmax, 0);

            // Extract the index of the maximum value
            // 1. Create a mask with the index of the maximum value
            let mask = vceqq_s16(value, vmax);
            // 2. Blend the mask with the index
            let search_index = vbslq_s16(
                mask,
                index,                 // if mask is 1, use index
                vdupq_n_s16(i16::MAX), // if mask is 0, use i16::MAX
            );
            // 3. Find the maximum index
            let mut imin: int16x8_t = search_index;
            imin = vminq_s16(imin, vextq_s16(imin, imin, 4));
            imin = vminq_s16(imin, vextq_s16(imin, imin, 2));
            imin = vminq_s16(imin, vextq_s16(imin, imin, 1));
            let max_index: usize = vgetq_lane_s16(imin, 0) as usize;

            (max_index, _i16ord_to_f16(max_value))
        }
    }

    impl SIMDSetOps<f16, int16x8_t> for NEONIgnoreNaN {
        #[inline(always)]
        unsafe fn _mm_set1(a: f16) -> int16x8_t {
            vdupq_n_s16(std::mem::transmute::<f16, i16>(a))
        }
    }

    impl SIMDArgMinMaxIgnoreNaN<f16, int16x8_t, uint16x8_t, LANE_SIZE> for NEONIgnoreNaN {
        #[target_feature(enable = "neon")]
        unsafe fn argminmax(data: &[f16]) -> (usize, usize) {
            Self::_argminmax(data)
        }
    }

    // ----------------------------------------- TESTS -----------------------------------------

    #[cfg(test)]
    mod tests {
        use super::NEONIgnoreNaN as NEON;
        use super::SIMDArgMinMaxIgnoreNaN;
        use crate::scalar::generic::scalar_argminmax;

        use half::f16;

        extern crate dev_utils;
        use dev_utils::utils;

        fn get_array_f16(n: usize) -> Vec<f16> {
            let arr = utils::get_random_array(n, i16::MIN, i16::MAX);
            arr.iter().map(|x| f16::from_f32(*x as f32)).collect()
        }

        #[test]
        fn test_both_versions_return_the_same_results() {
            let data: &[f16] = &get_array_f16(1025);
            assert_eq!(data.len() % 8, 1);

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            let (argmin_simd_index, argmax_simd_index) = unsafe { NEON::argminmax(data) };
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }

        #[test]
        fn test_first_index_is_returned_when_identical_values_found() {
            let data = [
                f16::from_f32(10.),
                f16::MAX,
                f16::from_f32(6.),
                f16::NEG_INFINITY,
                f16::NEG_INFINITY,
                f16::MAX,
                f16::from_f32(5_000.0),
            ];
            let data: Vec<f16> = data.iter().map(|x| *x).collect();
            let data: &[f16] = &data;

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            assert_eq!(argmin_index, 3);
            assert_eq!(argmax_index, 1);

            let (argmin_simd_index, argmax_simd_index) = unsafe { NEON::argminmax(data) };
            assert_eq!(argmin_simd_index, 3);
            assert_eq!(argmax_simd_index, 1);
        }

        #[test]
        fn test_no_overflow() {
            let n: usize = 1 << 18;
            let data: &[f16] = &get_array_f16(n);

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            let (argmin_simd_index, argmax_simd_index) = unsafe { NEON::argminmax(data) };
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }

        #[test]
        fn test_many_random_runs() {
            for _ in 0..10_000 {
                let data: &[f16] = &get_array_f16(32 * 8 + 1);
                let (argmin_index, argmax_index) = scalar_argminmax(data);
                let (argmin_simd_index, argmax_simd_index) = unsafe { NEON::argminmax(data) };
                assert_eq!(argmin_index, argmin_simd_index);
                assert_eq!(argmax_index, argmax_simd_index);
            }
        }
    }
}

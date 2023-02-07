use super::config::SIMDInstructionSet;
use super::generic::SIMD;
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "arm")]
use std::arch::arm::*;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::task::{max_index_value, min_index_value};

const XOR_VALUE: i32 = 0x7FFFFFFF;

fn _ord_i32_to_f32(ord_i32: i32) -> f32 {
    // TODO: more efficient transformation -> can be decreasing order as well
    let v = ((ord_i32 >> 31) & XOR_VALUE) ^ ord_i32;
    unsafe { std::mem::transmute::<i32, f32>(v) }
}

// ------------------------------------------ AVX2 ------------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx2 {
    use super::super::config::AVX2;
    use super::*;

    const LANE_SIZE: usize = AVX2::LANE_SIZE_32;
    const XOR_MASK: __m256i = unsafe { std::mem::transmute([XOR_VALUE; LANE_SIZE]) };
    const BIT_SHIFT: i32 = 31;

    #[inline(always)]
    unsafe fn _f32_as_m256i_to_i32ord(f32_as_m256i: __m256i) -> __m256i {
        // on a scalar: ((v >> 31) & 0x7FFFFFFF) ^ v
        let sign_bit_shifted = _mm256_srai_epi32(f32_as_m256i, BIT_SHIFT);
        let sign_bit_masked = _mm256_and_si256(sign_bit_shifted, XOR_MASK);
        _mm256_xor_epi32(sign_bit_masked, f32_as_m256i)
    }

    #[inline(always)]
    unsafe fn _reg_to_i32_arr(reg: __m256i) -> [i32; LANE_SIZE] {
        std::mem::transmute::<__m256i, [i32; LANE_SIZE]>(reg)
    }

    impl SIMD<f32, __m256i, __m256i, LANE_SIZE> for AVX2 {
        const INITIAL_INDEX: __m256i =
            unsafe { std::mem::transmute([0i32, 1i32, 2i32, 3i32, 4i32, 5i32, 6i32, 7i32]) };
        const MAX_INDEX: usize = i32::MAX as usize;

        #[inline(always)]
        unsafe fn _reg_to_arr(_: __m256i) -> [f32; LANE_SIZE] {
            unimplemented!()
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const f32) -> __m256i {
            _f32_as_m256i_to_i32ord(_mm256_loadu_epi32(data as *const i32))
        }

        #[inline(always)]
        unsafe fn _mm_set1(a: usize) -> __m256i {
            _mm256_set1_epi32(a as i32)
        }

        #[inline(always)]
        unsafe fn _mm_add(a: __m256i, b: __m256i) -> __m256i {
            _mm256_add_epi32(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: __m256i, b: __m256i) -> __m256i {
            _mm256_cmpgt_epi32(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: __m256i, b: __m256i) -> __m256i {
            _mm256_cmpgt_epi32(b, a)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: __m256i, b: __m256i, mask: __m256i) -> __m256i {
            _mm256_blendv_epi8(a, b, mask)
        }

        // ------------------------------------ ARGMINMAX --------------------------------------

        #[target_feature(enable = "avx2")]
        unsafe fn argminmax(data: &[f32]) -> (usize, usize) {
            Self::_argminmax(data)
        }

        #[inline(always)]
        unsafe fn _get_min_max_index_value(
            index_low: __m256i,
            values_low: __m256i,
            index_high: __m256i,
            values_high: __m256i,
        ) -> (usize, f32, usize, f32) {
            // Get the results as arrays
            let index_low_arr = _reg_to_i32_arr(index_low);
            let values_low_arr = _reg_to_i32_arr(values_low);
            let index_high_arr = _reg_to_i32_arr(index_high);
            let values_high_arr = _reg_to_i32_arr(values_high);
            // Find the min and max values and their indices
            let (min_index, min_value) = min_index_value(&index_low_arr, &values_low_arr);
            let (max_index, max_value) = max_index_value(&index_high_arr, &values_high_arr);
            // Return the results - convert the ordinal ints back to floats
            let min_value = _ord_i32_to_f32(min_value);
            let max_value = _ord_i32_to_f32(max_value);
            if min_value != min_value && max_value == max_value {
                // min_value is the only NaN
                return (min_index as usize, min_value, min_index as usize, min_value);
            } else if min_value == min_value && max_value != max_value {
                // max_value is the only NaN
                return (max_index as usize, max_value, max_index as usize, max_value);
            }
            (min_index as usize, min_value, max_index as usize, max_value)
        }
    }

    // ------------------------------------ TESTS --------------------------------------

    #[cfg(test)]
    mod tests {
        use super::{AVX2, SIMD};
        use crate::scalar::generic::scalar_argminmax;

        extern crate dev_utils;
        use dev_utils::utils;

        fn get_array_f32(n: usize) -> Vec<f32> {
            utils::get_random_array(n, f32::MIN, f32::MAX)
        }

        #[test]
        fn test_both_versions_return_the_same_results() {
            if !is_x86_feature_detected!("avx2") {
                return;
            }

            let data: &[f32] = &get_array_f32(1025);
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
                10.,
                std::f32::MAX,
                6.,
                std::f32::NEG_INFINITY,
                std::f32::NEG_INFINITY,
                std::f32::MAX,
                10_000.0,
            ];
            let data: Vec<f32> = data.iter().map(|x| *x).collect();
            let data: &[f32] = &data;

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            assert_eq!(argmin_index, 3);
            assert_eq!(argmax_index, 1);

            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX2::argminmax(data) };
            assert_eq!(argmin_simd_index, 3);
            assert_eq!(argmax_simd_index, 1);
        }

        #[test]
        fn test_return_nan_index() {
            if !is_x86_feature_detected!("avx2") {
                return;
            }

            // Case 1: NaN is the first element
            let mut data: Vec<f32> = get_array_f32(1027);
            data[0] = std::f32::NAN;

            let (argmin_index, argmax_index) = scalar_argminmax(&data);
            assert_eq!(argmin_index, 0);
            assert_eq!(argmax_index, 0);

            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX2::argminmax(&data) };
            assert_eq!(argmin_simd_index, 0);
            assert_eq!(argmax_simd_index, 0);

            // Case 2: NaN is the last element
            let mut data: Vec<f32> = get_array_f32(1027);
            data[1026] = std::f32::NAN;

            let (argmin_index, argmax_index) = scalar_argminmax(&data);
            assert_eq!(argmin_index, 1026);
            assert_eq!(argmax_index, 1026);

            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX2::argminmax(&data) };
            assert_eq!(argmin_simd_index, 1026);
            assert_eq!(argmax_simd_index, 1026);

            // Case 3: NaN is somewhere in the middle element
            let mut data: Vec<f32> = get_array_f32(1027);
            data[123] = std::f32::NAN;

            let (argmin_index, argmax_index) = scalar_argminmax(&data);
            assert_eq!(argmin_index, 123);
            assert_eq!(argmax_index, 123);

            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX2::argminmax(&data) };
            assert_eq!(argmin_simd_index, 123);
            assert_eq!(argmax_simd_index, 123);
        }

        #[test]
        fn test_no_overflow() {
            if !is_x86_feature_detected!("avx2") {
                return;
            }

            let n: usize = 1 << 25;
            let data: &[f32] = &get_array_f32(n);

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
                let data: &[f32] = &get_array_f32(32 * 8 + 1);
                let (argmin_index, argmax_index) = scalar_argminmax(data);
                let (argmin_simd_index, argmax_simd_index) = unsafe { AVX2::argminmax(data) };
                assert_eq!(argmin_index, argmin_simd_index);
                assert_eq!(argmax_index, argmax_simd_index);
            }
        }
    }
}

// ----------------------------------------- SSE -----------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod sse {
    use super::super::config::SSE;
    use super::*;

    const LANE_SIZE: usize = SSE::LANE_SIZE_32;
    const XOR_MASK: __m128i = unsafe { std::mem::transmute([XOR_VALUE; LANE_SIZE]) };
    const BIT_SHIFT: i32 = 31;

    #[inline(always)]
    unsafe fn _f32_as_m128i_to_i32ord(f32_as_m128i: __m128i) -> __m128i {
        // on a scalar: ((v >> 31) & 0x7FFFFFFF) ^ v
        let sign_bit_shifted = _mm_srai_epi32(f32_as_m128i, BIT_SHIFT);
        let sign_bit_masked = _mm_and_si128(sign_bit_shifted, XOR_MASK);
        _mm_xor_si128(sign_bit_masked, f32_as_m128i)
    }

    #[inline(always)]
    unsafe fn _reg_to_i32_arr(reg: __m128i) -> [i32; LANE_SIZE] {
        std::mem::transmute::<__m128i, [i32; LANE_SIZE]>(reg)
    }

    impl SIMD<f32, __m128i, __m128i, LANE_SIZE> for SSE {
        const INITIAL_INDEX: __m128i = unsafe { std::mem::transmute([0i32, 1i32, 2i32, 3i32]) };
        const MAX_INDEX: usize = i32::MAX as usize;

        #[inline(always)]
        unsafe fn _reg_to_arr(_: __m128i) -> [f32; LANE_SIZE] {
            unimplemented!()
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const f32) -> __m128i {
            _f32_as_m128i_to_i32ord(_mm_loadu_si128(data as *const __m128i))
        }

        #[inline(always)]
        unsafe fn _mm_set1(a: usize) -> __m128i {
            _mm_set1_epi32(a as i32)
        }

        #[inline(always)]
        unsafe fn _mm_add(a: __m128i, b: __m128i) -> __m128i {
            _mm_add_epi32(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: __m128i, b: __m128i) -> __m128i {
            _mm_cmpgt_epi32(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: __m128i, b: __m128i) -> __m128i {
            _mm_cmplt_epi32(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: __m128i, b: __m128i, mask: __m128i) -> __m128i {
            _mm_blendv_epi8(a, b, mask)
        }

        // ------------------------------------ ARGMINMAX --------------------------------------

        #[target_feature(enable = "sse4.1")]
        unsafe fn argminmax(data: &[f32]) -> (usize, usize) {
            Self::_argminmax(data)
        }

        #[inline(always)]
        unsafe fn _get_min_max_index_value(
            index_low: __m128i,
            values_low: __m128i,
            index_high: __m128i,
            values_high: __m128i,
        ) -> (usize, f32, usize, f32) {
            // Get the results as arrays
            let index_low_arr = _reg_to_i32_arr(index_low);
            let values_low_arr = _reg_to_i32_arr(values_low);
            let index_high_arr = _reg_to_i32_arr(index_high);
            let values_high_arr = _reg_to_i32_arr(values_high);
            // Find the min and max values and their indices
            let (min_index, min_value) = min_index_value(&index_low_arr, &values_low_arr);
            let (max_index, max_value) = max_index_value(&index_high_arr, &values_high_arr);
            // Return the results - convert the ordinal ints back to floats
            let min_value = _ord_i32_to_f32(min_value);
            let max_value = _ord_i32_to_f32(max_value);
            if min_value != min_value && max_value == max_value {
                // min_value is the only NaN
                return (min_index as usize, min_value, min_index as usize, min_value);
            } else if min_value == min_value && max_value != max_value {
                // max_value is the only NaN
                return (max_index as usize, max_value, max_index as usize, max_value);
            }
            (min_index as usize, min_value, max_index as usize, max_value)
        }
    }

    // ------------------------------------ TESTS --------------------------------------

    #[cfg(test)]
    mod tests {
        use super::{SIMD, SSE};
        use crate::scalar::generic::scalar_argminmax;

        extern crate dev_utils;
        use dev_utils::utils;

        fn get_array_f32(n: usize) -> Vec<f32> {
            utils::get_random_array(n, f32::MIN, f32::MAX)
        }

        #[test]
        fn test_both_versions_return_the_same_results() {
            let data: &[f32] = &get_array_f32(1025);
            assert_eq!(data.len() % 4, 1);

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            let (argmin_simd_index, argmax_simd_index) = unsafe { SSE::argminmax(data) };
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }

        #[test]
        fn test_first_index_is_returned_when_identical_values_found() {
            let data = [
                10.,
                std::f32::MAX,
                6.,
                std::f32::NEG_INFINITY,
                std::f32::NEG_INFINITY,
                std::f32::MAX,
                10_000.0,
            ];
            let data: Vec<f32> = data.iter().map(|x| *x).collect();
            let data: &[f32] = &data;

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            assert_eq!(argmin_index, 3);
            assert_eq!(argmax_index, 1);

            let (argmin_simd_index, argmax_simd_index) = unsafe { SSE::argminmax(data) };
            assert_eq!(argmin_simd_index, 3);
            assert_eq!(argmax_simd_index, 1);
        }

        #[test]
        fn test_return_nan_index() {
            // Case 1: NaN is the first element
            let mut data: Vec<f32> = get_array_f32(1027);
            data[0] = std::f32::NAN;

            let (argmin_index, argmax_index) = scalar_argminmax(&data);
            assert_eq!(argmin_index, 0);
            assert_eq!(argmax_index, 0);

            let (argmin_simd_index, argmax_simd_index) = unsafe { SSE::argminmax(&data) };
            assert_eq!(argmin_simd_index, 0);
            assert_eq!(argmax_simd_index, 0);

            // Case 2: NaN is the last element
            let mut data: Vec<f32> = get_array_f32(1027);
            data[1026] = std::f32::NAN;

            let (argmin_index, argmax_index) = scalar_argminmax(&data);
            assert_eq!(argmin_index, 1026);
            assert_eq!(argmax_index, 1026);

            let (argmin_simd_index, argmax_simd_index) = unsafe { SSE::argminmax(&data) };
            assert_eq!(argmin_simd_index, 1026);
            assert_eq!(argmax_simd_index, 1026);

            // Case 3: NaN is somewhere in the middle element
            let mut data: Vec<f32> = get_array_f32(1027);
            data[123] = std::f32::NAN;

            let (argmin_index, argmax_index) = scalar_argminmax(&data);
            assert_eq!(argmin_index, 123);
            assert_eq!(argmax_index, 123);

            let (argmin_simd_index, argmax_simd_index) = unsafe { SSE::argminmax(&data) };
            assert_eq!(argmin_simd_index, 123);
            assert_eq!(argmax_simd_index, 123);
        }

        #[test]
        fn test_no_overflow() {
            let n: usize = 1 << 25;
            let data: &[f32] = &get_array_f32(n);

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            let (argmin_simd_index, argmax_simd_index) = unsafe { SSE::argminmax(data) };
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }

        #[test]
        fn test_many_random_runs() {
            for _ in 0..10_000 {
                let data: &[f32] = &get_array_f32(32 * 4 + 1);
                let (argmin_index, argmax_index) = scalar_argminmax(data);
                let (argmin_simd_index, argmax_simd_index) = unsafe { SSE::argminmax(data) };
                assert_eq!(argmin_index, argmin_simd_index);
                assert_eq!(argmax_index, argmax_simd_index);
            }
        }
    }
}

// --------------------------------------- AVX512 ----------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx512 {
    use super::super::config::AVX512;
    use super::*;

    const LANE_SIZE: usize = AVX512::LANE_SIZE_32;

    impl SIMD<f32, __m512, u16, LANE_SIZE> for AVX512 {
        const INITIAL_INDEX: __m512 = unsafe {
            std::mem::transmute([
                0.0f32, 1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32, 7.0f32, 8.0f32, 9.0f32,
                10.0f32, 11.0f32, 12.0f32, 13.0f32, 14.0f32, 15.0f32,
            ])
        };
        // https://stackoverflow.com/a/3793950
        const MAX_INDEX: usize = 1 << f32::MANTISSA_DIGITS;

        #[inline(always)]
        unsafe fn _reg_to_arr(reg: __m512) -> [f32; LANE_SIZE] {
            std::mem::transmute::<__m512, [f32; LANE_SIZE]>(reg)
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const f32) -> __m512 {
            _mm512_loadu_ps(data as *const f32)
        }

        #[inline(always)]
        unsafe fn _mm_set1(a: usize) -> __m512 {
            _mm512_set1_ps(a as f32)
        }

        #[inline(always)]
        unsafe fn _mm_add(a: __m512, b: __m512) -> __m512 {
            _mm512_add_ps(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: __m512, b: __m512) -> u16 {
            _mm512_cmp_ps_mask(a, b, _CMP_GT_OQ)
        }
        // unimplemented!("AVX512 comparison instructions for ps output a u16 mask.")
        // let u16_mask = _mm512_cmp_ps_mask(a, b, _CMP_GT_OQ);
        // _mm512_mask_mov_ps(_mm512_setzero_ps(), u16_mask, _mm512_set1_ps(1.0))
        // }
        // { _mm512_cmp_ps_mask(a, b, _CMP_GT_OQ) }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: __m512, b: __m512) -> u16 {
            _mm512_cmp_ps_mask(a, b, _CMP_LT_OQ)
        }
        // unimplemented!("AVX512 comparison instructions for ps output a u16 mask.")
        // let u16_mask = _mm512_cmp_ps_mask(a, b, _CMP_LT_OQ);
        // _mm512_mask_mov_ps(_mm512_setzero_ps(), u16_mask, _mm512_set1_ps(1.0))
        // }
        // { _mm512_cmp_ps_mask(a, b, _CMP_LT_OQ) }

        #[inline(always)]
        unsafe fn _mm_blendv(a: __m512, b: __m512, mask: u16) -> __m512 {
            _mm512_mask_blend_ps(mask, a, b)
        }
        // unimplemented!("AVX512 blendv instructions for ps require a u16 mask.")
        // convert the mask to u16 by extracting the sign bit of each lane
        // let u16_mask = _mm512_castps_si512(mask);
        // _mm512_mask_mov_ps(a, u16_mask, b)
        // _mm512_mask_blend_ps(u16_mask, a, b)
        // _mm512_mask_mov_ps(a, _mm512_castps_si512(mask), b)
        // }

        // ------------------------------------ ARGMINMAX --------------------------------------

        #[target_feature(enable = "avx512f")]
        unsafe fn argminmax(data: &[f32]) -> (usize, usize) {
            Self::_argminmax(data)
        }
    }

    // ------------------------------------ TESTS --------------------------------------

    #[cfg(test)]
    mod tests {
        use super::{AVX512, SIMD};
        use crate::scalar::generic::scalar_argminmax;

        extern crate dev_utils;
        use dev_utils::utils;

        fn get_array_f32(n: usize) -> Vec<f32> {
            utils::get_random_array(n, f32::MIN, f32::MAX)
        }

        #[test]
        fn test_both_versions_return_the_same_results() {
            if !is_x86_feature_detected!("avx512f") {
                return;
            }

            let data: &[f32] = &get_array_f32(1025);
            assert_eq!(data.len() % 16, 1);

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX512::argminmax(data) };
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }

        #[test]
        fn test_first_index_is_returned_when_identical_values_found() {
            if !is_x86_feature_detected!("avx512f") {
                return;
            }

            let data = [
                10.,
                std::f32::MAX,
                6.,
                std::f32::NEG_INFINITY,
                std::f32::NEG_INFINITY,
                std::f32::MAX,
                10_000.0,
            ];
            let data: Vec<f32> = data.iter().map(|x| *x).collect();
            let data: &[f32] = &data;

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            assert_eq!(argmin_index, 3);
            assert_eq!(argmax_index, 1);

            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX512::argminmax(data) };
            assert_eq!(argmin_simd_index, 3);
            assert_eq!(argmax_simd_index, 1);
        }

        #[test]
        fn test_no_overflow() {
            if !is_x86_feature_detected!("avx512f") {
                return;
            }

            let n: usize = 1 << 25;
            let data: &[f32] = &get_array_f32(n);

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX512::argminmax(data) };
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }

        #[test]
        fn test_many_random_runs() {
            if !is_x86_feature_detected!("avx512f") {
                return;
            }

            for _ in 0..10_000 {
                let data: &[f32] = &get_array_f32(32 * 16 + 1);
                let (argmin_index, argmax_index) = scalar_argminmax(data);
                let (argmin_simd_index, argmax_simd_index) = unsafe { AVX512::argminmax(data) };
                assert_eq!(argmin_index, argmin_simd_index);
                assert_eq!(argmax_index, argmax_simd_index);
            }
        }
    }
}

// ---------------------------------------- NEON -----------------------------------------

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
mod neon {
    use super::super::config::NEON;
    use super::*;

    const LANE_SIZE: usize = NEON::LANE_SIZE_32;

    impl SIMD<f32, float32x4_t, uint32x4_t, LANE_SIZE> for NEON {
        const INITIAL_INDEX: float32x4_t =
            unsafe { std::mem::transmute([0.0f32, 1.0f32, 2.0f32, 3.0f32]) };

        #[inline(always)]
        unsafe fn _reg_to_arr(reg: float32x4_t) -> [f32; LANE_SIZE] {
            std::mem::transmute::<float32x4_t, [f32; LANE_SIZE]>(reg)
        }
        // https://stackoverflow.com/a/3793950
        const MAX_INDEX: usize = 1 << f32::MANTISSA_DIGITS;

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const f32) -> float32x4_t {
            vld1q_f32(data as *const f32)
        }

        #[inline(always)]
        unsafe fn _mm_set1(a: usize) -> float32x4_t {
            vdupq_n_f32(a as f32)
        }

        #[inline(always)]
        unsafe fn _mm_add(a: float32x4_t, b: float32x4_t) -> float32x4_t {
            vaddq_f32(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: float32x4_t, b: float32x4_t) -> uint32x4_t {
            vcgtq_f32(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: float32x4_t, b: float32x4_t) -> uint32x4_t {
            vcltq_f32(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: float32x4_t, b: float32x4_t, mask: uint32x4_t) -> float32x4_t {
            vbslq_f32(mask, b, a)
        }

        // ------------------------------------ ARGMINMAX --------------------------------------

        #[target_feature(enable = "neon")]
        unsafe fn argminmax(data: &[f32]) -> (usize, usize) {
            Self::_argminmax(data)
        }
    }

    // ------------------------------------ TESTS --------------------------------------

    #[cfg(test)]
    mod tests {
        use super::{NEON, SIMD};
        use crate::scalar::generic::scalar_argminmax;

        extern crate dev_utils;
        use dev_utils::utils;

        fn get_array_f32(n: usize) -> Vec<f32> {
            utils::get_random_array(n, f32::MIN, f32::MAX)
        }

        #[test]
        fn test_both_versions_return_the_same_results() {
            let data: &[f32] = &get_array_f32(1025);
            assert_eq!(data.len() % 4, 1);

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            let (argmin_simd_index, argmax_simd_index) = unsafe { NEON::argminmax(data) };
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }

        #[test]
        fn test_first_index_is_returned_when_identical_values_found() {
            let data = [
                10.,
                std::f32::MAX,
                6.,
                std::f32::NEG_INFINITY,
                std::f32::NEG_INFINITY,
                std::f32::MAX,
                10_000.0,
            ];
            let data: Vec<f32> = data.iter().map(|x| *x).collect();
            let data: &[f32] = &data;

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            assert_eq!(argmin_index, 3);
            assert_eq!(argmax_index, 1);

            let (argmin_simd_index, argmax_simd_index) = unsafe { NEON::argminmax(data) };
            assert_eq!(argmin_simd_index, 3);
            assert_eq!(argmax_simd_index, 1);
        }

        #[test]
        fn test_no_overflow() {
            let n: usize = 1 << 25;
            let data: &[f32] = &get_array_f32(n);

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            let (argmin_simd_index, argmax_simd_index) = unsafe { NEON::argminmax(data) };
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }

        #[test]
        fn test_many_random_runs() {
            for _ in 0..10_000 {
                let data: &[f32] = &get_array_f32(32 * 4 + 1);
                let (argmin_index, argmax_index) = scalar_argminmax(data);
                let (argmin_simd_index, argmax_simd_index) = unsafe { NEON::argminmax(data) };
                assert_eq!(argmin_index, argmin_simd_index);
                assert_eq!(argmax_index, argmax_simd_index);
            }
        }
    }
}

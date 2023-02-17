/// Implementation of the argminmax operations for f64 that ignores NaN values.
/// This implementation returns the index of the minimum and maximum values.
/// However, unexpected behavior may occur when there are
/// - *only* NaN values in the array
/// - *only* +/- infinity values in the array
/// - *only* NaN and +/- infinity values in the array
/// In these cases, the index of the first element is (most likely) returned.
///
/// NaN values are ignored and treated as if they are not present in the array.
/// To realize this we create an initial SIMD register with values +/- infinity.
/// As comparisons with NaN always return false, it is guaranteed that no NaN values
/// are added to the accumulating SIMD register.
///

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use super::config::SIMDInstructionSet;
use super::generic::{SIMDArgMinMaxIgnoreNaN, SIMDOps, SIMDSetOps};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// https://stackoverflow.com/a/3793950
#[cfg(target_arch = "x86_64")]
const MAX_INDEX: usize = 1 << f64::MANTISSA_DIGITS;
#[cfg(target_arch = "x86")] // https://stackoverflow.com/a/29592369
const MAX_INDEX: usize = u32::MAX as usize;

// ------------------------------------------ AVX2 ------------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx2_ignore_nan {
    use super::super::config::{AVX2IgnoreNaN, AVX2};
    use super::*;

    const LANE_SIZE: usize = AVX2::LANE_SIZE_64;

    impl SIMDOps<f64, __m256d, __m256d, LANE_SIZE> for AVX2IgnoreNaN {
        const INITIAL_INDEX: __m256d =
            unsafe { std::mem::transmute([0.0f64, 1.0f64, 2.0f64, 3.0f64]) };
        const INDEX_INCREMENT: __m256d =
            unsafe { std::mem::transmute([LANE_SIZE as f64; LANE_SIZE]) };
        const MAX_INDEX: usize = MAX_INDEX;

        #[inline(always)]
        unsafe fn _reg_to_arr(reg: __m256d) -> [f64; LANE_SIZE] {
            std::mem::transmute::<__m256d, [f64; LANE_SIZE]>(reg)
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const f64) -> __m256d {
            _mm256_loadu_pd(data as *const f64)
        }

        #[inline(always)]
        unsafe fn _mm_add(a: __m256d, b: __m256d) -> __m256d {
            _mm256_add_pd(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: __m256d, b: __m256d) -> __m256d {
            _mm256_cmp_pd(a, b, _CMP_GT_OQ)
        }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: __m256d, b: __m256d) -> __m256d {
            _mm256_cmp_pd(b, a, _CMP_GT_OQ)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: __m256d, b: __m256d, mask: __m256d) -> __m256d {
            _mm256_blendv_pd(a, b, mask)
        }
    }

    impl SIMDSetOps<f64, __m256d> for AVX2IgnoreNaN {
        #[inline(always)]
        unsafe fn _mm_set1(a: f64) -> __m256d {
            _mm256_set1_pd(a)
        }
    }

    impl SIMDArgMinMaxIgnoreNaN<f64, __m256d, __m256d, LANE_SIZE> for AVX2IgnoreNaN {
        #[target_feature(enable = "avx")]
        unsafe fn argminmax(data: &[f64]) -> (usize, usize) {
            Self::_argminmax(data)
        }
    }

    // ------------------------------------ TESTS --------------------------------------

    #[cfg(test)]
    mod tests {
        use super::AVX2IgnoreNaN as AVX2;
        use super::SIMDArgMinMaxIgnoreNaN;
        use crate::scalar::generic::scalar_argminmax_ignore_nans as scalar_argminmax;

        extern crate dev_utils;
        use dev_utils::utils;

        fn get_array_f64(n: usize) -> Vec<f64> {
            utils::get_random_array(n, f64::MIN, f64::MAX)
        }

        #[test]
        fn test_both_versions_return_the_same_results() {
            if !is_x86_feature_detected!("avx") {
                return;
            }

            let data: &[f64] = &get_array_f64(1025);
            assert_eq!(data.len() % 4, 1);

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX2::argminmax(data) };
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }

        #[test]
        fn test_first_index_is_returned_when_identical_values_found() {
            if !is_x86_feature_detected!("avx") {
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
            let data: Vec<f64> = data.iter().map(|x| *x).collect();
            let data: &[f64] = &data;

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            assert_eq!(argmin_index, 3);
            assert_eq!(argmax_index, 1);

            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX2::argminmax(data) };
            assert_eq!(argmin_simd_index, 3);
            assert_eq!(argmax_simd_index, 1);
        }

        #[test]
        fn test_return_infs() {
            if !is_x86_feature_detected!("avx") {
                return;
            }

            let arr_len: usize = 1027;
            let mut data: Vec<f64> = get_array_f64(arr_len);

            // Case 1: all elements are +inf
            for i in 0..data.len() {
                data[i] = f64::INFINITY;
            }

            let (argmin_index, argmax_index) = scalar_argminmax(&data);
            assert_eq!(argmin_index, 0);
            assert_eq!(argmax_index, 0);

            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX2::argminmax(&data) };
            assert_eq!(argmin_simd_index, 0);
            assert_eq!(argmax_simd_index, 0);

            // Case 2: all elements are -inf
            for i in 0..data.len() {
                data[i] = f64::NEG_INFINITY;
            }

            let (argmin_index, argmax_index) = scalar_argminmax(&data);
            assert_eq!(argmin_index, 0);
            assert_eq!(argmax_index, 0);

            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX2::argminmax(&data) };
            assert_eq!(argmin_simd_index, 0);
            assert_eq!(argmax_simd_index, 0);

            // Case 3: add some +inf and -inf in the middle
            let mut data: Vec<f64> = get_array_f64(arr_len);
            data[100] = f64::INFINITY;
            data[200] = f64::NEG_INFINITY;

            let (argmin_index, argmax_index) = scalar_argminmax(&data);
            assert_eq!(argmin_index, 200);
            assert_eq!(argmax_index, 100);

            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX2::argminmax(&data) };
            assert_eq!(argmin_simd_index, 200);
            assert_eq!(argmax_simd_index, 100);
        }

        #[test]
        fn test_ignore_nans() {
            if !is_x86_feature_detected!("avx") {
                return;
            }

            let arr_len: usize = 1027;

            // Case 1: NaN is the first element
            let mut data: Vec<f64> = get_array_f64(arr_len);
            data[0] = f64::NAN;

            let (argmin_index, argmax_index) = scalar_argminmax(&data);
            assert!(argmin_index != 0);
            assert!(argmax_index != 0);

            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX2::argminmax(&data) };
            assert!(argmin_simd_index != 0);
            assert!(argmax_simd_index != 0);

            // Case 2: first 100 elements are NaN
            for i in 0..100 {
                data[i] = f64::NAN;
            }

            let (argmin_index, argmax_index) = scalar_argminmax(&data);
            assert!(argmin_index > 99);
            assert!(argmax_index > 99);

            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX2::argminmax(&data) };
            assert!(argmin_simd_index > 99);
            assert!(argmax_simd_index > 99);

            // Case 3: NaN is the last element
            let mut data: Vec<f64> = get_array_f64(arr_len);
            data[arr_len - 1] = f64::NAN;

            let (argmin_index, argmax_index) = scalar_argminmax(&data);
            assert!(argmin_index != 1026);
            assert!(argmax_index != 1026);

            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX2::argminmax(&data) };
            assert!(argmin_simd_index != 1026);
            assert!(argmax_simd_index != 1026);

            // Case 4: last 100 elements are NaN
            for i in 0..100 {
                data[arr_len - 1 - i] = f64::NAN;
            }

            let (argmin_index, argmax_index) = scalar_argminmax(&data);
            assert!(argmin_index < arr_len - 100);
            assert!(argmax_index < arr_len - 100);

            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX2::argminmax(&data) };
            assert!(argmin_simd_index < arr_len - 100);
            assert!(argmax_simd_index < arr_len - 100);

            // Case 5: NaN is somewhere in the middle element
            let mut data: Vec<f64> = get_array_f64(arr_len);
            data[123] = f64::NAN;

            let (argmin_index, argmax_index) = scalar_argminmax(&data);
            assert!(argmin_index != 123);
            assert!(argmax_index != 123);

            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX2::argminmax(&data) };
            assert!(argmin_simd_index != 123);
            assert!(argmax_simd_index != 123);

            // Case 6: all elements are NaN
            for i in 0..data.len() {
                data[i] = f64::NAN;
            }

            let (argmin_index, argmax_index) = scalar_argminmax(&data);
            assert_eq!(argmin_index, 0);
            assert_eq!(argmax_index, 0);

            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX2::argminmax(&data) };
            assert_eq!(argmin_simd_index, 0);
            assert_eq!(argmax_simd_index, 0);
        }

        #[test]
        fn test_many_random_runs() {
            if !is_x86_feature_detected!("avx") {
                return;
            }

            for _ in 0..10_000 {
                let data: &[f64] = &get_array_f64(32 * 8 + 1);
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
mod sse_ignore_nan {
    use super::super::config::{SSEIgnoreNaN, SSE};
    use super::*;

    const LANE_SIZE: usize = SSE::LANE_SIZE_64;

    impl SIMDOps<f64, __m128d, __m128d, LANE_SIZE> for SSEIgnoreNaN {
        const INITIAL_INDEX: __m128d = unsafe { std::mem::transmute([0.0f64, 1.0f64]) };
        const INDEX_INCREMENT: __m128d =
            unsafe { std::mem::transmute([LANE_SIZE as f64; LANE_SIZE]) };
        const MAX_INDEX: usize = MAX_INDEX;

        #[inline(always)]
        unsafe fn _reg_to_arr(reg: __m128d) -> [f64; LANE_SIZE] {
            std::mem::transmute::<__m128d, [f64; LANE_SIZE]>(reg)
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const f64) -> __m128d {
            _mm_loadu_pd(data as *const f64)
        }

        #[inline(always)]
        unsafe fn _mm_add(a: __m128d, b: __m128d) -> __m128d {
            _mm_add_pd(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: __m128d, b: __m128d) -> __m128d {
            _mm_cmpgt_pd(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: __m128d, b: __m128d) -> __m128d {
            _mm_cmplt_pd(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: __m128d, b: __m128d, mask: __m128d) -> __m128d {
            _mm_blendv_pd(a, b, mask)
        }
    }

    impl SIMDSetOps<f64, __m128d> for SSEIgnoreNaN {
        #[inline(always)]
        unsafe fn _mm_set1(a: f64) -> __m128d {
            _mm_set1_pd(a)
        }
    }

    impl SIMDArgMinMaxIgnoreNaN<f64, __m128d, __m128d, LANE_SIZE> for SSEIgnoreNaN {
        #[target_feature(enable = "sse4.1")] // TODO: check if this is correct
        unsafe fn argminmax(data: &[f64]) -> (usize, usize) {
            Self::_argminmax(data)
        }
    }

    // ------------------------------------ TESTS --------------------------------------

    #[cfg(test)]
    mod tests {
        use super::SIMDArgMinMaxIgnoreNaN;
        use super::SSEIgnoreNaN as SSE;
        use crate::scalar::generic::scalar_argminmax_ignore_nans as scalar_argminmax;

        extern crate dev_utils;
        use dev_utils::utils;

        fn get_array_f64(n: usize) -> Vec<f64> {
            utils::get_random_array(n, f64::MIN, f64::MAX)
        }

        #[test]
        fn test_both_versions_return_the_same_results() {
            let data: &[f64] = &get_array_f64(1025);
            assert_eq!(data.len() % 2, 1);

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            let (argmin_simd_index, argmax_simd_index) = unsafe { SSE::argminmax(data) };
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }

        #[test]
        fn test_first_index_is_returned_when_identical_values_found() {
            let data = [
                10.,
                f64::MAX,
                6.,
                f64::NEG_INFINITY,
                f64::NEG_INFINITY,
                f64::MAX,
                10_000.0,
            ];
            let data: Vec<f64> = data.iter().map(|x| *x).collect();
            let data: &[f64] = &data;

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            assert_eq!(argmin_index, 3);
            assert_eq!(argmax_index, 1);

            let (argmin_simd_index, argmax_simd_index) = unsafe { SSE::argminmax(data) };
            assert_eq!(argmin_simd_index, 3);
            assert_eq!(argmax_simd_index, 1);
        }

        #[test]
        fn test_return_infs() {
            let arr_len: usize = 1027;
            let mut data: Vec<f64> = get_array_f64(arr_len);

            // Case 1: all elements are +inf
            for i in 0..data.len() {
                data[i] = f64::INFINITY;
            }

            let (argmin_index, argmax_index) = scalar_argminmax(&data);
            assert_eq!(argmin_index, 0);
            assert_eq!(argmax_index, 0);

            let (argmin_simd_index, argmax_simd_index) = unsafe { SSE::argminmax(&data) };
            assert_eq!(argmin_simd_index, 0);
            assert_eq!(argmax_simd_index, 0);

            // Case 2: all elements are -inf
            for i in 0..data.len() {
                data[i] = f64::NEG_INFINITY;
            }

            let (argmin_index, argmax_index) = scalar_argminmax(&data);
            assert_eq!(argmin_index, 0);
            assert_eq!(argmax_index, 0);

            let (argmin_simd_index, argmax_simd_index) = unsafe { SSE::argminmax(&data) };
            assert_eq!(argmin_simd_index, 0);
            assert_eq!(argmax_simd_index, 0);

            // Case 3: add some +inf and -inf in the middle
            let mut data: Vec<f64> = get_array_f64(arr_len);
            data[100] = f64::INFINITY;
            data[200] = f64::NEG_INFINITY;

            let (argmin_index, argmax_index) = scalar_argminmax(&data);
            assert_eq!(argmin_index, 200);
            assert_eq!(argmax_index, 100);

            let (argmin_simd_index, argmax_simd_index) = unsafe { SSE::argminmax(&data) };
            assert_eq!(argmin_simd_index, 200);
            assert_eq!(argmax_simd_index, 100);
        }

        #[test]
        fn test_ignore_nans() {
            let arr_len: usize = 1027;

            // Case 1: NaN is the first element
            let mut data: Vec<f64> = get_array_f64(arr_len);
            data[0] = f64::NAN;

            let (argmin_index, argmax_index) = scalar_argminmax(&data);
            assert!(argmin_index != 0);
            assert!(argmax_index != 0);

            let (argmin_simd_index, argmax_simd_index) = unsafe { SSE::argminmax(&data) };
            assert!(argmin_simd_index != 0);
            assert!(argmax_simd_index != 0);

            // Case 2: first 100 elements are NaN
            for i in 0..100 {
                data[i] = f64::NAN;
            }

            let (argmin_index, argmax_index) = scalar_argminmax(&data);
            assert!(argmin_index > 99);
            assert!(argmax_index > 99);

            let (argmin_simd_index, argmax_simd_index) = unsafe { SSE::argminmax(&data) };
            assert!(argmin_simd_index > 99);
            assert!(argmax_simd_index > 99);

            // Case 3: NaN is the last element
            let mut data: Vec<f64> = get_array_f64(arr_len);
            data[arr_len - 1] = f64::NAN;

            let (argmin_index, argmax_index) = scalar_argminmax(&data);
            assert!(argmin_index != 1026);
            assert!(argmax_index != 1026);

            let (argmin_simd_index, argmax_simd_index) = unsafe { SSE::argminmax(&data) };
            assert!(argmin_simd_index != 1026);
            assert!(argmax_simd_index != 1026);

            // Case 4: last 100 elements are NaN
            for i in 0..100 {
                data[arr_len - 1 - i] = f64::NAN;
            }

            let (argmin_index, argmax_index) = scalar_argminmax(&data);
            assert!(argmin_index < arr_len - 100);
            assert!(argmax_index < arr_len - 100);

            let (argmin_simd_index, argmax_simd_index) = unsafe { SSE::argminmax(&data) };
            assert!(argmin_simd_index < arr_len - 100);
            assert!(argmax_simd_index < arr_len - 100);

            // Case 5: NaN is somewhere in the middle element
            let mut data: Vec<f64> = get_array_f64(arr_len);
            data[123] = f64::NAN;

            let (argmin_index, argmax_index) = scalar_argminmax(&data);
            assert!(argmin_index != 123);
            assert!(argmax_index != 123);

            let (argmin_simd_index, argmax_simd_index) = unsafe { SSE::argminmax(&data) };
            assert!(argmin_simd_index != 123);
            assert!(argmax_simd_index != 123);

            // Case 6: all elements are NaN
            for i in 0..data.len() {
                data[i] = f64::NAN;
            }

            let (argmin_index, argmax_index) = scalar_argminmax(&data);
            assert_eq!(argmin_index, 0);
            assert_eq!(argmax_index, 0);

            let (argmin_simd_index, argmax_simd_index) = unsafe { SSE::argminmax(&data) };
            assert_eq!(argmin_simd_index, 0);
            assert_eq!(argmax_simd_index, 0);
        }

        #[test]
        fn test_many_random_runs() {
            for _ in 0..10_000 {
                let data: &[f64] = &get_array_f64(32 * 2 + 1);
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
mod avx512_ignore_nan {
    use super::super::config::{AVX512IgnoreNaN, AVX512};
    use super::*;

    const LANE_SIZE: usize = AVX512::LANE_SIZE_64;

    impl SIMDOps<f64, __m512d, u8, LANE_SIZE> for AVX512IgnoreNaN {
        const INITIAL_INDEX: __m512d = unsafe {
            std::mem::transmute([
                0.0f64, 1.0f64, 2.0f64, 3.0f64, 4.0f64, 5.0f64, 6.0f64, 7.0f64,
            ])
        };
        const INDEX_INCREMENT: __m512d =
            unsafe { std::mem::transmute([LANE_SIZE as f64; LANE_SIZE]) };
        const MAX_INDEX: usize = MAX_INDEX;

        #[inline(always)]
        unsafe fn _reg_to_arr(reg: __m512d) -> [f64; LANE_SIZE] {
            std::mem::transmute::<__m512d, [f64; LANE_SIZE]>(reg)
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const f64) -> __m512d {
            _mm512_loadu_pd(data as *const f64)
        }

        #[inline(always)]
        unsafe fn _mm_add(a: __m512d, b: __m512d) -> __m512d {
            _mm512_add_pd(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: __m512d, b: __m512d) -> u8 {
            _mm512_cmp_pd_mask(a, b, _CMP_GT_OQ)
        }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: __m512d, b: __m512d) -> u8 {
            _mm512_cmp_pd_mask(a, b, _CMP_LT_OQ)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: __m512d, b: __m512d, mask: u8) -> __m512d {
            _mm512_mask_blend_pd(mask, a, b)
        }
    }

    impl SIMDSetOps<f64, __m512d> for AVX512IgnoreNaN {
        #[inline(always)]
        unsafe fn _mm_set1(a: f64) -> __m512d {
            _mm512_set1_pd(a)
        }
    }

    impl SIMDArgMinMaxIgnoreNaN<f64, __m512d, u8, LANE_SIZE> for AVX512IgnoreNaN {
        #[target_feature(enable = "avx512f")]
        unsafe fn argminmax(data: &[f64]) -> (usize, usize) {
            Self::_argminmax(data)
        }
    }

    // ------------------------------------ TESTS --------------------------------------

    #[cfg(test)]
    mod tests {
        use super::AVX512IgnoreNaN as AVX512;
        use super::SIMDArgMinMaxIgnoreNaN;
        use crate::scalar::generic::scalar_argminmax_ignore_nans as scalar_argminmax;

        extern crate dev_utils;
        use dev_utils::utils;

        fn get_array_f64(n: usize) -> Vec<f64> {
            utils::get_random_array(n, f64::MIN, f64::MAX)
        }

        #[test]
        fn test_both_versions_return_the_same_results() {
            if !is_x86_feature_detected!("avx512f") {
                return;
            }

            let data: &[f64] = &get_array_f64(1025);
            assert_eq!(data.len() % 2, 1);

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
                f64::MAX,
                6.,
                f64::NEG_INFINITY,
                f64::NEG_INFINITY,
                f64::MAX,
                10_000.0,
            ];
            let data: Vec<f64> = data.iter().map(|x| *x).collect();
            let data: &[f64] = &data;

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            assert_eq!(argmin_index, 3);
            assert_eq!(argmax_index, 1);

            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX512::argminmax(data) };
            assert_eq!(argmin_simd_index, 3);
            assert_eq!(argmax_simd_index, 1);
        }

        #[test]
        fn test_return_infs() {
            if !is_x86_feature_detected!("avx512f") {
                return;
            }

            let arr_len: usize = 1027;
            let mut data: Vec<f64> = get_array_f64(arr_len);

            // Case 1: all elements are +inf
            for i in 0..data.len() {
                data[i] = f64::INFINITY;
            }

            let (argmin_index, argmax_index) = scalar_argminmax(&data);
            assert_eq!(argmin_index, 0);
            assert_eq!(argmax_index, 0);

            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX512::argminmax(&data) };
            assert_eq!(argmin_simd_index, 0);
            assert_eq!(argmax_simd_index, 0);

            // Case 2: all elements are -inf
            for i in 0..data.len() {
                data[i] = f64::NEG_INFINITY;
            }

            let (argmin_index, argmax_index) = scalar_argminmax(&data);
            assert_eq!(argmin_index, 0);
            assert_eq!(argmax_index, 0);

            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX512::argminmax(&data) };
            assert_eq!(argmin_simd_index, 0);
            assert_eq!(argmax_simd_index, 0);

            // Case 3: add some +inf and -inf in the middle
            let mut data: Vec<f64> = get_array_f64(arr_len);
            data[100] = f64::INFINITY;
            data[200] = f64::NEG_INFINITY;

            let (argmin_index, argmax_index) = scalar_argminmax(&data);
            assert_eq!(argmin_index, 200);
            assert_eq!(argmax_index, 100);

            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX512::argminmax(&data) };
            assert_eq!(argmin_simd_index, 200);
            assert_eq!(argmax_simd_index, 100);
        }

        #[test]
        fn test_ignore_nans() {
            if !is_x86_feature_detected!("avx512f") {
                return;
            }

            let arr_len: usize = 1027;

            // Case 1: NaN is the first element
            let mut data: Vec<f64> = get_array_f64(arr_len);
            data[0] = f64::NAN;

            let (argmin_index, argmax_index) = scalar_argminmax(&data);
            assert!(argmin_index != 0);
            assert!(argmax_index != 0);

            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX512::argminmax(&data) };
            assert!(argmin_simd_index != 0);
            assert!(argmax_simd_index != 0);

            // Case 2: first 100 elements are NaN
            for i in 0..100 {
                data[i] = f64::NAN;
            }

            let (argmin_index, argmax_index) = scalar_argminmax(&data);
            assert!(argmin_index > 99);
            assert!(argmax_index > 99);

            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX512::argminmax(&data) };
            assert!(argmin_simd_index > 99);
            assert!(argmax_simd_index > 99);

            // Case 3: NaN is the last element
            let mut data: Vec<f64> = get_array_f64(arr_len);
            data[arr_len - 1] = f64::NAN;

            let (argmin_index, argmax_index) = scalar_argminmax(&data);
            assert!(argmin_index != 1026);
            assert!(argmax_index != 1026);

            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX512::argminmax(&data) };
            assert!(argmin_simd_index != 1026);
            assert!(argmax_simd_index != 1026);

            // Case 4: last 100 elements are NaN
            for i in 0..100 {
                data[arr_len - 1 - i] = f64::NAN;
            }

            let (argmin_index, argmax_index) = scalar_argminmax(&data);
            assert!(argmin_index < arr_len - 100);
            assert!(argmax_index < arr_len - 100);

            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX512::argminmax(&data) };
            assert!(argmin_simd_index < arr_len - 100);
            assert!(argmax_simd_index < arr_len - 100);

            // Case 5: NaN is somewhere in the middle element
            let mut data: Vec<f64> = get_array_f64(arr_len);
            data[123] = f64::NAN;

            let (argmin_index, argmax_index) = scalar_argminmax(&data);
            assert!(argmin_index != 123);
            assert!(argmax_index != 123);

            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX512::argminmax(&data) };
            assert!(argmin_simd_index != 123);
            assert!(argmax_simd_index != 123);

            // Case 6: all elements are NaN
            for i in 0..data.len() {
                data[i] = f64::NAN;
            }

            let (argmin_index, argmax_index) = scalar_argminmax(&data);
            assert_eq!(argmin_index, 0);
            assert_eq!(argmax_index, 0);

            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX512::argminmax(&data) };
            assert_eq!(argmin_simd_index, 0);
            assert_eq!(argmax_simd_index, 0);
        }

        #[test]
        fn test_many_random_runs() {
            if !is_x86_feature_detected!("avx512f") {
                return;
            }

            for _ in 0..10_000 {
                let data: &[f64] = &get_array_f64(32 * 2 + 1);
                let (argmin_index, argmax_index) = scalar_argminmax(data);
                let (argmin_simd_index, argmax_simd_index) = unsafe { AVX512::argminmax(data) };
                assert_eq!(argmin_index, argmin_simd_index);
                assert_eq!(argmax_index, argmax_simd_index);
            }
        }
    }
}

// ---------------------------------------- NEON -----------------------------------------

// There are no NEON intrinsics for f64, so we need to use the scalar version.
//   although NEON intrinsics exist for i64 and u64, we cannot use them as
//   they there is no 64-bit variant (of any data type) for the following three
//   intrinsics: vadd_, vcgt_, vclt_

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
mod neon_ignore_nan {
    use super::super::config::NEONIgnoreNaN;
    use super::super::generic::{unimpl_SIMDArgMinMaxIgnoreNaN, unimpl_SIMDOps};
    use super::*;

    // We need to (un)implement the SIMD trait for the NEON struct as otherwise the
    // compiler will complain that the trait is not implemented for the struct -
    // even though we are not using the trait for the NEON struct when dealing with
    // > 64 bit data types.
    unimpl_SIMDOps!(f64, usize, NEONIgnoreNaN);
    unimpl_SIMDArgMinMaxIgnoreNaN!(f64, usize, NEONIgnoreNaN);
}

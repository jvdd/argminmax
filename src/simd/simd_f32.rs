use super::config::SIMDInstructionSet;
use super::generic::SIMD;
use std::arch::x86_64::*;

// ------------------------------------------ AVX2 ------------------------------------------

use super::config::AVX2;

mod avx2 {
    use super::*;

    const LANE_SIZE: usize = AVX2::LANE_SIZE_32;

    impl SIMD<f32, __m256, LANE_SIZE> for AVX2 {

        const INITIAL_INDEX: __m256 = unsafe { std::mem::transmute([0.0f32, 1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32, 7.0f32]) };

        #[inline(always)]
        unsafe fn _reg_to_arr(reg: __m256) -> [f32; LANE_SIZE] {
            std::mem::transmute::<__m256, [f32; LANE_SIZE]>(reg)
        }

        #[inline(always)]
        unsafe fn _mm_load(data: *const f32) -> __m256 { _mm256_loadu_ps(data as *const f32) }

        #[inline(always)]
        unsafe fn _mm_set1(a: usize) -> __m256 { _mm256_set1_ps(a as f32) }

        #[inline(always)]
        unsafe fn _mm_add(a: __m256, b: __m256) -> __m256 { _mm256_add_ps(a, b) }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: __m256, b: __m256) -> __m256 { _mm256_cmp_ps(a, b, _CMP_GT_OQ) }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: __m256, b: __m256) -> __m256 { _mm256_cmp_ps(b, a, _CMP_GT_OQ) }

        #[inline(always)]
        unsafe fn _mm_blendv(a: __m256, b: __m256, mask: __m256) -> __m256 { _mm256_blendv_ps(a, b, mask) }

        // ------------------------------------ ARGMINMAX --------------------------------------

        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn argminmax(data: ndarray::ArrayView1<f32>) -> (usize, usize) {
            Self::_argminmax(data)
        }
    }

    // ------------------------------------ TESTS --------------------------------------

    #[cfg(test)]
    mod tests {
        use super::{AVX2, SIMD};
        use crate::scalar::scalar_generic::scalar_argminmax;

        use ndarray::Array1;

        extern crate dev_utils;
        use dev_utils::utils;

        fn get_array_f32(n: usize) -> Array1<f32> {
            utils::get_random_array(n, f32::MIN, f32::MAX)
        }

        #[test]
        fn test_both_versions_return_the_same_results() {
            let data = get_array_f32(1025);
            assert_eq!(data.len() % 8, 1);

            let (argmin_index, argmax_index) = scalar_argminmax(data.view());
            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX2::argminmax(data.view()) };
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
            let data = Array1::from(data);

            let (argmin_index, argmax_index) = scalar_argminmax(data.view());
            assert_eq!(argmin_index, 3);
            assert_eq!(argmax_index, 1);

            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX2::argminmax(data.view()) };
            assert_eq!(argmin_simd_index, 3);
            assert_eq!(argmax_simd_index, 1);
        }

        #[test]
        fn test_many_random_runs() {
            for _ in 0..10_000 {
                let data = get_array_f32(32 * 8 + 1);
                let (argmin_index, argmax_index) = scalar_argminmax(data.view());
                let (argmin_simd_index, argmax_simd_index) = unsafe { AVX2::argminmax(data.view()) };
                assert_eq!(argmin_index, argmin_simd_index);
                assert_eq!(argmax_index, argmax_simd_index);
            }
        }
    }

}

// ----------------------------------------- SSE -----------------------------------------

use super::config::SSE;

mod sse {
    use super::*;

    const LANE_SIZE: usize = SSE::LANE_SIZE_32;

    impl SIMD<f32, __m128, LANE_SIZE> for SSE {

        const INITIAL_INDEX: __m128 = unsafe { std::mem::transmute([0.0f32, 1.0f32, 2.0f32, 3.0f32]) };

        #[inline(always)]
        unsafe fn _reg_to_arr(reg: __m128) -> [f32; LANE_SIZE] {
            std::mem::transmute::<__m128, [f32; LANE_SIZE]>(reg)
        }

        #[inline(always)]
        unsafe fn _mm_load(data: *const f32) -> __m128 { _mm_loadu_ps(data as *const f32) }

        #[inline(always)]
        unsafe fn _mm_set1(a: usize) -> __m128 { _mm_set1_ps(a as f32) }

        #[inline(always)]
        unsafe fn _mm_add(a: __m128, b: __m128) -> __m128 { _mm_add_ps(a, b) }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: __m128, b: __m128) -> __m128 { _mm_cmpgt_ps(a, b) }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: __m128, b: __m128) -> __m128 { _mm_cmplt_ps(a, b) }

        #[inline(always)]
        unsafe fn _mm_blendv(a: __m128, b: __m128, mask: __m128) -> __m128 { _mm_blendv_ps(a, b, mask) }

        // ------------------------------------ ARGMINMAX --------------------------------------

        #[inline]
        #[target_feature(enable = "sse4.1")]
        unsafe fn argminmax(data: ndarray::ArrayView1<f32>) -> (usize, usize) {
            Self::_argminmax(data)
        }
    }

    // ------------------------------------ TESTS --------------------------------------

    #[cfg(test)]
    mod tests {
        use super::{SSE, SIMD};
        use crate::scalar::scalar_generic::scalar_argminmax;

        use ndarray::Array1;

        extern crate dev_utils;
        use dev_utils::utils;

        fn get_array_f32(n: usize) -> Array1<f32> {
            utils::get_random_array(n, f32::MIN, f32::MAX)
        }

        #[test]
        fn test_both_versions_return_the_same_results() {
            let data = get_array_f32(1025);
            assert_eq!(data.len() % 4, 1);

            let (argmin_index, argmax_index) = scalar_argminmax(data.view());
            let (argmin_simd_index, argmax_simd_index) = unsafe { SSE::argminmax(data.view()) };
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
            let data = Array1::from(data);

            let (argmin_index, argmax_index) = scalar_argminmax(data.view());
            assert_eq!(argmin_index, 3);
            assert_eq!(argmax_index, 1);

            let (argmin_simd_index, argmax_simd_index) = unsafe { SSE::argminmax(data.view()) };
            assert_eq!(argmin_simd_index, 3);
            assert_eq!(argmax_simd_index, 1);
        }

        #[test]
        fn test_many_random_runs() {
            for _ in 0..10_000 {
                let data = get_array_f32(32 * 4 + 1);
                let (argmin_index, argmax_index) = scalar_argminmax(data.view());
                let (argmin_simd_index, argmax_simd_index) = unsafe { SSE::argminmax(data.view()) };
                assert_eq!(argmin_index, argmin_simd_index);
                assert_eq!(argmax_index, argmax_simd_index);
            }
        }
    }
}

// --------------------------------------- AVX512 ----------------------------------------

use super::config::AVX512;

mod avx512 {}

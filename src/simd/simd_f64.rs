use super::config::SIMDInstructionSet;
use super::generic::SIMD;
use std::arch::x86_64::*;

// ------------------------------------------ AVX2 ------------------------------------------

use super::config::AVX2;

mod avx2 {
    use super::*;

    const LANE_SIZE: usize = AVX2::LANE_SIZE_64;

    impl SIMD<f64, __m256d, LANE_SIZE> for AVX2 {
        const INITIAL_INDEX: __m256d =
            unsafe { std::mem::transmute([0.0f64, 1.0f64, 2.0f64, 3.0f64]) };

        #[inline(always)]
        unsafe fn _reg_to_arr(reg: __m256d) -> [f64; LANE_SIZE] {
            std::mem::transmute::<__m256d, [f64; LANE_SIZE]>(reg)
        }

        #[inline(always)]
        unsafe fn _mm_load(data: *const f64) -> __m256d {
            _mm256_loadu_pd(data as *const f64)
        }

        #[inline(always)]
        unsafe fn _mm_set1(a: usize) -> __m256d {
            _mm256_set1_pd(a as f64)
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

        // ------------------------------------ ARGMINMAX --------------------------------------

        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn argminmax(data: ndarray::ArrayView1<f64>) -> (usize, usize) {
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

        fn get_array_f64(n: usize) -> Array1<f64> {
            utils::get_random_array(n, f64::MIN, f64::MAX)
        }

        #[test]
        fn test_both_versions_return_the_same_results() {
            let data = get_array_f64(1025);
            assert_eq!(data.len() % 4, 1);

            let (argmin_index, argmax_index) = scalar_argminmax(data.view());
            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX2::argminmax(data.view()) };
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }

        #[test]
        fn test_first_index_is_returned_when_identical_values_found() {
            let data = [
                10.,
                std::f64::MAX,
                6.,
                std::f64::NEG_INFINITY,
                std::f64::NEG_INFINITY,
                std::f64::MAX,
                10_000.0,
            ];
            let data: Vec<f64> = data.iter().map(|x| *x).collect();
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
                let data = get_array_f64(32 * 8 + 1);
                let (argmin_index, argmax_index) = scalar_argminmax(data.view());
                let (argmin_simd_index, argmax_simd_index) =
                    unsafe { AVX2::argminmax(data.view()) };
                assert_eq!(argmin_index, argmin_simd_index);
                assert_eq!(argmax_index, argmax_simd_index);
            }
        }
    }
}

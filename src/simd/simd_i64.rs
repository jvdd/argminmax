use super::config::SIMDInstructionSet;
use super::generic::SIMD;
use std::arch::x86_64::*;

// ------------------------------------------ AVX2 ------------------------------------------

use super::config::AVX2;

mod avx2 {
    use super::*;

    const LANE_SIZE: usize = AVX2::LANE_SIZE_64;

    impl SIMD<i64, __m256i, LANE_SIZE> for AVX2 {
        const INITIAL_INDEX: __m256i = unsafe { std::mem::transmute([0i64, 1i64, 2i64, 3i64]) };

        #[inline(always)]
        unsafe fn _reg_to_arr(reg: __m256i) -> [i64; LANE_SIZE] {
            std::mem::transmute::<__m256i, [i64; LANE_SIZE]>(reg)
        }

        #[inline(always)]
        unsafe fn _mm_load(data: *const i64) -> __m256i {
            _mm256_loadu_si256(data as *const __m256i)
        }

        #[inline(always)]
        unsafe fn _mm_set1(a: usize) -> __m256i {
            _mm256_set1_epi64x(a as i64)
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

        // ------------------------------------ ARGMINMAX --------------------------------------

        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn argminmax(data: ndarray::ArrayView1<i64>) -> (usize, usize) {
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

        fn get_array_i64(n: usize) -> Array1<i64> {
            utils::get_random_array(n, i64::MIN, i64::MAX)
        }

        #[test]
        fn test_both_versions_return_the_same_results() {
            let data = get_array_i64(1025);
            assert_eq!(data.len() % 4, 1);

            let (argmin_index, argmax_index) = scalar_argminmax(data.view());
            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX2::argminmax(data.view()) };
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }

        #[test]
        fn test_first_index_is_returned_when_identical_values_found() {
            let data = [
                std::i64::MIN,
                std::i64::MIN,
                4,
                6,
                9,
                std::i64::MAX,
                22,
                std::i64::MAX,
            ];
            let data: Vec<i64> = data.iter().map(|x| *x).collect();
            let data = Array1::from(data);

            let (argmin_index, argmax_index) = scalar_argminmax(data.view());
            assert_eq!(argmin_index, 0);
            assert_eq!(argmax_index, 5);

            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX2::argminmax(data.view()) };
            assert_eq!(argmin_simd_index, 0);
            assert_eq!(argmax_simd_index, 5);
        }

        #[test]
        fn test_many_random_runs() {
            for _ in 0..10_000 {
                let data = get_array_i64(32 * 8 + 1);
                let (argmin_index, argmax_index) = scalar_argminmax(data.view());
                let (argmin_simd_index, argmax_simd_index) =
                    unsafe { AVX2::argminmax(data.view()) };
                assert_eq!(argmin_index, argmin_simd_index);
                assert_eq!(argmax_index, argmax_simd_index);
            }
        }
    }
}

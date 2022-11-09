use super::config::{SIMDInstructionSet, AVX2};
use super::generic::SIMD;
use std::arch::x86_64::*;

const LANE_SIZE: usize = AVX2::LANE_SIZE_16;

impl SIMD<i16, __m256i, LANE_SIZE> for AVX2 {
    #[inline(always)]
    fn _initial_index() -> __m256i {
        unsafe { _mm256_set_epi16(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0) }
    }

    #[inline(always)]
    fn _reg_to_arr(reg: __m256i) -> [i16; LANE_SIZE] {
        unsafe { std::mem::transmute::<__m256i, [i16; LANE_SIZE]>(reg) }
    }

    #[inline(always)]
    fn _mm_load(data: *const i16) -> __m256i {
        unsafe { _mm256_loadu_si256(data as *const __m256i) }
    }

    #[inline(always)]
    fn _mm_set1(a: usize) -> __m256i {
        unsafe { _mm256_set1_epi16(a as i16) }
    }

    #[inline(always)]
    fn _mm_add(a: __m256i, b: __m256i) -> __m256i {
        unsafe { _mm256_add_epi16(a, b) }
    }

    #[inline(always)]
    fn _mm_cmpgt(a: __m256i, b: __m256i) -> __m256i {
        unsafe { _mm256_cmpgt_epi16(a, b) }
    }

    #[inline(always)]
    fn _mm_cmplt(a: __m256i, b: __m256i) -> __m256i {
        unsafe { _mm256_cmpgt_epi16(b, a) }
    }

    #[inline(always)]
    fn _mm_blendv(a: __m256i, b: __m256i, mask: __m256i) -> __m256i {
        unsafe { _mm256_blendv_epi8(a, b, mask) }
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

    fn get_array_i16(n: usize) -> Array1<i16> {
        utils::get_random_array(n, i16::MIN, i16::MAX)
    }

    #[test]
    fn test_both_versions_return_the_same_results() {
        let data = get_array_i16(513);
        assert_eq!(data.len() % 16, 1);

        let (argmin_index, argmax_index) = scalar_argminmax(data.view());
        let (argmin_simd_index, argmax_simd_index) = AVX2::argminmax(data.view());
        assert_eq!(argmin_index, argmin_simd_index);
        assert_eq!(argmax_index, argmax_simd_index);
    }

    #[test]
    fn test_first_index_is_returned_when_identical_values_found() {
        let data = [
            10,
            std::i16::MIN,
            6,
            9,
            9,
            22,
            std::i16::MAX,
            4,
            std::i16::MAX,
        ];
        let data: Vec<i16> = data.iter().map(|x| *x).collect();
        let data = Array1::from(data);

        let (argmin_index, argmax_index) = scalar_argminmax(data.view());
        assert_eq!(argmin_index, 1);
        assert_eq!(argmax_index, 6);

        let (argmin_simd_index, argmax_simd_index) = AVX2::argminmax(data.view());
        assert_eq!(argmin_simd_index, 1);
        assert_eq!(argmax_simd_index, 6);
    }

    #[test]
    fn test_many_random_runs() {
        for _ in 0..10_000 {
            let data = get_array_i16(32 * 2 + 1);
            let (argmin_index, argmax_index) = scalar_argminmax(data.view());
            let (argmin_simd_index, argmax_simd_index) = AVX2::argminmax(data.view());
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }
    }
}

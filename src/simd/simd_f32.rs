use super::config::{SIMDInstructionSet, AVX2};
use super::generic::SIMD;
use std::arch::x86_64::*;

const LANE_SIZE: usize = AVX2::LANE_SIZE_32;

impl SIMD<f32, __m256, LANE_SIZE> for AVX2 {
    #[inline(always)]
    fn _initial_index() -> __m256 {
        unsafe { _mm256_set_ps(7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0) }
    }

    #[inline(always)]
    fn _reg_to_arr(reg: __m256) -> [f32; LANE_SIZE] {
        unsafe { std::mem::transmute::<__m256, [f32; LANE_SIZE]>(reg) }
    }

    #[inline(always)]
    fn _mm_load(data: *const f32) -> __m256 {
        unsafe { _mm256_loadu_ps(data as *const f32) }
    }

    #[inline(always)]
    fn _mm_set1(a: usize) -> __m256 {
        unsafe { _mm256_set1_ps(a as f32) }
    }

    #[inline(always)]
    fn _mm_add(a: __m256, b: __m256) -> __m256 {
        unsafe { _mm256_add_ps(a, b) }
    }

    #[inline(always)]
    fn _mm_cmpgt(a: __m256, b: __m256) -> __m256 {
        unsafe { _mm256_cmp_ps(a, b, _CMP_GT_OQ) }
    }

    #[inline(always)]
    fn _mm_cmplt(a: __m256, b: __m256) -> __m256 {
        unsafe { _mm256_cmp_ps(b, a, _CMP_GT_OQ) }
    }

    #[inline(always)]
    fn _mm_blendv(a: __m256, b: __m256, mask: __m256) -> __m256 {
        unsafe { _mm256_blendv_ps(a, b, mask) }
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
        let (argmin_simd_index, argmax_simd_index) = AVX2::argminmax(data.view());
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

        let (argmin_simd_index, argmax_simd_index) = AVX2::argminmax(data.view());
        assert_eq!(argmin_simd_index, 3);
        assert_eq!(argmax_simd_index, 1);
    }

    #[test]
    fn test_many_random_runs() {
        for _ in 0..10_000 {
            let data = get_array_f32(32 * 8 + 1);
            let (argmin_index, argmax_index) = scalar_argminmax(data.view());
            let (argmin_simd_index, argmax_simd_index) = AVX2::argminmax(data.view());
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }
    }
}

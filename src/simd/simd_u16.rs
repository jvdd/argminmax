use super::config::{SIMDInstructionSet, AVX2};
use super::generic::SIMD;
use crate::utils::{max_index_value, min_index_value};
use num_traits::AsPrimitive;
use std::arch::x86_64::*;

const LANE_SIZE: usize = AVX2::LANE_SIZE_16;

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn _u16_to_i16decrord(u16: __m256i) -> __m256i {
    // on a scalar: v^ 0x7FFF
    // transforms to monotonically **decreasing** order
    _mm256_xor_si256(u16, _mm256_set1_epi16(0x7FFF))
}

#[inline(always)]
fn _decr_ord_i16_to_u16(ord_i16: i16) -> u16 {
    // let v = ord_i16 ^ 0x7FFF;
    unsafe { std::mem::transmute::<i16, u16>(ord_i16 ^ 0x7FFF) }
}

#[inline(always)]
fn _reg_to_i16_arr(reg: __m256i) -> [i16; LANE_SIZE] {
    unsafe { std::mem::transmute::<__m256i, [i16; LANE_SIZE]>(reg) }
}

impl SIMD<u16, __m256i, LANE_SIZE> for AVX2 {
    #[inline(always)]
    fn _initial_index() -> __m256i {
        unsafe { _mm256_set_epi16(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0) }
    }

    #[inline(always)]
    fn _reg_to_arr(_: __m256i) -> [u16; LANE_SIZE] {
        // Not used because we work with i16ord and override _get_min_index_value and _get_max_index_value
        unimplemented!()
    }

    #[inline(always)]
    fn _mm_load(data: *const u16) -> __m256i {
        unsafe { _u16_to_i16decrord(_mm256_loadu_si256(data as *const __m256i)) }
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

    // ------------------------------------ ARGMINMAX --------------------------------------

    #[inline]
    fn _get_min_max_index_value(
        index_low: __m256i,
        values_low: __m256i,
        index_high: __m256i,
        values_high: __m256i,
    ) -> (usize, u16, usize, u16) {
        let index_low_arr = _reg_to_i16_arr(index_low);
        let values_low_arr = _reg_to_i16_arr(values_low);
        let index_high_arr = _reg_to_i16_arr(index_high);
        let values_high_arr = _reg_to_i16_arr(values_high);
        let (min_index, min_value) = min_index_value(&index_low_arr, &values_low_arr);
        let (max_index, max_value) = max_index_value(&index_high_arr, &values_high_arr);
        // Swap min and max here because we worked with i16ord in decreasing order (max => actual min, and vice versa)
        (max_index.as_(), _decr_ord_i16_to_u16(max_value), min_index.as_(), _decr_ord_i16_to_u16(min_value))
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

    fn get_array_u16(n: usize) -> Array1<u16> {
        utils::get_random_array(n, u16::MIN, u16::MAX)
    }

    #[test]
    fn test_both_versions_return_the_same_results() {
        let data = get_array_u16(513);
        assert_eq!(data.len() % 16, 1);

        let (argmin_index, argmax_index) = scalar_argminmax(data.view());
        let (simd_argmin_index, simd_argmax_index) = AVX2::argminmax(data.view());
        assert_eq!(argmin_index, simd_argmin_index);
        assert_eq!(argmax_index, simd_argmax_index);
    }

    #[test]
    fn test_first_index_is_returned_when_identical_values_found() {
        let data = [
            10,
            std::u16::MIN,
            6,
            9,
            9,
            22,
            std::u16::MAX,
            4,
            std::u16::MAX,
        ];
        let data: Vec<u16> = data.iter().map(|x| *x).collect();
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
            let data = get_array_u16(32 * 2 + 1);
            let (argmin_index, argmax_index) = scalar_argminmax(data.view());
            let (argmin_simd_index, argmax_simd_index) = AVX2::argminmax(data.view());
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }
    }
}

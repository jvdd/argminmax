use super::config::{SIMDInstructionSet, AVX2};
use super::generic::SIMD;
use std::arch::x86_64::*;

use crate::utils::{max_index_value, min_index_value};
#[cfg(feature = "half")]
use half::f16;
use num_traits::AsPrimitive;

const LANE_SIZE: usize = AVX2::LANE_SIZE_16;

// ------------------------------------ ARGMINMAX --------------------------------------

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn _f16_as_m256i_to_i16ord(f16_as_m256i: __m256i) -> __m256i {
    // on a scalar: ((v >> 15) & 0x7FFF) ^ v
    let sign_bit_shifted = _mm256_srai_epi16(f16_as_m256i, 15);
    let sign_bit_masked = _mm256_and_si256(sign_bit_shifted, _mm256_set1_epi16(0x7FFF));
    _mm256_xor_si256(sign_bit_masked, f16_as_m256i)
}

#[cfg(feature = "half")]
#[inline]
fn _ord_i16_to_f16(ord_i16: i16) -> f16 {
    let v = ((ord_i16 >> 15) & 0x7FFF) ^ ord_i16;
    unsafe { std::mem::transmute::<i16, f16>(v) }
}

#[inline]
fn _reg_to_i16_arr(reg: __m256i) -> [i16; 16] {
    unsafe { std::mem::transmute::<__m256i, [i16; 16]>(reg) }
}

#[cfg(feature = "half")]
impl SIMD<f16, __m256i, LANE_SIZE> for AVX2 {
    fn _initial_index() -> __m256i {
        unsafe { _mm256_set_epi16(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0) }
    }

    fn _reg_to_arr(_: __m256i) -> [f16; LANE_SIZE] {
        // Not used because we work with i16ord and override _get_min_index_value and _get_max_index_value
        unimplemented!()
    }

    fn _mm_load(data: *const f16) -> __m256i {
        unsafe { _f16_as_m256i_to_i16ord(_mm256_loadu_si256(data as *const __m256i)) }
    }

    fn _mm_set1(a: usize) -> __m256i {
        unsafe { _mm256_set1_epi16(a as i16) }
    }

    fn _mm_add(a: __m256i, b: __m256i) -> __m256i {
        unsafe { _mm256_add_epi16(a, b) }
    }

    fn _mm_cmpgt(a: __m256i, b: __m256i) -> __m256i {
        unsafe { _mm256_cmpgt_epi16(a, b) }
    }

    fn _mm_cmplt(a: __m256i, b: __m256i) -> __m256i {
        unsafe { _mm256_cmpgt_epi16(b, a) }
    }

    fn _mm_blendv(a: __m256i, b: __m256i, mask: __m256i) -> __m256i {
        unsafe { _mm256_blendv_epi8(a, b, mask) }
    }

    // ------------------------------------ ARGMINMAX --------------------------------------

    fn _get_min_index_value(index_low: __m256i, values_low: __m256i) -> (usize, f16) {
        let index_low_arr = _reg_to_i16_arr(index_low);
        let values_low_arr = _reg_to_i16_arr(values_low);
        let (min_index, min_value) = min_index_value(&index_low_arr, &values_low_arr);
        (min_index.as_(), _ord_i16_to_f16(min_value))
    }

    fn _get_max_index_value(index_low: __m256i, values_low: __m256i) -> (usize, f16) {
        let index_low_arr = _reg_to_i16_arr(index_low);
        let values_low_arr = _reg_to_i16_arr(values_low);
        let (max_index, max_value) = max_index_value(&index_low_arr, &values_low_arr);
        (max_index.as_(), _ord_i16_to_f16(max_value))
    }
}

//----- TESTS -----

#[cfg(feature = "half")]
#[cfg(test)]
mod tests {
    use super::{AVX2, SIMD};
    use crate::scalar::scalar_generic::scalar_argminmax;

    use half::f16;
    use ndarray::Array1;

    extern crate dev_utils;
    use dev_utils::utils;

    fn get_array_f16(n: usize) -> Array1<f16> {
        let arr = utils::get_random_array(n, i16::MIN, i16::MAX);
        let arr = arr.mapv(|x| f16::from_f32(x as f32));
        Array1::from(arr)
    }

    #[test]
    fn test_both_versions_return_the_same_results() {
        let data = get_array_f16(1025);
        assert_eq!(data.len() % 8, 1);

        let (argmin_index, argmax_index) = scalar_argminmax(data.view());
        let (argmin_simd_index, argmax_simd_index) = AVX2::argminmax(data.view());
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
            let data = get_array_f16(32 * 8 + 1);
            let (argmin_index, argmax_index) = scalar_argminmax(data.view());
            let (argmin_simd_index, argmax_simd_index) = AVX2::argminmax(data.view());
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }
    }
}

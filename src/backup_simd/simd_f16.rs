use crate::task::argminmax_generic;
use crate::utils::{max_index_value, min_index_value};
use ndarray::ArrayView1;
use std::arch::x86_64::*;

#[cfg(feature = "half")]
use half::f16;

const LANE_SIZE: usize = 16;

// ------------------------------------ ARGMINMAX --------------------------------------

#[cfg(feature = "half")]
pub fn argminmax_f16(arr: ArrayView1<f16>) -> (usize, usize) {
    argminmax_generic(arr, LANE_SIZE, core_argminmax_256)
}

#[inline]
fn reg_to_i16_arr(reg: __m256i) -> [i16; 16] {
    unsafe { std::mem::transmute::<__m256i, [i16; 16]>(reg) }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn f16_as_m256i_to_ord_i16(f16_as_m256i: __m256i) -> __m256i {
    // on a scalar: ((v >> 15) & 0x7FFF) ^ v
    let sign_bit_shifted = _mm256_srai_epi16(f16_as_m256i, 15);
    let sign_bit_masked = _mm256_and_si256(sign_bit_shifted, _mm256_set1_epi16(0x7FFF));
    _mm256_xor_si256(sign_bit_masked, f16_as_m256i)
}

#[cfg(feature = "half")]
#[inline]
fn ord_i16_to_f16(ord_i16: i16) -> f16 {
    let v = ((ord_i16 >> 15) & 0x7FFF) ^ ord_i16;
    unsafe { std::mem::transmute::<i16, f16>(v) }
}

#[cfg(feature = "half")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn core_argminmax_256(sim_arr: ArrayView1<f16>, offset: usize) -> (f16, usize, f16, usize) {
    // Efficient calculation of argmin and argmax together
    let offset = _mm256_set1_epi16(offset as i16);
    let mut new_index = _mm256_add_epi16(
        _mm256_set_epi16(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
        offset,
    );
    let mut index_low = new_index;
    let mut index_high = new_index;

    let increment = _mm256_set1_epi16(16);

    // println!("raw new values: {:?}", sim_arr.slice(s![0..16]));
    let new_values = _mm256_loadu_si256(sim_arr.as_ptr() as *const __m256i);
    // println!("new_values: {:?}", reg_to_i16_arr(new_values));
    let new_values = f16_as_m256i_to_ord_i16(new_values);
    // println!("new_values: {:?}", reg_to_i16_arr(new_values));
    // println!();
    let mut values_low = new_values;
    let mut values_high = new_values;

    sim_arr
        .exact_chunks(LANE_SIZE)
        .into_iter()
        .skip(1)
        .for_each(|step| {
            new_index = _mm256_add_epi16(new_index, increment);

            let new_values = _mm256_loadu_si256(step.as_ptr() as *const __m256i);
            let new_values = f16_as_m256i_to_ord_i16(new_values);
            let gt_mask = _mm256_cmpgt_epi16(new_values, values_high);
            // Below does not work (bc instruction is not available)
            //      let lt_mask = _mm256_cmplt_epi16(new_values, values_low);
            // Solution: swap parameters and use gt instead
            let lt_mask = _mm256_cmpgt_epi16(values_low, new_values);

            index_low = _mm256_blendv_epi8(index_low, new_index, lt_mask);
            index_high = _mm256_blendv_epi8(index_high, new_index, gt_mask);

            values_low = _mm256_blendv_epi8(values_low, new_values, lt_mask);
            values_high = _mm256_blendv_epi8(values_high, new_values, gt_mask);
        });

    // Select max_index and max_value
    let value_array = reg_to_i16_arr(values_high);
    let index_array = reg_to_i16_arr(index_high);
    let (index_max, value_max) = max_index_value(&index_array, &value_array);

    // Select min_index and min_value
    let value_array = reg_to_i16_arr(values_low);
    let index_array = reg_to_i16_arr(index_low);
    let (index_min, value_min) = min_index_value(&index_array, &value_array);

    (
        ord_i16_to_f16(value_min),
        index_min as usize,
        ord_i16_to_f16(value_max),
        index_max as usize,
    )
}

//----- TESTS -----

#[cfg(feature = "half")]
#[cfg(test)]
mod tests {
    use super::argminmax_f16;
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
        let (argmin_simd_index, argmax_simd_index) = argminmax_f16(data.view());
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

        let (argmin_simd_index, argmax_simd_index) = argminmax_f16(data.view());
        assert_eq!(argmin_simd_index, 3);
        assert_eq!(argmax_simd_index, 1);
    }

    #[test]
    fn test_many_random_runs() {
        for _ in 0..10_000 {
            let data = get_array_f16(32 * 8 + 1);
            let (argmin_index, argmax_index) = scalar_argminmax(data.view());
            let (argmin_simd_index, argmax_simd_index) = argminmax_f16(data.view());
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }
    }
}

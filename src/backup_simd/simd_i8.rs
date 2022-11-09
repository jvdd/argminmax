use crate::utils::{max_index_value, min_index_value};
use crate::task::argminmax_generic;
use ndarray::ArrayView1;
use std::arch::x86_64::*;

const LANE_SIZE: usize = 32;

// ------------------------------------ ARGMINMAX --------------------------------------

pub fn argminmax_i8(arr: ArrayView1<i8>) -> (usize, usize) {
    argminmax_generic(arr, LANE_SIZE, core_argminmax_256)
}

#[inline]
fn reg_to_i8_arr(reg: __m256i) -> [i8; 32] {
    unsafe { std::mem::transmute::<__m256i, [i8; 32]>(reg) }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn core_argminmax_256(sim_arr: ArrayView1<i8>, offset: usize) -> (i8, usize, i8, usize) {
    // Efficient calculation of argmin and argmax together
    let offset = _mm256_set1_epi8(offset as i8);
    let mut index_low = _mm256_add_epi8(
        _mm256_set_epi8(
            31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10,
            9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
        ),
        offset,
    );
    let mut new_index_low = index_low;
    let mut index_high = index_low;
    let mut new_index_high = index_high;

    let increment = _mm256_set1_epi8(32);

    let mut values_low = _mm256_loadu_si256(sim_arr.as_ptr() as *const __m256i);
    let mut values_high = values_low;

    sim_arr
        .exact_chunks(32)
        .into_iter()
        .skip(1)
        .for_each(|step| {
            new_index_low = _mm256_add_epi8(new_index_low, increment);
            new_index_high = _mm256_add_epi8(new_index_high, increment);

            let new_values = _mm256_loadu_si256(step.as_ptr() as *const __m256i);
            let gt_mask = _mm256_cmpgt_epi8(new_values, values_high);
            // Below does not work (bc instruction is not available)
            //      let lt_mask = _mm256_cmplt_epi8(new_values, values_low);
            // Solution: swap parameters and use gt instead
            let lt_mask = _mm256_cmpgt_epi8(values_low, new_values);

            values_high = _mm256_or_si256(
                _mm256_and_si256(new_values, gt_mask),
                _mm256_andnot_si256(gt_mask, values_high),
            );
            index_high = _mm256_or_si256(
                _mm256_and_si256(new_index_high, gt_mask),
                _mm256_andnot_si256(gt_mask, index_high),
            );

            values_low = _mm256_or_si256(
                _mm256_and_si256(new_values, lt_mask),
                _mm256_andnot_si256(lt_mask, values_low),
            );
            index_low = _mm256_or_si256(
                _mm256_and_si256(new_index_low, lt_mask),
                _mm256_andnot_si256(lt_mask, index_low),
            );
        });

    // Select max_index and max_value
    let value_array = reg_to_i8_arr(values_high);
    let index_array = reg_to_i8_arr(index_high);
    let (index_max, value_max) = max_index_value(&index_array, &value_array);

    // Select min_index and min_value
    let value_array = reg_to_i8_arr(values_low);
    let index_array = reg_to_i8_arr(index_low);
    let (index_min, value_min) = min_index_value(&index_array, &value_array);

    (value_min, index_min as usize, value_max, index_max as usize)
}

//----- TESTS -----

#[cfg(test)]
mod tests {
    use super::argminmax_i8;
    use crate::scalar::scalar_generic::scalar_argminmax;

    use ndarray::Array1;

    extern crate dev_utils;
    use dev_utils::utils;

    fn get_array_i8(n: usize) -> Array1<i8> {
        utils::get_random_array(n, i8::MIN, i8::MAX)
    }

    #[test]
    fn test_both_versions_return_the_same_results() {
        let data = get_array_i8(32 * 6 + 1); // TODO: lengte mag niet > 2^8 zijn...
        assert_eq!(data.len() % 32, 1);

        let (argmin_index, argmax_index) = scalar_argminmax(data.view());
        let (argmin_simd_index, argmax_simd_index) = argminmax_i8(data.view());
        assert_eq!(argmin_index, argmin_simd_index);
        assert_eq!(argmax_index, argmax_simd_index);
    }

    #[test]
    fn test_first_index_is_returned_when_identical_values_found() {
        let data = [10, std::i8::MIN, 6, 9, 9, 22, std::i8::MAX, 4, std::i8::MAX];
        let data: Vec<i8> = data.iter().map(|x| *x).collect();
        let data = Array1::from(data);

        let (argmin_index, argmax_index) = scalar_argminmax(data.view());
        assert_eq!(argmin_index, 1);
        assert_eq!(argmax_index, 6);

        let (argmin_simd_index, argmax_simd_index) = argminmax_i8(data.view());
        assert_eq!(argmin_simd_index, 1);
        assert_eq!(argmax_simd_index, 6);
    }

    #[test]
    fn test_many_random_runs() {
        for _ in 0..10_000 {
            let data = get_array_i8(32 * 2 + 1);
            let (argmin_index, argmax_index) = scalar_argminmax(data.view());
            let (argmin_simd_index, argmax_simd_index) = argminmax_i8(data.view());
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }
    }
}

use crate::task::argminmax_generic;
use crate::utils::{max_index_value, min_index_value};
use ndarray::ArrayView1;
use std::arch::x86_64::*;

const LANE_SIZE: usize = 4;

// ------------------------------------ ARGMINMAX --------------------------------------

pub fn argminmax_f64(arr: ArrayView1<f64>) -> (usize, usize) {
    argminmax_generic(arr, LANE_SIZE, core_argminmax_256)
}

#[inline]
fn reg_to_f64_arr(reg: __m256d) -> [f64; 4] {
    unsafe { std::mem::transmute::<__m256d, [f64; 4]>(reg) }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn core_argminmax_256(sim_arr: ArrayView1<f64>, offset: usize) -> (f64, usize, f64, usize) {
    // Efficient calculation of argmin and argmax together
    let offset = _mm256_set1_pd(offset as f64);
    let mut new_index = _mm256_add_pd(_mm256_set_pd(3.0, 2.0, 1.0, 0.0), offset);
    let mut index_low = new_index;
    let mut index_high = new_index;

    let increment = _mm256_set1_pd(4.0);

    let new_values = _mm256_loadu_pd(sim_arr.as_ptr() as *const f64);
    let mut values_low = new_values;
    let mut values_high = new_values;

    sim_arr
        .exact_chunks(4)
        .into_iter()
        .skip(1)
        .for_each(|step| {
            new_index = _mm256_add_pd(new_index, increment);

            let new_values = _mm256_loadu_pd(step.as_ptr() as *const f64);
            let gt_mask = _mm256_cmp_pd(new_values, values_high, _CMP_GT_OQ);
            let lt_mask = _mm256_cmp_pd(new_values, values_low, _CMP_LT_OQ);

            values_low = _mm256_min_pd(new_values, values_low);
            values_high = _mm256_max_pd(new_values, values_high);

            index_low = _mm256_blendv_pd(index_low, new_index, lt_mask);
            index_high = _mm256_blendv_pd(index_high, new_index, gt_mask);
        });

    // Select max_index and max_value
    let value_array = reg_to_f64_arr(values_high);
    let index_array = reg_to_f64_arr(index_high);
    let (index_max, value_max) = max_index_value(&index_array, &value_array);

    // Select min_index and min_value
    let value_array = reg_to_f64_arr(values_low);
    let index_array = reg_to_f64_arr(index_low);
    let (index_min, value_min) = min_index_value(&index_array, &value_array);

    (value_min, index_min as usize, value_max, index_max as usize)
}

//----- TESTS -----

#[cfg(test)]
mod tests {
    use super::argminmax_f64;
    use crate::scalar_generic::scalar_argminmax;

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
        let (argmin_simd_index, argmax_simd_index) = argminmax_f64(data.view());
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

        let (argmin_simd_index, argmax_simd_index) = argminmax_f64(data.view());
        assert_eq!(argmin_simd_index, 3);
        assert_eq!(argmax_simd_index, 1);
    }

    #[test]
    fn test_many_random_runs() {
        for _ in 0..10_000 {
            let data = get_array_f64(32 * 8 + 1);
            let (argmin_index, argmax_index) = scalar_argminmax(data.view());
            let (argmin_simd_index, argmax_simd_index) = argminmax_f64(data.view());
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }
    }
}

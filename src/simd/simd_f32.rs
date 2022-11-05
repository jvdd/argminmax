use crate::utils::{max_index_value, min_index_value};
use crate::task::argminmax_generic;
use ndarray::ArrayView1;
use std::arch::x86_64::*;

const LANE_SIZE: usize = 8;

// ------------------------------------ ARGMINMAX --------------------------------------

pub fn argminmax_f32(arr: ArrayView1<f32>) -> (usize, usize) {
    argminmax_generic(arr, LANE_SIZE, core_argminmax_256)
}

#[inline]
fn reg_to_f32_arr(reg: __m256) -> [f32; 8] {
    unsafe { std::mem::transmute::<__m256, [f32; 8]>(reg) }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn core_argminmax_256(sim_arr: ArrayView1<f32>, offset: usize) -> (f32, usize, f32, usize) {
    // Efficient calculation of argmin and argmax together
    let offset = _mm256_set1_ps(offset as f32);
    let mut new_index = _mm256_add_ps(
        _mm256_set_ps(7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0),
        offset,
    );
    let mut index_low = new_index;
    let mut index_high = new_index;

    let increment = _mm256_set1_ps(8.0);

    let new_values = _mm256_loadu_ps(sim_arr.as_ptr() as *const f32);
    let mut values_low = new_values;
    let mut values_high = new_values;

    sim_arr
        .exact_chunks(8)
        .into_iter()
        .skip(1)
        .for_each(|step| {
            new_index = _mm256_add_ps(new_index, increment);

            let new_values = _mm256_loadu_ps(step.as_ptr() as *const f32);
            let lt_mask = _mm256_cmp_ps(new_values, values_low, _CMP_LT_OQ);
            let gt_mask = _mm256_cmp_ps(new_values, values_high, _CMP_GT_OQ);

            // Performing the _mm256_testz_ps check for lt_mask and gt_mask (and then
            // updating values and index if needed) is slower than directly updating

            // Performing min_ps and max_ps is faster than using _mm256_blendv_ps with mask
            values_low = _mm256_min_ps(new_values, values_low);
            values_high = _mm256_max_ps(new_values, values_high);

            // Performing _mm256_blendv_ps is faster than using or_ps ( and_ps, andnot_ps )
            index_low = _mm256_blendv_ps(index_low, new_index, lt_mask);
            index_high = _mm256_blendv_ps(index_high, new_index, gt_mask);
        });

    // Select max_index and max_value
    let value_array = reg_to_f32_arr(values_high);
    let index_array = reg_to_f32_arr(index_high);
    let (index_max, value_max) = max_index_value(&index_array, &value_array);

    // Select min_index and min_value
    let value_array = reg_to_f32_arr(values_low);
    let index_array = reg_to_f32_arr(index_low);
    let (index_min, value_min) = min_index_value(&index_array, &value_array);

    (value_min, index_min as usize, value_max, index_max as usize)
}

//----- TESTS -----

#[cfg(test)]
mod tests {
    use super::argminmax_f32;
    use crate::scalar_generic::scalar_argminmax;

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
        let (argmin_simd_index, argmax_simd_index) = argminmax_f32(data.view());
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

        let (argmin_simd_index, argmax_simd_index) = argminmax_f32(data.view());
        assert_eq!(argmin_simd_index, 3);
        assert_eq!(argmax_simd_index, 1);
    }

    #[test]
    fn test_many_random_runs() {
        for _ in 0..10_000 {
            let data = get_array_f32(32 * 8 + 1);
            let (argmin_index, argmax_index) = scalar_argminmax(data.view());
            let (argmin_simd_index, argmax_simd_index) = argminmax_f32(data.view());
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }
    }
}

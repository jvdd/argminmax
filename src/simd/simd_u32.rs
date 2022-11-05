use crate::generic::{max_index_value, min_index_value};
use crate::task::argminmax_generic;
use ndarray::ArrayView1;
use std::arch::x86_64::*;

const LANE_SIZE: usize = 8;

// ------------------------------------ ARGMINMAX --------------------------------------

pub fn argminmax_u16(arr: ArrayView1<u16>) -> Option<(usize, usize)> {
    argminmax_generic(arr, LANE_SIZE, core_argminmax_256)
}

#[inline]
fn reg_to_u16_arr(reg: __m256i) -> [u16; 8] {
    unsafe { std::mem::transmute::<__m256i, [u16; 8]>(reg) }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn core_argminmax_256(sim_arr: ArrayView1<u16>, offset: usize) -> (u16, usize, u16, usize) {
    // Efficient calculation of argmin and argmax together
    let offset = _mm256_set1_epi32(offset as i32);
    let mut new_index = _mm256_add_epi32(
        _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0),
        offset,
    );
    let mut index_low = new_index;
    let mut index_high = new_index;

    let increment = _mm256_set1_epi32(8);

    let new_values = _mm256_loadu_si256(sim_arr.as_ptr() as *const __m256i);
    let mut values_low = new_values;
    let mut values_high = new_values;

    sim_arr
        .exact_chunks(8)
        .into_iter()
        .skip(1)
        .for_each(|step| {
            new_index = _mm256_add_epi32(new_index, increment);

            let new_values = _mm256_loadu_si256(step.as_ptr() as *const __m256i);
            let gt_mask = _mm256_cmpgt_epi32(new_values, values_high);
            let lt_mask = _mm256_cmpgt_epi32(values_low, new_values);

            values_high = _mm256_blendv_epi8(values_high, new_values, gt_mask);
            values_low = _mm256_blendv_epi8(values_low, new_values, lt_mask);

            index_high = _mm256_blendv_epi8(index_high, new_index, gt_mask);
            index_low = _mm256_blendv_epi8(index_low, new_index, lt_mask);
        });

    // Select max_index and max_value
    let value_array = reg_to_u16_arr(values_high);
    let index_array = reg_to_u16_arr(index_high);
    let (index_max, value_max) = max_index_value(&index_array, &value_array);

    // Select min_index and min_value
    let value_array = reg_to_u16_arr(values_low);
    let index_array = reg_to_u16_arr(index_low);
    let (index_min, value_min) = min_index_value(&index_array, &value_array);

    (value_min, index_min as usize, value_max, index_max as usize)
}

//----- TESTS -----

#[cfg(test)]
mod tests {
    use super::argminmax_u32;
    use crate::generic;
    use generic::scalar_argminmax;

    use ndarray::Array1;

    extern crate dev_utils;
    use dev_utils::utils;

    // TODO: duplicate code in bench config
    fn get_array_u16(n: usize) -> Array1<u16> {
        let rng = thread_rng();
        let uni = Uniform::new_inclusive(std::u16::MIN, std::u16::MAX);
        let arr: Vec<u16> = rng.sample_iter(uni).take(n).collect();
        Array1::from(arr)
    }

    #[test]
    fn test_both_versions_return_the_same_results() {
        let data = get_array_u16(1025);
        assert_eq!(data.len() % 8, 1);

        let (argmin_index, argmax_index) = scalar_argminmax(data.view());
        let (argmin_simd_index, argmax_simd_index) = argminmax_u16(data.view());
        assert_eq!(argmin_index, argmin_simd_index);
        assert_eq!(argmax_index, argmax_simd_index);
    }

    #[test]
    fn test_first_index_is_returned_when_identical_values_found() {
        let data = [
            10,
            std::u16::MAX,
            6,
            std::u16::MAX,
            10_000,
        ];
        let data: Vec<u16> = data.iter().map(|x| *x).collect();
        let data = Array1::from(data);

        let (argmin_index, argmax_index) = scalar_argminmax(data.view());
        assert_eq!(argmin_index, 3);
        assert_eq!(argmax_index, 1);

        let (argmin_simd_index, argmax_simd_index) = argminmax_u16(data.view());
        assert_eq!(argmin_simd_index, 3);
        assert_eq!(argmax_simd_index, 1);
    }

    #[test]
    fn test_many_random_runs() {
        for _ in 0..10_000 {
            let data = get_array_u16(32 * 8 + 1);
            let (argmin_index, argmax_index) = scalar_argminmax(data.view());
            let (argmin_simd_index, argmax_simd_index) = argminmax_u16(data.view());
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }
    }
}

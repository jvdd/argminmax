use crate::scalar::{ScalarArgMinMax, SCALAR};

use std::cmp::Ordering;

#[inline(always)]
pub(crate) fn argminmax_generic<T: Copy + PartialOrd>(
    arr: &[T],
    lane_size: usize,
    core_argminmax: unsafe fn(&[T]) -> (usize, T, usize, T),
) -> (usize, usize)
where
    SCALAR: ScalarArgMinMax<T>,
{
    assert!(!arr.is_empty()); // split_array should never return (None, None)
    match split_array(arr, lane_size) {
        (Some(sim), Some(rem)) => {
            let (rem_min_index, rem_max_index) = SCALAR::argminmax(rem);
            let rem_result = (
                rem_min_index + sim.len(),
                rem[rem_min_index],
                rem_max_index + sim.len(),
                rem[rem_max_index],
            );
            let sim_result = unsafe { core_argminmax(sim) };
            find_final_index_minmax(rem_result, sim_result)
        }
        (None, Some(rem)) => {
            let (rem_min_index, rem_max_index) = SCALAR::argminmax(rem);
            (rem_min_index, rem_max_index)
        }
        (Some(sim), None) => {
            let sim_result = unsafe { core_argminmax(sim) };
            (sim_result.0, sim_result.2)
        }
        (None, None) => panic!("Array is empty"), // Should never occur because of assert
    }
}

// #[inline(always)]
fn split_array<T: Copy>(arr: &[T], lane_size: usize) -> (Option<&[T]>, Option<&[T]>) {
    let n = arr.len();

    // if n < lane_size * 2 {
    //     // TODO: check if this is the best threshold
    //     return (None, Some(arr));
    // };

    let (left_arr, right_arr) = arr.split_at(n - n % lane_size);

    match (left_arr.is_empty(), right_arr.is_empty()) {
        (true, true) => (None, None),
        (false, false) => (Some(left_arr), Some(right_arr)),
        (true, false) => (None, Some(right_arr)),
        (false, true) => (Some(left_arr), None),
    }
}

#[inline(always)]
fn find_final_index_minmax<T: Copy + PartialOrd>(
    remainder_result: (usize, T, usize, T),
    simd_result: (usize, T, usize, T),
) -> (usize, usize) {
    let min_result = match remainder_result.1.partial_cmp(&simd_result.1).unwrap() {
        Ordering::Less => remainder_result.0,
        Ordering::Equal => std::cmp::min(remainder_result.0, simd_result.0),
        Ordering::Greater => simd_result.0,
    };

    let max_result = match simd_result.3.partial_cmp(&remainder_result.3).unwrap() {
        Ordering::Less => remainder_result.2,
        Ordering::Equal => std::cmp::min(remainder_result.2, simd_result.2),
        Ordering::Greater => simd_result.2,
    };

    (min_result, max_result)
}

// ------------ Other helper functions

// #[inline(always)]
pub(crate) fn min_index_value<T: Copy + PartialOrd>(index: &[T], values: &[T]) -> (T, T) {
    assert!(!index.is_empty());
    assert_eq!(index.len(), values.len());
    let mut min_index: T = unsafe { *index.get_unchecked(0) };
    let mut min_value: T = unsafe { *values.get_unchecked(0) };
    for i in 0..values.len() {
        let v: T = unsafe { *values.get_unchecked(i) };
        let idx: T = unsafe { *index.get_unchecked(i) };
        if v < min_value || (v == min_value && idx < min_index) {
            min_value = v;
            min_index = idx;
        }
    }
    (min_index, min_value)
}

// #[inline(always)]
pub(crate) fn max_index_value<T: Copy + PartialOrd>(index: &[T], values: &[T]) -> (T, T) {
    assert!(!index.is_empty());
    assert_eq!(index.len(), values.len());
    let mut max_index: T = unsafe { *index.get_unchecked(0) };
    let mut max_value: T = unsafe { *values.get_unchecked(0) };
    for i in 0..values.len() {
        let v: T = unsafe { *values.get_unchecked(i) };
        let idx: T = unsafe { *index.get_unchecked(i) };
        if v > max_value || (v == max_value && idx < max_index) {
            max_value = v;
            max_index = idx;
        }
    }
    (max_index, max_value)
}

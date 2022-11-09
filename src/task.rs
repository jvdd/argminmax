use crate::scalar::scalar_generic::scalar_argminmax; // TODO: dit in macro doorgeven

use ndarray::{ArrayView1, Axis};
use std::cmp::Ordering;

#[inline]
pub(crate) fn argminmax_generic<T: Copy + PartialOrd>(
    arr: ArrayView1<T>,
    lane_size: usize,
    core_argminmax: unsafe fn(ArrayView1<T>, usize) -> (T, usize, T, usize),
) -> (usize, usize) {
    assert!(!arr.is_empty()); // split_array should never return (None, None)
    match split_array(arr, lane_size) {
        (Some(rem), Some(sim)) => {
            let (rem_min_index, rem_max_index) = scalar_argminmax(rem);
            let rem_result = (
                rem[rem_min_index],
                rem_min_index,
                rem[rem_max_index],
                rem_max_index,
            );
            let sim_result = unsafe { core_argminmax(sim, rem.len()) };
            find_final_index_minmax(rem_result, sim_result)
        }
        (Some(rem), None) => {
            let (rem_min_index, rem_max_index) = scalar_argminmax(rem);
            (rem_min_index, rem_max_index)
        }
        (None, Some(sim)) => {
            let sim_result = unsafe { core_argminmax(sim, 0) };
            (sim_result.1, sim_result.3)
        }
        (None, None) => panic!("Array is empty"), // Should never occur because of assert
    }
}

#[inline]
fn split_array<T: Copy>(
    arr: ArrayView1<T>,
    lane_size: usize,
) -> (Option<ArrayView1<T>>, Option<ArrayView1<T>>) {
    let n = arr.len();

    if n < lane_size * 2 {
        return (Some(arr), None);
    };

    let (left_arr, right_arr) = arr.split_at(Axis(0), n % lane_size);

    match (left_arr.is_empty(), right_arr.is_empty()) {
        (true, true) => (None, None),
        (false, false) => (Some(left_arr), Some(right_arr)),
        (true, false) => (None, Some(right_arr)),
        (false, true) => (Some(left_arr), None),
    }
}

#[inline]
fn find_final_index_minmax<T: Copy + PartialOrd>(
    remainder_result: (T, usize, T, usize),
    simd_result: (T, usize, T, usize),
) -> (usize, usize) {
    let min_result = match remainder_result.0.partial_cmp(&simd_result.0).unwrap() {
        Ordering::Less => remainder_result.1,
        Ordering::Equal => std::cmp::min(remainder_result.1, simd_result.1),
        Ordering::Greater => simd_result.1,
    };

    let max_result = match simd_result.2.partial_cmp(&remainder_result.2).unwrap() {
        Ordering::Less => remainder_result.3,
        Ordering::Equal => std::cmp::min(remainder_result.3, simd_result.3),
        Ordering::Greater => simd_result.3,
    };

    (min_result, max_result)
}

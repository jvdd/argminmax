use std::cmp::Ordering;

use num_traits::Bounded;

use crate::scalar::ScalarArgMinMax;
use crate::validity::{first_valid_eq, merge_max, merge_min, scalar_masked, FoundMinMax};

#[inline(always)]
pub(crate) fn argminmax_generic<T: Copy + PartialOrd>(
    arr: &[T],
    lane_size: usize,
    core_argminmax: unsafe fn(&[T]) -> (usize, T, usize, T),
    scalar_argminmax: fn(&[T]) -> (usize, usize),
    nan_check: fn(T) -> bool, // returns true if value is NaN
    ignore_nan: bool,         // if false, NaNs will be returned
) -> (usize, usize) {
    assert!(!arr.is_empty()); // split_array should never return (None, None)
    let (min_index, max_index) = match split_array(arr, lane_size) {
        (Some(simd_arr), Some(rem)) => {
            // Perform SIMD operation on the first part of the array
            let simd_result = unsafe { core_argminmax(simd_arr) };
            // Perform scalar operation on the remainder of the array
            let (rem_min_index, rem_max_index) = scalar_argminmax(rem);
            // let (rem_min_index, rem_max_index) = SCALAR::argminmax(rem);
            let rem_result = (
                rem_min_index + simd_arr.len(),
                rem[rem_min_index],
                rem_max_index + simd_arr.len(),
                rem[rem_max_index],
            );
            // Find the final min and max values
            let (min_index, min_value) = find_final_index_min(
                (simd_result.0, simd_result.1),
                (rem_result.0, rem_result.1),
                nan_check,
                ignore_nan,
            );
            let (max_index, max_value) = find_final_index_max(
                (simd_result.2, simd_result.3),
                (rem_result.2, rem_result.3),
                nan_check,
                ignore_nan,
            );
            get_correct_argminmax_result(
                min_index, min_value, max_index, max_value, nan_check, ignore_nan,
            )
        }
        (Some(simd_arr), None) => {
            let (min_index, min_value, max_index, max_value) = unsafe { core_argminmax(simd_arr) };
            get_correct_argminmax_result(
                min_index, min_value, max_index, max_value, nan_check, ignore_nan,
            )
        }
        (None, Some(rem)) => {
            let (rem_min_index, rem_max_index) = scalar_argminmax(rem);
            (rem_min_index, rem_max_index)
        }
        (None, None) => panic!("Array is empty"), // Should never occur because of assert
    };
    if ignore_nan {
        // A NaN min (max) index means that all non-NaN values are +inf (-inf), see
        // `first_non_nan_if_nan`. The max (min) index is then the first of them, as ties
        // keep the first index (or 0 if all values are NaN). So no scan is needed.
        if nan_check(arr[min_index]) {
            return (max_index, max_index);
        }
        if nan_check(arr[max_index]) {
            return (min_index, min_index);
        }
    }
    (min_index, max_index)
}

#[inline(always)]
pub(crate) fn argmin_generic<T: Copy + PartialOrd>(
    arr: &[T],
    lane_size: usize,
    core_argmin: unsafe fn(&[T]) -> (usize, T),
    scalar_argmin: fn(&[T]) -> usize,
    nan_check: fn(T) -> bool, // returns true if value is NaN
    ignore_nan: bool,         // if false, NaNs will be returned
) -> usize {
    assert!(!arr.is_empty()); // split_array should never return (None, None)
    let index = match split_array(arr, lane_size) {
        (Some(simd_arr), Some(rem)) => {
            // Perform SIMD operation on the first part of the array
            let simd_result = unsafe { core_argmin(simd_arr) };
            // Perform scalar operation on the remainder of the array
            let rem_min_index = scalar_argmin(rem);
            let rem_result = (rem_min_index + simd_arr.len(), rem[rem_min_index]);
            // Find the final min value
            let (min_index, _) =
                find_final_index_min(simd_result, rem_result, nan_check, ignore_nan);
            min_index
        }
        (Some(simd_arr), None) => {
            let (min_index, _) = unsafe { core_argmin(simd_arr) };
            min_index
        }
        (None, Some(rem)) => scalar_argmin(rem),
        (None, None) => panic!("Array is empty"), // Should never occur because of assert
    };
    first_non_nan_if_nan(arr, index, nan_check, ignore_nan)
}

#[inline(always)]
pub(crate) fn argmax_generic<T: Copy + PartialOrd>(
    arr: &[T],
    lane_size: usize,
    core_argmax: unsafe fn(&[T]) -> (usize, T),
    scalar_argmax: fn(&[T]) -> usize,
    nan_check: fn(T) -> bool, // returns true if value is NaN
    ignore_nan: bool,         // if false, NaNs will be returned
) -> usize {
    assert!(!arr.is_empty()); // split_array should never return (None, None)
    let index = match split_array(arr, lane_size) {
        (Some(simd_arr), Some(rem)) => {
            // Perform SIMD operation on the first part of the array
            let simd_result = unsafe { core_argmax(simd_arr) };
            // Perform scalar operation on the remainder of the array
            let rem_max_index = scalar_argmax(rem);
            let rem_result = (rem_max_index + simd_arr.len(), rem[rem_max_index]);
            // Find the final max value
            let (max_index, _) =
                find_final_index_max(simd_result, rem_result, nan_check, ignore_nan);
            max_index
        }
        (Some(simd_arr), None) => {
            let (max_index, _) = unsafe { core_argmax(simd_arr) };
            max_index
        }
        (None, Some(rem)) => scalar_argmax(rem),
        (None, None) => panic!("Array is empty"), // Should never occur because of assert
    };
    first_non_nan_if_nan(arr, index, nan_check, ignore_nan)
}

/// The SIMD cores that ignore NaNs start from +inf (-inf) with index 0 and only update
/// on a smaller (larger) value. So they return index 0, a NaN, when it is NaN and all
/// other values are NaN or +inf (-inf). The min (max) is then the first non-NaN value
/// (index 0 if all values are NaN). Only O(1) unless the index is NaN.
#[inline(always)]
fn first_non_nan_if_nan<T: Copy>(
    arr: &[T],
    index: usize,
    nan_check: fn(T) -> bool,
    ignore_nan: bool,
) -> usize {
    if ignore_nan && nan_check(arr[index]) {
        arr.iter().position(|&v| !nan_check(v)).unwrap_or(0)
    } else {
        index
    }
}

#[inline(always)]
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

/// Get the final index of the min value when both a SIMD and scalar result is available
/// If not ignoring NaNs (thus returning NaN index if any present):
/// - If both values are NaN, returns the index of the simd result (as the first part
///   of the array is passed to the SIMD function)
/// - If one value is NaN, returns the index of the NaN value
/// - If neither value is NaN, returns the index of the min value
///
/// If ignoring NaNs: returns the index of the min value
///
/// Note: when the values are equal, the index of the simd result is returned (as the
/// first part of the array is passed to the SIMD function)
#[inline(always)]
fn find_final_index_min<T: Copy + PartialOrd>(
    simd_result: (usize, T),
    remainder_result: (usize, T),
    nan_check: fn(T) -> bool,
    ignore_nan: bool,
) -> (usize, T) {
    let (min_index, min_value) = match simd_result.1.partial_cmp(&remainder_result.1) {
        Some(Ordering::Less) => simd_result,
        Some(Ordering::Equal) => simd_result,
        Some(Ordering::Greater) => remainder_result,
        // When returning NaNs, prefer the simd result if it is NaN (also if both are).
        // When ignoring NaNs, only the remainder value can be NaN: the simd value starts
        // at +inf and never becomes NaN.
        None if ignore_nan || nan_check(simd_result.1) => simd_result,
        None => remainder_result,
    };
    (min_index, min_value)
}

/// Get the final index of the max value when both a SIMD and scalar result is available
/// If not ignoring NaNs (thus returning NaN index if any present):
/// - If both values are NaN, returns the index of the simd result (as the first part
///   of the array is passed to the SIMD function)
/// - If one value is NaN, returns the index of the NaN value
/// - If neither value is NaN, returns the index of the max value
///
/// If ignoring NaNs: returns the index of the max value
///
/// Note: when the values are equal, the index of the simd result is returned (as the
/// first part of the array is passed to the SIMD function)
#[inline(always)]
fn find_final_index_max<T: Copy + PartialOrd>(
    simd_result: (usize, T),
    remainder_result: (usize, T),
    nan_check: fn(T) -> bool,
    ignore_nan: bool,
) -> (usize, T) {
    let (max_index, max_value) = match simd_result.1.partial_cmp(&remainder_result.1) {
        Some(Ordering::Greater) => simd_result,
        Some(Ordering::Equal) => simd_result,
        Some(Ordering::Less) => remainder_result,
        // When returning NaNs, prefer the simd result if it is NaN (also if both are).
        // When ignoring NaNs, only the remainder value can be NaN: the simd value starts
        // at -inf and never becomes NaN.
        None if ignore_nan || nan_check(simd_result.1) => simd_result,
        None => remainder_result,
    };
    (max_index, max_value)
}

/// Get the correct index(es) for the argmin and argmax functions
/// If not ignoring NaNs (thus returning NaN index if any present):
/// - If both values are NaN, returns the lowest index twice
/// - If one value is NaN, returns the index of the NaN value twice
/// - If neither value is NaN, returns the min_index and max_index
///
/// If ignoring NaNs: returns the min_index and max_index
pub(crate) fn get_correct_argminmax_result<T: Copy + PartialOrd>(
    min_index: usize,
    min_value: T,
    max_index: usize,
    max_value: T,
    nan_check: fn(T) -> bool,
    ignore_nan: bool,
) -> (usize, usize) {
    if !ignore_nan && (nan_check(min_value) || nan_check(max_value)) {
        // --- Return NaNs
        // -> at least one of the values is NaN
        if nan_check(min_value) && nan_check(max_value) {
            // If both are NaN, return lowest index
            let lowest_index = std::cmp::min(min_index, max_index);
            return (lowest_index, lowest_index);
        } else if nan_check(min_value) {
            // If min is the only NaN, return min index
            return (min_index, min_index);
        } else {
            // If max is the only NaN, return max index
            return (max_index, max_index);
        }
    }
    (min_index, max_index)
}

/// Masked counterpart of the `*_generic` functions above: returns the (argmin, min) if
/// `MIN` and the (argmax, max) if `MAX` of the valid elements.
/// The (masked) SIMD core runs on chunks of at most `chunk_size` elements (to avoid
/// overflow), the (masked) scalar implementation runs on the remainder.
#[inline(always)]
pub(crate) fn masked_generic<T, SCALAR, const MIN: bool, const MAX: bool>(
    arr: &[T],
    validity: &[u8],
    offset: usize,
    lane_size: usize,
    chunk_size: usize,
    core_masked: unsafe fn(&[T], &[u8], usize) -> FoundMinMax<T>,
) -> FoundMinMax<T>
where
    T: Copy + PartialOrd + Bounded,
    SCALAR: ScalarArgMinMax<T>,
{
    let simd_len = arr.len() - arr.len() % lane_size;
    let (mut min, mut max) = (None, None);
    for start in (0..simd_len).step_by(chunk_size) {
        let chunk = &arr[start..simd_len.min(start + chunk_size)];
        let chunk_offset = offset + start;
        let (mut chunk_min, mut chunk_max) = unsafe { core_masked(chunk, validity, chunk_offset) };
        // The core returns the neutral value (MAX for the min, MIN for the max) only when
        // no valid value of the chunk is below (above) it, i.e., when all valid values
        // equal it or are NaN or +inf (-inf) (or there are none). Rare (e.g., saturated
        // data), but then the min (max) is the first valid element that equals it, or,
        // if there is none, the scalar implementation finds it.
        if chunk_min.is_some_and(|(_, v)| v == T::max_value()) {
            chunk_min =
                first_valid_eq(chunk, validity, chunk_offset, T::max_value()).or_else(|| {
                    scalar_masked::<T, SCALAR, true, false>(chunk, validity, chunk_offset).0
                });
        }
        if chunk_max.is_some_and(|(_, v)| v == T::min_value()) {
            chunk_max =
                first_valid_eq(chunk, validity, chunk_offset, T::min_value()).or_else(|| {
                    scalar_masked::<T, SCALAR, false, true>(chunk, validity, chunk_offset).1
                });
        }
        min = merge_min::<T, SCALAR>(min, chunk_min.map(|(i, v)| (start + i, v)));
        max = merge_max::<T, SCALAR>(max, chunk_max.map(|(i, v)| (start + i, v)));
    }
    let (rem_min, rem_max) =
        scalar_masked::<T, SCALAR, MIN, MAX>(&arr[simd_len..], validity, offset + simd_len);
    min = merge_min::<T, SCALAR>(min, rem_min.map(|(i, v)| (simd_len + i, v)));
    max = merge_max::<T, SCALAR>(max, rem_max.map(|(i, v)| (simd_len + i, v)));
    (min, max)
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

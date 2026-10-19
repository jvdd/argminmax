use std::cmp::Ordering;

use num_traits::Bounded;

use crate::scalar::ScalarArgMinMax;
use crate::validity::{first_valid_eq, merge_max, merge_min, scalar_masked, FoundMinMax};

/// A remainder (the elements after the last full vector) of at least this many elements
/// runs through the SIMD core once more, on the last `lane_size` elements of the slice.
/// That last vector overlaps elements that the first call already saw, which is fine:
/// ties, and a NaN in both results, keep the first call's result.
/// A shorter remainder runs through the scalar implementation: on 1-3 elements it is
/// faster than the extra core call (with its horizontal reduction), on 4-7 about as fast.
const MIN_SIMD_REMAINDER: usize = 8;

#[inline(always)]
pub(crate) fn argminmax_generic<T: Copy + PartialOrd>(
    arr: &[T],
    lane_size: usize,
    core_argminmax: unsafe fn(&[T]) -> (usize, T, usize, T),
    scalar_argminmax: fn(&[T]) -> (usize, usize),
    nan_check: fn(T) -> bool, // returns true if value is NaN
    ignore_nan: bool,         // if false, NaNs will be returned
) -> (usize, usize) {
    assert!(!arr.is_empty());
    if arr.len() < lane_size {
        return scalar_argminmax(arr);
    }
    let simd_len = arr.len() - arr.len() % lane_size;
    let (mut min_index, mut min_value, mut max_index, mut max_value) =
        unsafe { core_argminmax(&arr[..simd_len]) };
    if simd_len < arr.len() {
        // The remainder, see `MIN_SIMD_REMAINDER`
        let (rem_min, rem_max) = if arr.len() - simd_len < MIN_SIMD_REMAINDER {
            let (i, j) = scalar_argminmax(&arr[simd_len..]);
            let (i, j) = (simd_len + i, simd_len + j);
            ((i, arr[i]), (j, arr[j]))
        } else {
            let start = arr.len() - lane_size;
            let (i, min, j, max) = unsafe { core_argminmax(&arr[start..]) };
            ((start + i, min), (start + j, max))
        };
        (min_index, min_value) =
            find_final_index_min((min_index, min_value), rem_min, nan_check, ignore_nan);
        (max_index, max_value) =
            find_final_index_max((max_index, max_value), rem_max, nan_check, ignore_nan);
    }
    let (min_index, max_index) = get_correct_argminmax_result(
        min_index, min_value, max_index, max_value, nan_check, ignore_nan,
    );
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
    assert!(!arr.is_empty());
    if arr.len() < lane_size {
        return scalar_argmin(arr);
    }
    let simd_len = arr.len() - arr.len() % lane_size;
    let mut min = unsafe { core_argmin(&arr[..simd_len]) };
    if simd_len < arr.len() {
        // The remainder, see `MIN_SIMD_REMAINDER`
        let rem_min = if arr.len() - simd_len < MIN_SIMD_REMAINDER {
            let i = simd_len + scalar_argmin(&arr[simd_len..]);
            (i, arr[i])
        } else {
            let start = arr.len() - lane_size;
            let (i, value) = unsafe { core_argmin(&arr[start..]) };
            (start + i, value)
        };
        min = find_final_index_min(min, rem_min, nan_check, ignore_nan);
    }
    first_non_nan_if_nan(arr, min.0, nan_check, ignore_nan)
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
    assert!(!arr.is_empty());
    if arr.len() < lane_size {
        return scalar_argmax(arr);
    }
    let simd_len = arr.len() - arr.len() % lane_size;
    let mut max = unsafe { core_argmax(&arr[..simd_len]) };
    if simd_len < arr.len() {
        // The remainder, see `MIN_SIMD_REMAINDER`
        let rem_max = if arr.len() - simd_len < MIN_SIMD_REMAINDER {
            let i = simd_len + scalar_argmax(&arr[simd_len..]);
            (i, arr[i])
        } else {
            let start = arr.len() - lane_size;
            let (i, value) = unsafe { core_argmax(&arr[start..]) };
            (start + i, value)
        };
        max = find_final_index_max(max, rem_max, nan_check, ignore_nan);
    }
    first_non_nan_if_nan(arr, max.0, nan_check, ignore_nan)
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

/// Get the final index of the min value from the results of the SIMD part and the remainder
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

/// Get the final index of the max value from the results of the SIMD part and the remainder
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
/// overflow), the (masked) scalar implementation runs on the remainder (a last vector as
/// in `argminmax_generic` measured slower here).
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

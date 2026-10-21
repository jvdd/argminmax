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
    assert!(!arr.is_empty());
    if arr.len() < lane_size {
        return scalar_argminmax(arr);
    }
    let (min_index, min_value, max_index, max_value) = unsafe { core_argminmax(arr) };
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
    let min = unsafe { core_argmin(arr) };
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
    let max = unsafe { core_argmax(arr) };
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
/// overflow), the (masked) scalar implementation runs on the remainder (a second masked
/// core call on the last vector measured slower).
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

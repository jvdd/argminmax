use num_traits::AsPrimitive;
use num_traits::{Bounded, Float, One};

// ------- Generic tests for argminmax

/// The generic tests check whether the scalar and SIMD function return the same result.

#[cfg(test)]
const LONG_ARR_LEN: usize = 8193; // 8192 + 1

/// Test for a long array of random DType values whether the scalar and SIMD function
/// return the same result.
#[cfg(test)]
pub(crate) fn test_long_array_argminmax<DType>(
    get_data: fn(usize) -> Vec<DType>,
    scalar_f: fn(&[DType]) -> (usize, usize),
    simd_f: unsafe fn(&[DType]) -> (usize, usize),
) where
    DType: Copy + PartialOrd + AsPrimitive<usize>,
{
    let data: &[DType] = &get_data(LONG_ARR_LEN);
    assert_eq!(data.len() % 64, 1); // assert that data does not fully fit in a register

    let (argmin_index, argmax_index) = scalar_f(data);
    let (argmin_simd_index, argmax_simd_index) = unsafe { simd_f(data) };
    assert_eq!(argmin_index, argmin_simd_index);
    assert_eq!(argmax_index, argmax_simd_index);
}

#[cfg(test)]
const NB_RUNS: usize = 10_000;
#[cfg(test)]
const RANDOM_RUN_ARR_LEN: usize = 32 * 4 + 1;

/// Test for many arrays of random DType values whether the scalar and SIMD function
/// return the same result.
#[cfg(test)]
pub(crate) fn test_random_runs_argminmax<DType>(
    get_data: fn(usize) -> Vec<DType>,
    scalar_f: fn(&[DType]) -> (usize, usize),
    simd_f: unsafe fn(&[DType]) -> (usize, usize),
) where
    DType: Copy + PartialOrd + AsPrimitive<usize>,
{
    for _ in 0..NB_RUNS {
        let data: &[DType] = &get_data(RANDOM_RUN_ARR_LEN);
        let (argmin_index, argmax_index) = scalar_f(data);
        let (argmin_simd_index, argmax_simd_index) = unsafe { simd_f(data) };
        assert_eq!(argmin_index, argmin_simd_index);
        assert_eq!(argmax_index, argmax_simd_index);
    }
}

/// Test if the first index is returned when the MIN/MAX value occurs multiple times.
#[cfg(test)]
pub(crate) fn test_first_index_identical_values_argminmax<DType>(
    scalar_f: fn(&[DType]) -> (usize, usize),
    simd_f: unsafe fn(&[DType]) -> (usize, usize),
) where
    DType: Copy + One + Bounded,
{
    let mut data: [DType; 64] = [DType::one(); 64]; // multiple of lane size

    // Case 1: all elements are identical
    let (argmin_index, argmax_index) = scalar_f(&data);
    assert_eq!(argmin_index, 0);
    assert_eq!(argmax_index, 0);

    let (argmin_simd_index, argmax_simd_index) = unsafe { simd_f(&data) };
    assert_eq!(argmin_simd_index, 0);
    assert_eq!(argmax_simd_index, 0);

    // Case 2: all elements are identical except for a couple of MIN/MAX values
    // Add multiple MIN values to the array
    data[5] = DType::min_value();
    data[13] = DType::min_value();
    data[41] = DType::min_value();

    // Add multiple MAX values to the array
    data[7] = DType::max_value();
    data[17] = DType::max_value();
    data[31] = DType::max_value();

    let (argmin_index, argmax_index) = scalar_f(&data);
    assert_eq!(argmin_index, 5);
    assert_eq!(argmax_index, 7);

    let (argmin_simd_index, argmax_simd_index) = unsafe { simd_f(&data) };
    assert_eq!(argmin_simd_index, 5);
    assert_eq!(argmax_simd_index, 7);
}

// ------- Overflow test

/// Test wheter no overflow occurs when the array is too long.
/// The array length is 2^(size_of(DType) * 8 + 1), unless `arr_len` is specified.
#[cfg(test)]
pub(crate) fn test_no_overflow_argminmax<DType>(
    get_data: fn(usize) -> Vec<DType>,
    scalar_f: fn(&[DType]) -> (usize, usize),
    simd_f: unsafe fn(&[DType]) -> (usize, usize),
    arr_len: Option<usize>,
) where
    DType: Copy + PartialOrd + AsPrimitive<usize>,
{
    let arr_len = arr_len.unwrap_or(1 << (std::mem::size_of::<DType>() * 8 + 1));
    let data: &[DType] = &get_data(arr_len);

    let (argmin_index, argmax_index) = scalar_f(data);
    let (argmin_simd_index, argmax_simd_index) = unsafe { simd_f(data) };
    assert_eq!(argmin_index, argmin_simd_index);
    assert_eq!(argmax_index, argmax_simd_index);
}

// ------- Float tests for argminmax

#[cfg(test)]
const FLOAT_ARR_LEN: usize = 1024 + 3;

/// Test whether infinities are handled correctly.
/// -> infinities should be returned as the argmin/argmax
#[cfg(test)]
pub(crate) fn test_return_infs_argminmax<DType>(
    get_data: fn(usize) -> Vec<DType>,
    scalar_f: fn(&[DType]) -> (usize, usize),
    simd_f: unsafe fn(&[DType]) -> (usize, usize),
) where
    DType: Float + AsPrimitive<usize>,
{
    let mut data: Vec<DType> = get_data(FLOAT_ARR_LEN);
    // Case 1: all elements are +inf
    for i in 0..data.len() {
        data[i] = DType::infinity();
    }

    let (argmin_index, argmax_index) = scalar_f(&data);
    assert_eq!(argmin_index, 0);
    assert_eq!(argmax_index, 0);

    let (argmin_simd_index, argmax_simd_index) = unsafe { simd_f(&data) };
    assert_eq!(argmin_simd_index, 0);
    assert_eq!(argmax_simd_index, 0);

    // Case 2: all elements are -inf
    for i in 0..data.len() {
        data[i] = DType::neg_infinity();
    }

    let (argmin_index, argmax_index) = scalar_f(&data);
    assert_eq!(argmin_index, 0);
    assert_eq!(argmax_index, 0);

    let (argmin_simd_index, argmax_simd_index) = unsafe { simd_f(&data) };
    assert_eq!(argmin_simd_index, 0);
    assert_eq!(argmax_simd_index, 0);

    // Case 3: add some +inf and -inf in the middle
    let mut data: Vec<DType> = get_data(FLOAT_ARR_LEN);
    data[100] = DType::infinity();
    data[200] = DType::neg_infinity();

    let (argmin_index, argmax_index) = scalar_f(&data);
    assert_eq!(argmin_index, 200);
    assert_eq!(argmax_index, 100);

    let (argmin_simd_index, argmax_simd_index) = unsafe { simd_f(&data) };
    assert_eq!(argmin_simd_index, 200);
    assert_eq!(argmax_simd_index, 100);
}

/// Test whether NaNs are handled correctly - in this case, they should be ignored.
#[cfg(test)]
pub(crate) fn test_ignore_nans_argminmax<DType>(
    get_data: fn(usize) -> Vec<DType>,
    scalar_f: fn(&[DType]) -> (usize, usize),
    simd_f: unsafe fn(&[DType]) -> (usize, usize),
) where
    DType: Float + AsPrimitive<usize>,
{
    // Case 1: NaN is the first element
    let mut data: Vec<DType> = get_data(FLOAT_ARR_LEN);
    data[0] = DType::nan();

    let (argmin_index, argmax_index) = scalar_f(&data);
    assert!(argmin_index != 0);
    assert!(argmax_index != 0);

    let (argmin_simd_index, argmax_simd_index) = unsafe { simd_f(&data) };
    assert!(argmin_simd_index != 0);
    assert!(argmax_simd_index != 0);

    assert_eq!(argmin_index, argmin_simd_index);
    assert_eq!(argmax_index, argmax_simd_index);

    // Case 2: first 100 elements are NaN
    for i in 0..100 {
        data[i] = DType::nan();
    }

    let (argmin_index, argmax_index) = scalar_f(&data);
    assert!(argmin_index > 99);
    assert!(argmax_index > 99);

    let (argmin_simd_index, argmax_simd_index) = unsafe { simd_f(&data) };
    assert!(argmin_simd_index > 99);
    assert!(argmax_simd_index > 99);

    assert_eq!(argmin_index, argmin_simd_index);
    assert_eq!(argmax_index, argmax_simd_index);

    // Case 3: NaN is the last element
    let mut data: Vec<DType> = get_data(FLOAT_ARR_LEN);
    data[FLOAT_ARR_LEN - 1] = DType::nan();

    let (argmin_index, argmax_index) = scalar_f(&data);
    assert!(argmin_index != 1026);
    assert!(argmax_index != 1026);

    let (argmin_simd_index, argmax_simd_index) = unsafe { simd_f(&data) };
    assert!(argmin_simd_index != 1026);
    assert!(argmax_simd_index != 1026);

    // Case 4: last 100 elements are NaN
    for i in 0..100 {
        data[FLOAT_ARR_LEN - 1 - i] = DType::nan();
    }

    let (argmin_index, argmax_index) = scalar_f(&data);
    assert!(argmin_index < FLOAT_ARR_LEN - 100);
    assert!(argmax_index < FLOAT_ARR_LEN - 100);

    let (argmin_simd_index, argmax_simd_index) = unsafe { simd_f(&data) };
    assert!(argmin_simd_index < FLOAT_ARR_LEN - 100);
    assert!(argmax_simd_index < FLOAT_ARR_LEN - 100);

    // Case 5: NaN is somewhere in the middle element
    let mut data: Vec<DType> = get_data(FLOAT_ARR_LEN);
    data[123] = DType::nan();

    let (argmin_index, argmax_index) = scalar_f(&data);
    assert!(argmin_index != 123);
    assert!(argmax_index != 123);

    let (argmin_simd_index, argmax_simd_index) = unsafe { simd_f(&data) };
    assert!(argmin_simd_index != 123);
    assert!(argmax_simd_index != 123);

    // Case 6: all elements are NaN
    for i in 0..data.len() {
        data[i] = DType::nan();
    }

    let (argmin_index, argmax_index) = scalar_f(&data);
    assert_eq!(argmin_index, 0);
    assert_eq!(argmax_index, 0);

    let (argmin_simd_index, argmax_simd_index) = unsafe { simd_f(&data) };
    assert_eq!(argmin_simd_index, 0);
    assert_eq!(argmax_simd_index, 0);
}

/// Test whether NaNs are handled correctly - in this case, the index of the first NaN
/// should be returned.
#[cfg(test)]
pub(crate) fn test_return_nans_argminmax<DType>(
    get_data: fn(usize) -> Vec<DType>,
    scalar_f: fn(&[DType]) -> (usize, usize),
    simd_f: unsafe fn(&[DType]) -> (usize, usize),
) where
    DType: Float + AsPrimitive<usize>,
{
    // Case 1: NaN is the first element
    let mut data: Vec<DType> = get_data(FLOAT_ARR_LEN);
    data[0] = DType::nan();

    let (argmin_index, argmax_index) = scalar_f(&data);
    assert_eq!(argmin_index, 0);
    assert_eq!(argmax_index, 0);

    let (argmin_simd_index, argmax_simd_index) = unsafe { simd_f(&data) };
    assert_eq!(argmin_simd_index, 0);
    assert_eq!(argmax_simd_index, 0);

    // Case 2: first 100 elements are NaN
    for i in 0..100 {
        data[i] = DType::nan();
    }

    let (argmin_index, argmax_index) = scalar_f(&data);
    assert_eq!(argmin_index, 0);
    assert_eq!(argmax_index, 0);

    let (argmin_simd_index, argmax_simd_index) = unsafe { simd_f(&data) };
    assert_eq!(argmin_simd_index, 0);
    assert_eq!(argmax_simd_index, 0);

    // Case 3: NaN is the last element
    let mut data: Vec<DType> = get_data(FLOAT_ARR_LEN);
    data[FLOAT_ARR_LEN - 1] = DType::nan();

    let (argmin_index, argmax_index) = scalar_f(&data);
    assert_eq!(argmin_index, 1026);
    assert_eq!(argmax_index, 1026);

    let (argmin_simd_index, argmax_simd_index) = unsafe { simd_f(&data) };
    assert_eq!(argmin_simd_index, 1026);
    assert_eq!(argmax_simd_index, 1026);

    // Case 4: last 100 elements are NaN
    for i in 0..100 {
        data[FLOAT_ARR_LEN - 1 - i] = DType::nan();
    }

    let (argmin_index, argmax_index) = scalar_f(&data);
    assert_eq!(argmin_index, FLOAT_ARR_LEN - 100);
    assert_eq!(argmax_index, FLOAT_ARR_LEN - 100);

    let (argmin_simd_index, argmax_simd_index) = unsafe { simd_f(&data) };
    assert_eq!(argmin_simd_index, FLOAT_ARR_LEN - 100);
    assert_eq!(argmax_simd_index, FLOAT_ARR_LEN - 100);

    // Case 5: NaN is somewhere in the middle element
    let mut data: Vec<DType> = get_data(FLOAT_ARR_LEN);
    data[123] = DType::nan();

    let (argmin_index, argmax_index) = scalar_f(&data);
    assert_eq!(argmin_index, 123);
    assert_eq!(argmax_index, 123);

    let (argmin_simd_index, argmax_simd_index) = unsafe { simd_f(&data) };
    assert_eq!(argmin_simd_index, 123);
    assert_eq!(argmax_simd_index, 123);

    // Case 6: NaN in the middle of the array and last 100 elements are NaN
    for i in 0..100 {
        data[FLOAT_ARR_LEN - 1 - i] = DType::nan();
    }

    let (argmin_index, argmax_index) = scalar_f(&data);
    assert_eq!(argmin_index, 123);
    assert_eq!(argmax_index, 123);

    let (argmin_simd_index, argmax_simd_index) = unsafe { simd_f(&data) };
    assert_eq!(argmin_simd_index, 123);
    assert_eq!(argmax_simd_index, 123);

    // Case 7: all elements are NaN
    for i in 0..data.len() {
        data[i] = DType::nan();
    }

    let (argmin_index, argmax_index) = scalar_f(&data);
    assert_eq!(argmin_index, 0);
    assert_eq!(argmax_index, 0);

    let (argmin_simd_index, argmax_simd_index) = unsafe { simd_f(&data) };
    assert_eq!(argmin_simd_index, 0);
    assert_eq!(argmax_simd_index, 0);

    // Case 8: array exact multiple of LANE_SIZE and only 1 element is NaN
    let mut data: Vec<DType> = get_data(128);
    data[17] = DType::nan();

    let (argmin_index, argmax_index) = scalar_f(&data);
    assert_eq!(argmin_index, 17);
    assert_eq!(argmax_index, 17);

    let (argmin_simd_index, argmax_simd_index) = unsafe { simd_f(&data) };
    assert_eq!(argmin_simd_index, 17);
    assert_eq!(argmax_simd_index, 17);
}

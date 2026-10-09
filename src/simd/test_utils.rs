#[cfg(any(feature = "float", feature = "half"))]
use num_traits::float::FloatCore;
use num_traits::AsPrimitive;
use num_traits::{Bounded, One};

use crate::simd::generic::VECTORS_PER_GROUP;
use crate::{SIMDArgMinMax, ScalarArgMinMax};

// ------- Generic tests for argminmax

// The generic tests check whether the scalar and SIMD function return the same result.

#[cfg(test)]
const LONG_ARR_LEN: usize = 8193; // 8192 + 1

#[cfg(test)]
const NB_RUNS: usize = 10_000;
#[cfg(test)]
const RANDOM_RUN_ARR_LEN: usize = 32 * 4 + 1;

/// Tests whether the scalar and SIMD function return the same result.
/// - tests for a long array of random DType values whether the scalar and SIMD function
///   return the same result.
/// - tests for many arrays of random DType values whether the scalar and SIMD function
///   return the same result.
/// - tests the same for every short length, for random data and data with many ties.
#[cfg(test)]
pub(crate) fn test_return_same_result_argminmax<
    DType,
    SCALAR,
    SIMD,
    SV,
    SM,
    const LANE_SIZE: usize,
>(
    get_data: fn(usize) -> Vec<DType>,
    _scalar: SCALAR, // necessary to use SCALAR
    _simd: SIMD,     // necessary to use SIMD
) where
    DType: Copy + PartialOrd + AsPrimitive<usize> + One + Bounded,
    SV: Copy, // SIMD vector type
    SM: Copy, // SIMD mask type
    SCALAR: ScalarArgMinMax<DType>,
    SIMD: SIMDArgMinMax<DType, SV, SM, LANE_SIZE, SCALAR>,
{
    // 1. Test for a long array
    assert_eq!(LONG_ARR_LEN % 64, 1); // assert that data does not fully fit in a register
    let long_arr = std::iter::once(get_data(LONG_ARR_LEN));
    // 2. Test for many arrays
    let random_runs = std::iter::repeat_with(|| get_data(RANDOM_RUN_ARR_LEN)).take(NB_RUNS);
    // 3. Test for every length up to 16 vectors + 3 (the first vector, the groups of
    // vectors, the remaining vectors and the scalar remainder)
    let short_arrs = (1..=4 * VECTORS_PER_GROUP * LANE_SIZE + 3)
        .flat_map(|len| [get_data(len), get_ties_data(len)]);
    for data in long_arr.chain(random_runs).chain(short_arrs) {
        let data: &[DType] = &data;
        // argminmax
        let (argmin_index, argmax_index) = SCALAR::argminmax(data);
        let (argmin_simd_index, argmax_simd_index) = unsafe { SIMD::argminmax(data) };
        // argmin
        let argmin_index_single = SCALAR::argmin(data);
        let argmin_simd_index_single = unsafe { SIMD::argmin(data) };
        // argmax
        let argmax_index_single = SCALAR::argmax(data);
        let argmax_simd_index_single = unsafe { SIMD::argmax(data) };

        assert_eq!(argmin_index, argmin_simd_index);
        assert_eq!(argmin_index, argmin_index_single);
        assert_eq!(argmin_index, argmin_simd_index_single);
        assert_eq!(argmax_index, argmax_simd_index);
        assert_eq!(argmax_index, argmax_index_single);
        assert_eq!(argmax_index, argmax_simd_index_single);
    }
}

/// Data with many ties: the first index of the min / max should be returned
#[cfg(test)]
fn get_ties_data<DType: One + Bounded>(len: usize) -> Vec<DType> {
    (0..len)
        .map(|i| match (i * 7 + len) % 5 {
            0 => DType::min_value(),
            1 => DType::max_value(),
            _ => DType::one(),
        })
        .collect()
}

/// Test if the first index is returned when the MIN/MAX value occurs multiple times.
#[cfg(test)]
pub(crate) fn test_first_index_identical_values_argminmax<
    DType,
    SCALAR,
    SIMD,
    SV,
    SM,
    const LANE_SIZE: usize,
>(
    _scalar: SCALAR, // necessary to use SCALAR
    _simd: SIMD,     // necessary to use SIMD
) where
    DType: Copy + PartialOrd + AsPrimitive<usize> + One + Bounded,
    SV: Copy, // SIMD vector type
    SM: Copy, // SIMD mask type
    SCALAR: ScalarArgMinMax<DType>,
    SIMD: SIMDArgMinMax<DType, SV, SM, LANE_SIZE, SCALAR>,
{
    let mut data: [DType; 64] = [DType::one(); 64]; // multiple of lane size

    // Case 1: all elements are identical
    let (argmin_index, argmax_index) = SCALAR::argminmax(&data);
    let argmin_index_single = SCALAR::argmin(&data);
    let argmax_index_single = SCALAR::argmax(&data);
    assert_eq!(argmin_index, 0);
    assert_eq!(argmin_index_single, 0);
    assert_eq!(argmax_index, 0);
    assert_eq!(argmax_index_single, 0);

    let (argmin_simd_index, argmax_simd_index) = unsafe { SIMD::argminmax(&data) };
    let argmin_simd_index_single = unsafe { SIMD::argmin(&data) };
    let argmax_simd_index_single = unsafe { SIMD::argmax(&data) };
    assert_eq!(argmin_simd_index, 0);
    assert_eq!(argmin_simd_index_single, 0);
    assert_eq!(argmax_simd_index, 0);
    assert_eq!(argmax_simd_index_single, 0);

    // Case 2: all elements are identical except for a couple of MIN/MAX values
    // Add multiple MIN values to the array
    data[5] = DType::min_value();
    data[13] = DType::min_value();
    data[41] = DType::min_value();

    // Add multiple MAX values to the array
    data[7] = DType::max_value();
    data[17] = DType::max_value();
    data[31] = DType::max_value();

    let (argmin_index, argmax_index) = SCALAR::argminmax(&data);
    let argmin_index_single = SCALAR::argmin(&data);
    let argmax_index_single = SCALAR::argmax(&data);
    assert_eq!(argmin_index, 5);
    assert_eq!(argmin_index_single, 5);
    assert_eq!(argmax_index, 7);
    assert_eq!(argmax_index_single, 7);

    let (argmin_simd_index, argmax_simd_index) = unsafe { SIMD::argminmax(&data) };
    let argmin_simd_index_single = unsafe { SIMD::argmin(&data) };
    let argmax_simd_index_single = unsafe { SIMD::argmax(&data) };
    assert_eq!(argmin_simd_index, 5);
    assert_eq!(argmin_simd_index_single, 5);
    assert_eq!(argmax_simd_index, 7);
    assert_eq!(argmax_simd_index_single, 7);

    // Case 3: the MIN/MAX value occurs in the same lane of two vectors, for every pair
    // of vectors (covering the first vector, the groups of vectors and the remaining
    // vectors)
    for nb_vectors in 2..=16 {
        for first in 0..nb_vectors - 1 {
            for second in first + 1..nb_vectors {
                let min_lane = (first + second) % LANE_SIZE;
                let max_lane = (min_lane + 1) % LANE_SIZE;
                let mut data = vec![DType::one(); nb_vectors * LANE_SIZE];
                for vector in [first, second] {
                    data[vector * LANE_SIZE + min_lane] = DType::min_value();
                    data[vector * LANE_SIZE + max_lane] = DType::max_value();
                }
                let argmin_index = first * LANE_SIZE + min_lane;
                let argmax_index = first * LANE_SIZE + max_lane;

                let (argmin_simd_index, argmax_simd_index) = unsafe { SIMD::argminmax(&data) };
                assert_eq!(argmin_simd_index, argmin_index);
                assert_eq!(argmax_simd_index, argmax_index);
                assert_eq!(unsafe { SIMD::argmin(&data) }, argmin_index);
                assert_eq!(unsafe { SIMD::argmax(&data) }, argmax_index);
            }
        }
    }
}

// ------- Overflow test

/// Test wheter no overflow occurs when the array is too long.
/// The array length is 2^(size_of(DType) * 8 + 1), unless `arr_len` is specified.
#[cfg(test)]
pub(crate) fn test_no_overflow_argminmax<DType, SCALAR, SIMD, SV, SM, const LANE_SIZE: usize>(
    get_data: fn(usize) -> Vec<DType>,
    _scalar: SCALAR, // necessary to use SCALAR
    _simd: SIMD,     // necessary to use SIMD
    arr_len: Option<usize>,
) where
    DType: Copy + PartialOrd + AsPrimitive<usize> + One + Bounded,
    SV: Copy, // SIMD vector type
    SM: Copy, // SIMD mask type
    SCALAR: ScalarArgMinMax<DType>,
    SIMD: SIMDArgMinMax<DType, SV, SM, LANE_SIZE, SCALAR>,
{
    // left shift 1 by the number of bits in DType + 1
    let shift_size = {
        #[cfg(target_arch = "arm")] // clip shift size to 31 for armv7 (32-bit usize)
        {
            std::cmp::min(std::mem::size_of::<DType>() * 8 + 1, 31)
        }
        #[cfg(not(target_arch = "arm"))]
        {
            std::mem::size_of::<DType>() * 8 + 1 // #bits + 1
        }
    };
    let arr_len = arr_len.unwrap_or(1 << shift_size);
    let data: &[DType] = &get_data(arr_len);

    // argminmax
    let (argmin_index, argmax_index) = SCALAR::argminmax(data);
    let (argmin_simd_index, argmax_simd_index) = unsafe { SIMD::argminmax(data) };
    // argmin
    let argmin_index_single = SCALAR::argmin(data);
    let argmin_simd_index_single = unsafe { SIMD::argmin(data) };
    // argmax
    let argmax_index_single = SCALAR::argmax(data);
    let argmax_simd_index_single = unsafe { SIMD::argmax(data) };

    assert_eq!(argmin_index, argmin_simd_index);
    assert_eq!(argmin_index, argmin_index_single);
    assert_eq!(argmax_index, argmax_simd_index_single);
    assert_eq!(argmax_index, argmax_simd_index);
    assert_eq!(argmax_index, argmax_index_single);
    assert_eq!(argmin_index, argmin_simd_index_single);

    // The MIN/MAX value in the last lanes of each of the last vectors of the first chunk
    // of the overflow-safe loop (the highest indices, in the vectors after the last
    // group and in the last group), and again in the second chunk: the first
    // occurrence wins
    let chunk = SIMD::_get_overflow_lane_size_limit();
    if 2 * chunk <= arr_len {
        // (i8/u8 AVX2/AVX512 chunks hold fewer than VECTORS_PER_GROUP + 1 vectors)
        let nb_vectors = (VECTORS_PER_GROUP + 1).min(chunk / LANE_SIZE);
        for end in (0..nb_vectors).map(|v| chunk - v * LANE_SIZE) {
            let mut data = vec![DType::one(); 2 * chunk];
            for start in [0, chunk] {
                data[start + end - 2] = DType::min_value();
                data[start + end - 1] = DType::max_value();
            }
            assert_eq!(unsafe { SIMD::argminmax(&data) }, (end - 2, end - 1));
            assert_eq!(unsafe { SIMD::argmin(&data) }, end - 2);
            assert_eq!(unsafe { SIMD::argmax(&data) }, end - 1);
        }
        // The MIN/MAX value only in the last lanes of a partial second chunk (of one
        // vector and of a group of vectors and one more vector) and of a full second
        // chunk, and again in the scalar remainder: the index is offset by the chunk start
        // and the first occurrence wins
        let group_len = chunk + (VECTORS_PER_GROUP + 1) * LANE_SIZE;
        for simd_len in [chunk + LANE_SIZE, group_len, 2 * chunk] {
            let mut data = vec![DType::one(); simd_len + 3];
            for start in [simd_len - 2, simd_len] {
                data[start] = DType::min_value();
                data[start + 1] = DType::max_value();
            }
            let (min, max) = (simd_len - 2, simd_len - 1);
            assert_eq!(unsafe { SIMD::argminmax(&data) }, (min, max));
            assert_eq!(unsafe { SIMD::argmin(&data) }, min);
            assert_eq!(unsafe { SIMD::argmax(&data) }, max);
        }
    }
}

// ------- Float tests for argminmax

#[cfg(any(feature = "float", feature = "half"))]
#[cfg(test)]
const FLOAT_ARR_LEN: usize = 1024 + 3;

/// Test whether infinities are handled correctly.
/// -> infinities should be returned as the argmin/argmax
#[cfg(any(feature = "float", feature = "half"))]
#[cfg(test)]
pub(crate) fn test_return_infs_argminmax<DType, SCALAR, SIMD, SV, SM, const LANE_SIZE: usize>(
    get_data: fn(usize) -> Vec<DType>,
    _scalar: SCALAR, // necessary to use SCALAR
    _simd: SIMD,     // necessary to use SIMD
) where
    DType: FloatCore + AsPrimitive<usize>,
    SV: Copy, // SIMD vector type
    SM: Copy, // SIMD mask type
    SCALAR: ScalarArgMinMax<DType>,
    SIMD: SIMDArgMinMax<DType, SV, SM, LANE_SIZE, SCALAR>,
{
    let mut data: Vec<DType> = get_data(FLOAT_ARR_LEN);
    // Case 1: all elements are +inf
    data.fill(DType::infinity());

    let (argmin_index, argmax_index) = SCALAR::argminmax(&data);
    let argmin_index_single = SCALAR::argmin(&data);
    let argmax_index_single = SCALAR::argmax(&data);
    assert_eq!(argmin_index, 0);
    assert_eq!(argmin_index_single, 0);
    assert_eq!(argmax_index, 0);
    assert_eq!(argmax_index_single, 0);

    let (argmin_simd_index, argmax_simd_index) = unsafe { SIMD::argminmax(&data) };
    let argmin_simd_index_single = unsafe { SIMD::argmin(&data) };
    let argmax_simd_index_single = unsafe { SIMD::argmax(&data) };
    assert_eq!(argmin_simd_index, 0);
    assert_eq!(argmin_simd_index_single, 0);
    assert_eq!(argmax_simd_index, 0);
    assert_eq!(argmax_simd_index_single, 0);

    // Case 2: all elements are -inf
    data.fill(DType::neg_infinity());

    let (argmin_index, argmax_index) = SCALAR::argminmax(&data);
    let argmin_index_single = SCALAR::argmin(&data);
    let argmax_index_single = SCALAR::argmax(&data);
    assert_eq!(argmin_index, 0);
    assert_eq!(argmin_index_single, 0);
    assert_eq!(argmax_index, 0);
    assert_eq!(argmax_index_single, 0);

    let (argmin_simd_index, argmax_simd_index) = unsafe { SIMD::argminmax(&data) };
    let argmin_simd_index_single = unsafe { SIMD::argmin(&data) };
    let argmax_simd_index_single = unsafe { SIMD::argmax(&data) };
    assert_eq!(argmin_simd_index, 0);
    assert_eq!(argmin_simd_index_single, 0);
    assert_eq!(argmax_simd_index, 0);
    assert_eq!(argmax_simd_index_single, 0);

    // Case 3: add some +inf and -inf in the middle
    let mut data: Vec<DType> = get_data(FLOAT_ARR_LEN);
    data[100] = DType::infinity();
    data[200] = DType::neg_infinity();

    let (argmin_index, argmax_index) = SCALAR::argminmax(&data);
    let argmin_index_single = SCALAR::argmin(&data);
    let argmax_index_single = SCALAR::argmax(&data);
    assert_eq!(argmin_index, 200);
    assert_eq!(argmin_index_single, 200);
    assert_eq!(argmax_index, 100);
    assert_eq!(argmax_index_single, 100);

    let (argmin_simd_index, argmax_simd_index) = unsafe { SIMD::argminmax(&data) };
    let argmin_simd_index_single = unsafe { SIMD::argmin(&data) };
    let argmax_simd_index_single = unsafe { SIMD::argmax(&data) };
    assert_eq!(argmin_simd_index, 200);
    assert_eq!(argmin_simd_index_single, 200);
    assert_eq!(argmax_simd_index, 100);
    assert_eq!(argmax_simd_index_single, 100);
}

/// Test whether -0.0 and 0.0 are equal - thus the first index is returned - within and
/// across the SIMD registers and the remainder of the array.
#[cfg(any(feature = "float", feature = "half"))]
#[cfg(test)]
pub(crate) fn test_signed_zeros_argminmax<DType, SCALAR, SIMD, SV, SM, const LANE_SIZE: usize>(
    _scalar: SCALAR, // necessary to use SCALAR
    _simd: SIMD,     // necessary to use SIMD
) where
    DType: FloatCore + AsPrimitive<usize>,
    SV: Copy, // SIMD vector type
    SM: Copy, // SIMD mask type
    SCALAR: ScalarArgMinMax<DType>,
    SIMD: SIMDArgMinMax<DType, SV, SM, LANE_SIZE, SCALAR>,
{
    // The first vector, a group of vectors, one more vector and a scalar remainder
    let len = 6 * LANE_SIZE + 3;
    for (zero_i, zero_j) in [
        (DType::zero(), -DType::zero()),
        (-DType::zero(), DType::zero()),
    ] {
        for i in 0..len {
            for j in i + 1..len {
                // The zeros are the min (max) among ones (minus ones), or all values are zero
                for other in [DType::one(), -DType::one(), zero_i] {
                    let mut data = vec![other; len];
                    (data[i], data[j]) = (zero_i, zero_j);
                    let (argmin, argmax) = unsafe { SIMD::argminmax(&data) };
                    assert_eq!((argmin, argmax), SCALAR::argminmax(&data));
                    assert_eq!(argmin, unsafe { SIMD::argmin(&data) });
                    assert_eq!(argmax, unsafe { SIMD::argmax(&data) });
                    if other != -DType::one() {
                        assert_eq!(argmin, if other == zero_i { 0 } else { i });
                    }
                    if other != DType::one() {
                        assert_eq!(argmax, if other == zero_i { 0 } else { i });
                    }
                }
            }
        }
    }
}

/// Test whether NaNs are handled correctly - in this case, they should be ignored.
#[cfg(any(feature = "float", feature = "half"))]
#[cfg(test)]
pub(crate) fn test_ignore_nans_argminmax<DType, SCALAR, SIMD, SV, SM, const LANE_SIZE: usize>(
    get_data: fn(usize) -> Vec<DType>,
    _scalar: SCALAR, // necessary to use SCALAR
    _simd: SIMD,     // necessary to use SIMD
) where
    DType: FloatCore + AsPrimitive<usize>,
    SV: Copy, // SIMD vector type
    SM: Copy, // SIMD mask type
    SCALAR: ScalarArgMinMax<DType>,
    SIMD: SIMDArgMinMax<DType, SV, SM, LANE_SIZE, SCALAR>,
{
    // Test both signs, as e.g. on x86 0.0 / 0.0 returns a negative NaN
    for nan in [DType::nan(), -DType::nan()] {
        // Case 1: NaN is the first element
        let mut data: Vec<DType> = get_data(FLOAT_ARR_LEN);
        data[0] = nan;

        let (argmin_index, argmax_index) = SCALAR::argminmax(&data);
        let argmin_index_single = SCALAR::argmin(&data);
        let argmax_index_single = SCALAR::argmax(&data);
        assert!(argmin_index != 0);
        assert!(argmin_index_single != 0);
        assert!(argmax_index != 0);
        assert!(argmax_index_single != 0);

        let (argmin_simd_index, argmax_simd_index) = unsafe { SIMD::argminmax(&data) };
        let argmin_simd_index_single = unsafe { SIMD::argmin(&data) };
        let argmax_simd_index_single = unsafe { SIMD::argmax(&data) };
        assert!(argmin_simd_index != 0);
        assert!(argmin_simd_index_single != 0);
        assert!(argmax_simd_index != 0);
        assert!(argmax_simd_index_single != 0);

        assert_eq!(argmin_index, argmin_simd_index);
        assert_eq!(argmin_index, argmin_index_single);
        assert_eq!(argmin_index, argmin_simd_index_single);
        assert_eq!(argmax_index, argmax_simd_index);
        assert_eq!(argmax_index, argmax_index_single);
        assert_eq!(argmax_index, argmax_simd_index_single);

        // Case 1.1 - NaN is the first element, other values are all the same
        let mut data: Vec<DType> = get_data(FLOAT_ARR_LEN);
        data[0] = nan;
        data[1..].fill(DType::from(1.0).unwrap());

        let (argmin_index, argmax_index) = SCALAR::argminmax(&data);
        let argmin_index_single = SCALAR::argmin(&data);
        let argmax_index_single = SCALAR::argmax(&data);
        assert_eq!(argmin_index, 1);
        assert_eq!(argmin_index_single, 1);
        assert_eq!(argmax_index, 1);
        assert_eq!(argmax_index_single, 1);

        let (argmin_simd_index, argmax_simd_index) = unsafe { SIMD::argminmax(&data) };
        let argmin_simd_index_single = unsafe { SIMD::argmin(&data) };
        let argmax_simd_index_single = unsafe { SIMD::argmax(&data) };
        assert_eq!(argmin_simd_index, 1);
        assert_eq!(argmin_simd_index_single, 1);
        assert_eq!(argmax_simd_index, 1);
        assert_eq!(argmax_simd_index_single, 1);

        // Case 1.2 - NaN is the first element, other values are monotonic increasing
        let mut data: Vec<DType> = get_data(FLOAT_ARR_LEN);
        data[0] = nan;
        for (i, v) in data.iter_mut().enumerate().skip(1) {
            *v = DType::from(i as f64).unwrap();
        }

        let (argmin_index, argmax_index) = SCALAR::argminmax(&data);
        let argmin_index_single = SCALAR::argmin(&data);
        let argmax_index_single = SCALAR::argmax(&data);
        assert_eq!(argmin_index, 1);
        assert_eq!(argmin_index_single, 1);
        assert_eq!(argmax_index, FLOAT_ARR_LEN - 1);
        assert_eq!(argmax_index_single, FLOAT_ARR_LEN - 1);

        let (argmin_simd_index, argmax_simd_index) = unsafe { SIMD::argminmax(&data) };
        let argmin_simd_index_single = unsafe { SIMD::argmin(&data) };
        let argmax_simd_index_single = unsafe { SIMD::argmax(&data) };
        assert_eq!(argmin_simd_index, 1);
        assert_eq!(argmin_simd_index_single, 1);
        assert_eq!(argmax_simd_index, FLOAT_ARR_LEN - 1);
        assert_eq!(argmax_simd_index_single, FLOAT_ARR_LEN - 1);

        // Case 1.3 - NaN is the first element, other values are monotonic decreasing
        let mut data: Vec<DType> = get_data(FLOAT_ARR_LEN);
        data[0] = nan;
        for (i, v) in data.iter_mut().enumerate().skip(1) {
            *v = DType::from((FLOAT_ARR_LEN - i) as f64).unwrap();
        }

        let (argmin_index, argmax_index) = SCALAR::argminmax(&data);
        let argmin_index_single = SCALAR::argmin(&data);
        let argmax_index_single = SCALAR::argmax(&data);
        assert_eq!(argmin_index, FLOAT_ARR_LEN - 1);
        assert_eq!(argmin_index_single, FLOAT_ARR_LEN - 1);
        assert_eq!(argmax_index, 1);
        assert_eq!(argmax_index_single, 1);

        let (argmin_simd_index, argmax_simd_index) = unsafe { SIMD::argminmax(&data) };
        let argmin_simd_index_single = unsafe { SIMD::argmin(&data) };
        let argmax_simd_index_single = unsafe { SIMD::argmax(&data) };
        assert_eq!(argmin_simd_index, FLOAT_ARR_LEN - 1);
        assert_eq!(argmin_simd_index_single, FLOAT_ARR_LEN - 1);
        assert_eq!(argmax_simd_index, 1);
        assert_eq!(argmax_simd_index_single, 1);

        // Case 2: first 100 elements are NaN
        data[..100].fill(nan);

        let (argmin_index, argmax_index) = SCALAR::argminmax(&data);
        let argmin_index_single = SCALAR::argmin(&data);
        let argmax_index_single = SCALAR::argmax(&data);
        assert!(argmin_index > 99);
        assert!(argmin_index_single > 99);
        assert!(argmax_index > 99);
        assert!(argmax_index_single > 99);

        let (argmin_simd_index, argmax_simd_index) = unsafe { SIMD::argminmax(&data) };
        let argmin_simd_index_single = unsafe { SIMD::argmin(&data) };
        let argmax_simd_index_single = unsafe { SIMD::argmax(&data) };
        assert!(argmin_simd_index > 99);
        assert!(argmin_simd_index_single > 99);
        assert!(argmax_simd_index > 99);
        assert!(argmax_simd_index_single > 99);

        assert_eq!(argmin_index, argmin_simd_index);
        assert_eq!(argmin_index, argmin_index_single);
        assert_eq!(argmin_index, argmin_simd_index_single);
        assert_eq!(argmax_index, argmax_simd_index);
        assert_eq!(argmax_index, argmax_index_single);
        assert_eq!(argmax_index, argmax_simd_index_single);

        // Case 3: NaN is the last element
        let mut data: Vec<DType> = get_data(FLOAT_ARR_LEN);
        data[FLOAT_ARR_LEN - 1] = nan;

        let (argmin_index, argmax_index) = SCALAR::argminmax(&data);
        let argmin_index_single = SCALAR::argmin(&data);
        let argmax_index_single = SCALAR::argmax(&data);
        assert!(argmin_index != 1026);
        assert!(argmin_index_single != 1026);
        assert!(argmax_index != 1026);
        assert!(argmax_index_single != 1026);

        let (argmin_simd_index, argmax_simd_index) = unsafe { SIMD::argminmax(&data) };
        let argmin_simd_index_single = unsafe { SIMD::argmin(&data) };
        let argmax_simd_index_single = unsafe { SIMD::argmax(&data) };
        assert!(argmin_simd_index != 1026);
        assert!(argmin_simd_index_single != 1026);
        assert!(argmax_simd_index != 1026);
        assert!(argmax_simd_index_single != 1026);

        assert_eq!(argmin_index, argmin_simd_index);
        assert_eq!(argmin_index, argmin_index_single);
        assert_eq!(argmin_index, argmin_simd_index_single);
        assert_eq!(argmax_index, argmax_simd_index);
        assert_eq!(argmax_index, argmax_index_single);
        assert_eq!(argmax_index, argmax_simd_index_single);

        // Case 4: last 100 elements are NaN
        for i in 0..100 {
            data[FLOAT_ARR_LEN - 1 - i] = nan;
        }

        let (argmin_index, argmax_index) = SCALAR::argminmax(&data);
        let argmin_index_single = SCALAR::argmin(&data);
        let argmax_index_single = SCALAR::argmax(&data);
        assert!(argmin_index < FLOAT_ARR_LEN - 100);
        assert!(argmin_index_single < FLOAT_ARR_LEN - 100);
        assert!(argmax_index < FLOAT_ARR_LEN - 100);
        assert!(argmax_index_single < FLOAT_ARR_LEN - 100);

        let (argmin_simd_index, argmax_simd_index) = unsafe { SIMD::argminmax(&data) };
        let argmin_simd_index_single = unsafe { SIMD::argmin(&data) };
        let argmax_simd_index_single = unsafe { SIMD::argmax(&data) };
        assert!(argmin_simd_index < FLOAT_ARR_LEN - 100);
        assert!(argmin_simd_index_single < FLOAT_ARR_LEN - 100);
        assert!(argmax_simd_index < FLOAT_ARR_LEN - 100);
        assert!(argmax_simd_index_single < FLOAT_ARR_LEN - 100);

        assert_eq!(argmin_index, argmin_simd_index);
        assert_eq!(argmin_index, argmin_index_single);
        assert_eq!(argmin_index, argmin_simd_index_single);
        assert_eq!(argmax_index, argmax_simd_index);
        assert_eq!(argmax_index, argmax_index_single);
        assert_eq!(argmax_index, argmax_simd_index_single);

        // Case 5: NaN is somewhere in the middle element
        let mut data: Vec<DType> = get_data(FLOAT_ARR_LEN);
        data[123] = nan;

        let (argmin_index, argmax_index) = SCALAR::argminmax(&data);
        let argmin_index_single = SCALAR::argmin(&data);
        let argmax_index_single = SCALAR::argmax(&data);
        assert!(argmin_index != 123);
        assert!(argmin_index_single != 123);
        assert!(argmax_index != 123);
        assert!(argmax_index_single != 123);

        let (argmin_simd_index, argmax_simd_index) = unsafe { SIMD::argminmax(&data) };
        let argmin_simd_index_single = unsafe { SIMD::argmin(&data) };
        let argmax_simd_index_single = unsafe { SIMD::argmax(&data) };
        assert!(argmin_simd_index != 123);
        assert!(argmin_simd_index_single != 123);
        assert!(argmax_simd_index != 123);
        assert!(argmax_simd_index_single != 123);

        assert_eq!(argmin_index, argmin_simd_index);
        assert_eq!(argmin_index, argmin_index_single);
        assert_eq!(argmin_index, argmin_simd_index_single);
        assert_eq!(argmax_index, argmax_simd_index);
        assert_eq!(argmax_index, argmax_index_single);
        assert_eq!(argmax_index, argmax_simd_index_single);

        // Case 5.1: the only non-NaN values are +inf (or -inf), so the first non-NaN
        // index k is the argmin and the argmax - also when the NaNs fill the SIMD part
        // or (for f16) the first overflow chunk
        // (k, array length) pairs: 6 vectors, without and with a scalar remainder
        let mut cases: Vec<(usize, usize)> = [6 * LANE_SIZE, 6 * LANE_SIZE + 3]
            .into_iter()
            .flat_map(|len| (0..len).map(move |k| (k, len)))
            .collect();
        let chunk = SIMD::_get_overflow_lane_size_limit();
        if chunk < 1 << 16 {
            // only f16 (i16 indices) has a chunk that is small enough to test: k around
            // the first chunk boundary, with a full or a partial (one vector) second chunk
            for len in [2 * chunk + 3, chunk + LANE_SIZE, chunk + LANE_SIZE + 3] {
                cases.extend([chunk - 1, chunk, chunk + 1].map(|k| (k, len)));
            }
        }
        for inf in [DType::infinity(), DType::neg_infinity()] {
            for &(k, len) in &cases {
                let mut leading = vec![inf; len]; // NaNs before k
                leading[..k].fill(nan);
                let mut single = vec![nan; len]; // NaNs around k
                single[k] = inf;
                for data in [leading, single] {
                    assert_eq!(SCALAR::argminmax(&data), (k, k));
                    assert_eq!(SCALAR::argmin(&data), k);
                    assert_eq!(SCALAR::argmax(&data), k);
                    assert_eq!(unsafe { SIMD::argminmax(&data) }, (k, k));
                    assert_eq!(unsafe { SIMD::argmin(&data) }, k);
                    assert_eq!(unsafe { SIMD::argmax(&data) }, k);
                }
            }
        }

        // Case 6: all elements are NaN
        data.fill(nan);

        let (argmin_index, argmax_index) = SCALAR::argminmax(&data);
        let argmin_index_single = SCALAR::argmin(&data);
        let argmax_index_single = SCALAR::argmax(&data);
        assert_eq!(argmin_index, 0);
        assert_eq!(argmin_index_single, 0);
        assert_eq!(argmax_index, 0);
        assert_eq!(argmax_index_single, 0);

        let (argmin_simd_index, argmax_simd_index) = unsafe { SIMD::argminmax(&data) };
        let argmin_simd_index_single = unsafe { SIMD::argmin(&data) };
        let argmax_simd_index_single = unsafe { SIMD::argmax(&data) };
        assert_eq!(argmin_simd_index, 0);
        assert_eq!(argmin_simd_index_single, 0);
        assert_eq!(argmax_simd_index, 0);
        assert_eq!(argmax_simd_index_single, 0);

        // Case 6.1 - every other vector is NaN, the MIN/MAX values follow a NaN vector
        let mut data: Vec<DType> = vec![DType::one(); 16 * LANE_SIZE + 1];
        for (i, v) in data.iter_mut().enumerate() {
            if (i / LANE_SIZE) % 2 == 1 {
                *v = nan;
            }
        }
        data[2 * LANE_SIZE] = DType::min_value();
        data[2 * LANE_SIZE + 1] = DType::max_value();

        let (argmin_simd_index, argmax_simd_index) = unsafe { SIMD::argminmax(&data) };
        assert_eq!(argmin_simd_index, 2 * LANE_SIZE);
        assert_eq!(argmax_simd_index, 2 * LANE_SIZE + 1);
        assert_eq!(unsafe { SIMD::argmin(&data) }, 2 * LANE_SIZE);
        assert_eq!(unsafe { SIMD::argmax(&data) }, 2 * LANE_SIZE + 1);
    }

    // Case 7: negative zero & negative subnormal values are no NaNs
    let subnormal = DType::min_positive_value() / DType::from(2.0).unwrap();
    for value in [-DType::zero(), -subnormal] {
        let mut data: Vec<DType> = vec![DType::one(); FLOAT_ARR_LEN];
        data[123] = value;
        assert_eq!(SCALAR::argmin(&data), 123);
        assert_eq!(unsafe { SIMD::argmin(&data) }, 123);
        assert_eq!(unsafe { SIMD::argminmax(&data) }.0, 123);
    }
}

/// Test whether NaNs are handled correctly - in this case, the index of the first NaN
/// should be returned.
#[cfg(any(feature = "float", feature = "half"))]
#[cfg(test)]
pub(crate) fn test_return_nans_argminmax<DType, SCALAR, SIMD, SV, SM, const LANE_SIZE: usize>(
    get_data: fn(usize) -> Vec<DType>,
    _scalar: SCALAR, // necessary to use SCALAR
    _simd: SIMD,     // necessary to use SIMD
) where
    DType: FloatCore + AsPrimitive<usize>,
    SV: Copy, // SIMD vector type
    SM: Copy, // SIMD mask type
    SCALAR: ScalarArgMinMax<DType>,
    SIMD: SIMDArgMinMax<DType, SV, SM, LANE_SIZE, SCALAR>,
{
    // Test both signs, as e.g. on x86 0.0 / 0.0 returns a negative NaN
    for nan in [DType::nan(), -DType::nan()] {
        // Case 1: NaN is the first element
        let mut data: Vec<DType> = get_data(FLOAT_ARR_LEN);
        data[0] = nan;

        let (argmin_index, argmax_index) = SCALAR::argminmax(&data);
        let argmin_index_single = SCALAR::argmin(&data);
        let argmax_index_single = SCALAR::argmax(&data);
        assert_eq!(argmin_index, 0);
        assert_eq!(argmin_index_single, 0);
        assert_eq!(argmax_index, 0);
        assert_eq!(argmax_index_single, 0);

        let (argmin_simd_index, argmax_simd_index) = unsafe { SIMD::argminmax(&data) };
        let argmin_simd_index_single = unsafe { SIMD::argmin(&data) };
        let argmax_simd_index_single = unsafe { SIMD::argmax(&data) };
        assert_eq!(argmin_simd_index, 0);
        assert_eq!(argmin_simd_index_single, 0);
        assert_eq!(argmax_simd_index, 0);
        assert_eq!(argmax_simd_index_single, 0);

        // Case 2: first 100 elements are NaN
        data[..100].fill(nan);

        let (argmin_index, argmax_index) = SCALAR::argminmax(&data);
        let argmin_index_single = SCALAR::argmin(&data);
        let argmax_index_single = SCALAR::argmax(&data);
        assert_eq!(argmin_index, 0);
        assert_eq!(argmin_index_single, 0);
        assert_eq!(argmax_index, 0);
        assert_eq!(argmax_index_single, 0);

        let (argmin_simd_index, argmax_simd_index) = unsafe { SIMD::argminmax(&data) };
        let argmin_simd_index_single = unsafe { SIMD::argmin(&data) };
        let argmax_simd_index_single = unsafe { SIMD::argmax(&data) };
        assert_eq!(argmin_simd_index, 0);
        assert_eq!(argmin_simd_index_single, 0);
        assert_eq!(argmax_simd_index, 0);
        assert_eq!(argmax_simd_index_single, 0);

        // Case 3: NaN is the last element
        let mut data: Vec<DType> = get_data(FLOAT_ARR_LEN);
        data[FLOAT_ARR_LEN - 1] = nan;

        let (argmin_index, argmax_index) = SCALAR::argminmax(&data);
        let argmin_index_single = SCALAR::argmin(&data);
        let argmax_index_single = SCALAR::argmax(&data);
        assert_eq!(argmin_index, 1026);
        assert_eq!(argmin_index_single, 1026);
        assert_eq!(argmax_index, 1026);
        assert_eq!(argmax_index_single, 1026);

        let (argmin_simd_index, argmax_simd_index) = unsafe { SIMD::argminmax(&data) };
        let argmin_simd_index_single = unsafe { SIMD::argmin(&data) };
        let argmax_simd_index_single = unsafe { SIMD::argmax(&data) };
        assert_eq!(argmin_simd_index, 1026);
        assert_eq!(argmin_simd_index_single, 1026);
        assert_eq!(argmax_simd_index, 1026);
        assert_eq!(argmax_simd_index_single, 1026);

        // Case 4: last 100 elements are NaN
        for i in 0..100 {
            data[FLOAT_ARR_LEN - 1 - i] = nan;
        }

        let (argmin_index, argmax_index) = SCALAR::argminmax(&data);
        let argmin_index_single = SCALAR::argmin(&data);
        let argmax_index_single = SCALAR::argmax(&data);
        assert_eq!(argmin_index, FLOAT_ARR_LEN - 100);
        assert_eq!(argmin_index_single, FLOAT_ARR_LEN - 100);
        assert_eq!(argmax_index, FLOAT_ARR_LEN - 100);
        assert_eq!(argmax_index_single, FLOAT_ARR_LEN - 100);

        let (argmin_simd_index, argmax_simd_index) = unsafe { SIMD::argminmax(&data) };
        let argmin_simd_index_single = unsafe { SIMD::argmin(&data) };
        let argmax_simd_index_single = unsafe { SIMD::argmax(&data) };
        assert_eq!(argmin_simd_index, FLOAT_ARR_LEN - 100);
        assert_eq!(argmin_simd_index_single, FLOAT_ARR_LEN - 100);
        assert_eq!(argmax_simd_index, FLOAT_ARR_LEN - 100);
        assert_eq!(argmax_simd_index_single, FLOAT_ARR_LEN - 100);

        // Case 5: NaN is somewhere in the middle element
        let mut data: Vec<DType> = get_data(FLOAT_ARR_LEN);
        data[123] = nan;

        let (argmin_index, argmax_index) = SCALAR::argminmax(&data);
        let argmin_index_single = SCALAR::argmin(&data);
        let argmax_index_single = SCALAR::argmax(&data);
        assert_eq!(argmin_index, 123);
        assert_eq!(argmin_index_single, 123);
        assert_eq!(argmax_index, 123);
        assert_eq!(argmax_index_single, 123);

        let (argmin_simd_index, argmax_simd_index) = unsafe { SIMD::argminmax(&data) };
        let argmin_simd_index_single = unsafe { SIMD::argmin(&data) };
        let argmax_simd_index_single = unsafe { SIMD::argmax(&data) };
        assert_eq!(argmin_simd_index, 123);
        assert_eq!(argmin_simd_index_single, 123);
        assert_eq!(argmax_simd_index, 123);
        assert_eq!(argmax_simd_index_single, 123);

        // Case 6: NaN in the middle of the array and last 100 elements are NaN
        for i in 0..100 {
            data[FLOAT_ARR_LEN - 1 - i] = nan;
        }

        let (argmin_index, argmax_index) = SCALAR::argminmax(&data);
        let argmin_index_single = SCALAR::argmin(&data);
        let argmax_index_single = SCALAR::argmax(&data);
        assert_eq!(argmin_index, 123);
        assert_eq!(argmin_index_single, 123);
        assert_eq!(argmax_index, 123);
        assert_eq!(argmax_index_single, 123);

        let (argmin_simd_index, argmax_simd_index) = unsafe { SIMD::argminmax(&data) };
        let argmin_simd_index_single = unsafe { SIMD::argmin(&data) };
        let argmax_simd_index_single = unsafe { SIMD::argmax(&data) };
        assert_eq!(argmin_simd_index, 123);
        assert_eq!(argmin_simd_index_single, 123);
        assert_eq!(argmax_simd_index, 123);
        assert_eq!(argmax_simd_index_single, 123);

        // Case 7: all elements are NaN
        data.fill(nan);

        let (argmin_index, argmax_index) = SCALAR::argminmax(&data);
        let argmin_index_single = SCALAR::argmin(&data);
        let argmax_index_single = SCALAR::argmax(&data);
        assert_eq!(argmin_index, 0);
        assert_eq!(argmin_index_single, 0);
        assert_eq!(argmax_index, 0);
        assert_eq!(argmax_index_single, 0);

        let (argmin_simd_index, argmax_simd_index) = unsafe { SIMD::argminmax(&data) };
        let argmin_simd_index_single = unsafe { SIMD::argmin(&data) };
        let argmax_simd_index_single = unsafe { SIMD::argmax(&data) };
        assert_eq!(argmin_simd_index, 0);
        assert_eq!(argmin_simd_index_single, 0);
        assert_eq!(argmax_simd_index, 0);
        assert_eq!(argmax_simd_index_single, 0);

        // Case 8: array exact multiple of LANE_SIZE and only 1 element is NaN
        let mut data: Vec<DType> = get_data(128);
        data[17] = nan;

        let (argmin_index, argmax_index) = SCALAR::argminmax(&data);
        let argmin_index_single = SCALAR::argmin(&data);
        let argmax_index_single = SCALAR::argmax(&data);
        assert_eq!(argmin_index, 17);
        assert_eq!(argmin_index_single, 17);
        assert_eq!(argmax_index, 17);
        assert_eq!(argmax_index_single, 17);

        let (argmin_simd_index, argmax_simd_index) = unsafe { SIMD::argminmax(&data) };
        let argmin_simd_index_single = unsafe { SIMD::argmin(&data) };
        let argmax_simd_index_single = unsafe { SIMD::argmax(&data) };
        assert_eq!(argmin_simd_index, 17);
        assert_eq!(argmin_simd_index_single, 17);
        assert_eq!(argmax_simd_index, 17);
        assert_eq!(argmax_simd_index_single, 17);

        // Case 9: NaN in the first overflow chunk, and again in the last lanes of a
        // partial and of a full second chunk and in the scalar remainder: the first NaN
        // is returned (only for f16, the f32 and f64 chunks are too long to allocate)
        let chunk = SIMD::_get_overflow_lane_size_limit();
        if chunk <= 1 << 16 {
            for simd_len in [chunk + LANE_SIZE, 2 * chunk] {
                let mut data = vec![DType::one(); simd_len + 3];
                for i in [7, simd_len - 1, simd_len] {
                    data[i] = nan;
                }
                assert_eq!(unsafe { SIMD::argminmax(&data) }, (7, 7));
                assert_eq!(unsafe { SIMD::argmin(&data) }, 7);
                assert_eq!(unsafe { SIMD::argmax(&data) }, 7);
            }
        }
    }
}

// ------- Masked tests for argminmax

/// Lengths for the masked tests: around the lane sizes and the 64-bit validity words
#[cfg(test)]
const MASKED_ARR_LENS: [usize; 13] = [0, 1, 2, 7, 8, 15, 16, 17, 63, 64, 65, 129, 1027];
/// Bit offsets of the first element in the validity bitmap
#[cfg(test)]
const MASKED_OFFSETS: [usize; 5] = [0, 1, 7, 13, 64 + 5];

/// Validity bitmaps for `len` elements (starting at bit `offset`): random ones with
/// different fractions of nulls (including none and all), clustered nulls, and a single
/// valid element. The bits outside the elements are random.
#[cfg(test)]
fn get_validities(len: usize, offset: usize) -> Vec<Vec<u8>> {
    use dev_utils::utils::{get_validity, SampleUniformFullRange};
    let random: Vec<u8> = SampleUniformFullRange::get_random_array(len);
    let mut validities: Vec<Vec<u8>> = [0, 16, 64, 128, 192, 240, 252, 256]
        .iter()
        .map(|&threshold| get_validity(len, offset, |i| (random[i] as u16) < threshold))
        .collect();
    validities.push(get_validity(len, offset, |i| (i / 37) % 3 != 0));
    validities.push(get_validity(len, offset, |i| i == len / 2));
    validities.push(get_validity(len, offset, |i| i + 1 == len));
    validities
}

#[cfg(test)]
fn is_valid(validity: &[u8], offset: usize, i: usize) -> bool {
    (validity[(offset + i) / 8] >> ((offset + i) % 8)) & 1 == 1
}

/// Asserts that the masked SIMD and scalar functions return the same result, a valid
/// index (or `None` iff there are no valid elements), and the unmasked result when all
/// elements are valid.
#[cfg(test)]
fn assert_same_result_masked<DType, SCALAR, SIMD, SV, SM, const LANE_SIZE: usize>(
    data: &[DType],
    validity: &[u8],
    offset: usize,
) where
    DType: Copy + PartialOrd + AsPrimitive<usize>,
    SV: Copy, // SIMD vector type
    SM: Copy, // SIMD mask type
    SCALAR: ScalarArgMinMax<DType>,
    SIMD: SIMDArgMinMax<DType, SV, SM, LANE_SIZE, SCALAR>,
{
    let argminmax = SCALAR::argminmax_masked(data, validity, offset);
    let argmin = SCALAR::argmin_masked(data, validity, offset);
    let argmax = SCALAR::argmax_masked(data, validity, offset);
    assert_eq!(argminmax, unsafe {
        SIMD::argminmax_masked(data, validity, offset)
    });
    assert_eq!(argmin, unsafe {
        SIMD::argmin_masked(data, validity, offset)
    });
    assert_eq!(argmax, unsafe {
        SIMD::argmax_masked(data, validity, offset)
    });

    let nb_valid = (0..data.len())
        .filter(|&i| is_valid(validity, offset, i))
        .count();
    assert_eq!(argminmax.is_none(), nb_valid == 0);
    for index in [argmin, argmax].into_iter().flatten() {
        assert!(is_valid(validity, offset, index));
    }
    if nb_valid == data.len() && nb_valid > 0 {
        assert_eq!(argminmax, Some(SCALAR::argminmax(data)));
    }
}

/// Tests whether the masked scalar and SIMD functions return the same result, for many
/// lengths, validity bitmaps (and offsets) and for random data and data with many ties.
/// The null elements hold the min and max value of the data type.
#[cfg(test)]
pub(crate) fn test_return_same_result_masked_argminmax<
    DType,
    SCALAR,
    SIMD,
    SV,
    SM,
    const LANE_SIZE: usize,
>(
    get_data: fn(usize) -> Vec<DType>,
    _scalar: SCALAR, // necessary to use SCALAR
    _simd: SIMD,     // necessary to use SIMD
) where
    DType: Copy + PartialOrd + AsPrimitive<usize> + One + Bounded,
    SV: Copy, // SIMD vector type
    SM: Copy, // SIMD mask type
    SCALAR: ScalarArgMinMax<DType>,
    SIMD: SIMDArgMinMax<DType, SV, SM, LANE_SIZE, SCALAR>,
{
    for len in MASKED_ARR_LENS.into_iter().chain([LONG_ARR_LEN]) {
        for offset in MASKED_OFFSETS {
            for validity in get_validities(len, offset) {
                for mut data in [get_data(len), get_ties_data(len)] {
                    for (i, v) in data.iter_mut().enumerate() {
                        if !is_valid(&validity, offset, i) {
                            *v = [DType::min_value(), DType::max_value()][i % 2];
                        }
                    }
                    assert_same_result_masked::<_, SCALAR, SIMD, SV, SM, LANE_SIZE>(
                        &data, &validity, offset,
                    );
                    if len > 0 && (0..len).all(|i| is_valid(&validity, offset, i)) {
                        assert_eq!(
                            unsafe { SIMD::argminmax_masked(&data, &validity, offset) },
                            Some(unsafe { SIMD::argminmax(&data) })
                        );
                    }
                }
            }
        }
    }
}

/// Tests whether the masked scalar and SIMD functions return the same result for an array
/// that is longer than the SIMD index can represent (see `test_no_overflow_argminmax`),
/// and whether the SIMD functions return the expected indices around the chunk boundary.
#[cfg(test)]
pub(crate) fn test_no_overflow_masked_argminmax<
    DType,
    SCALAR,
    SIMD,
    SV,
    SM,
    const LANE_SIZE: usize,
>(
    get_data: fn(usize) -> Vec<DType>,
    _scalar: SCALAR, // necessary to use SCALAR
    _simd: SIMD,     // necessary to use SIMD
    arr_len: Option<usize>,
) where
    DType: Copy + PartialOrd + AsPrimitive<usize> + Bounded + One,
    SV: Copy, // SIMD vector type
    SM: Copy, // SIMD mask type
    SCALAR: ScalarArgMinMax<DType>,
    SIMD: SIMDArgMinMax<DType, SV, SM, LANE_SIZE, SCALAR>,
{
    use dev_utils::utils::{get_validity, SampleUniformFullRange};
    let arr_len = arr_len.unwrap_or(1 << (std::mem::size_of::<DType>() * 8 + 1));
    let data: &[DType] = &get_data(arr_len);
    let offset = 3;
    let random: Vec<u8> = SampleUniformFullRange::get_random_array(arr_len);
    // ~50% and ~6% nulls
    let half_valid = get_validity(arr_len, offset, |i| random[i] >= 128);
    let mostly_valid = get_validity(arr_len, offset, |i| random[i] >= 16);
    for validity in [half_valid, mostly_valid] {
        assert_same_result_masked::<_, SCALAR, SIMD, SV, SM, LANE_SIZE>(data, &validity, offset);
    }

    // The expected indices of the cases below are known: compare the SIMD functions with
    // them (the scalar implementation is not needed and slow on these long arrays)
    let assert_expected = |data: &[DType], validity: &[u8], offset, (min, max)| unsafe {
        let argminmax = SIMD::argminmax_masked(data, validity, offset);
        assert_eq!(argminmax, Some((min, max)));
        assert_eq!(SIMD::argmin_masked(data, validity, offset), Some(min));
        assert_eq!(SIMD::argmax_masked(data, validity, offset), Some(max));
    };
    // The MIN/MAX value in the last lanes of the last two vectors of the first chunk of
    // the SIMD loop, and again in the second chunk: the first valid occurrence wins
    let chunk = SIMD::_get_overflow_lane_size_limit();
    if 2 * chunk <= arr_len {
        // (i8/u8 AVX512 chunks hold a single vector)
        let nb_vectors = 2.min(chunk / LANE_SIZE);
        for offset in [7, 13] {
            let all_valid = get_validity(2 * chunk, offset, |_| true);
            let second_chunk = get_validity(2 * chunk, offset, |i| i >= chunk);
            for end in (0..nb_vectors).map(|v| chunk - v * LANE_SIZE) {
                let mut data = vec![DType::one(); 2 * chunk];
                for start in [0, chunk] {
                    data[start + end - 2] = DType::min_value();
                    data[start + end - 1] = DType::max_value();
                }
                assert_expected(&data, &all_valid, offset, (end - 2, end - 1));
                let expected = (chunk + end - 2, chunk + end - 1);
                assert_expected(&data, &second_chunk, offset, expected);
            }
            // Only the MAX (MIN) value, which the SIMD loop also uses for its null lanes:
            // the first valid element is returned, also when it is the last element of
            // the first chunk or the first element of the second chunk
            for first_valid in [1, chunk - 1, chunk] {
                let validity = get_validity(2 * chunk, offset, |i| i >= first_valid);
                for value in [DType::max_value(), DType::min_value()] {
                    let data = vec![value; 2 * chunk];
                    assert_expected(&data, &validity, offset, (first_valid, first_valid));
                }
            }
        }
    }
}

/// Tests whether the masked scalar and SIMD functions return the same result when there
/// are NaNs and infinities, both in the valid and the null elements.
#[cfg(any(feature = "float", feature = "half"))]
#[cfg(test)]
pub(crate) fn test_nans_masked_argminmax<DType, SCALAR, SIMD, SV, SM, const LANE_SIZE: usize>(
    get_data: fn(usize) -> Vec<DType>,
    _scalar: SCALAR, // necessary to use SCALAR
    _simd: SIMD,     // necessary to use SIMD
) where
    DType: FloatCore + AsPrimitive<usize>,
    SV: Copy, // SIMD vector type
    SM: Copy, // SIMD mask type
    SCALAR: ScalarArgMinMax<DType>,
    SIMD: SIMDArgMinMax<DType, SV, SM, LANE_SIZE, SCALAR>,
{
    // Test both signs, as e.g. on x86 0.0 / 0.0 returns a negative NaN
    for nan in [DType::nan(), -DType::nan()] {
        let null_values = [
            nan,
            DType::infinity(),
            DType::neg_infinity(),
            DType::min_value(),
            DType::max_value(),
        ];
        for len in MASKED_ARR_LENS.into_iter().chain([LONG_ARR_LEN]) {
            for offset in MASKED_OFFSETS {
                for validity in get_validities(len, offset) {
                    // NaNs at some valid elements, only NaNs, and only NaNs and infinities
                    let some_nans: Vec<DType> = (get_data(len).into_iter().enumerate())
                        .map(|(i, v)| if i % 23 == 5 { nan } else { v })
                        .collect();
                    let only_nans = vec![nan; len];
                    let nans_and_infs = (0..len)
                        .map(|i| [nan, DType::infinity(), DType::neg_infinity()][i % 3])
                        .collect();
                    for mut data in [some_nans, only_nans, nans_and_infs] {
                        for (i, v) in data.iter_mut().enumerate() {
                            if !is_valid(&validity, offset, i) {
                                *v = null_values[i % null_values.len()];
                            }
                        }
                        assert_same_result_masked::<_, SCALAR, SIMD, SV, SM, LANE_SIZE>(
                            &data, &validity, offset,
                        );
                    }
                }
            }
        }
    }
}

/// Tests the masked functions on adversarial values (signed zeros, infinities, NaNs, and
/// the min and max values), also with signed zeros on both sides of a SIMD register, the
/// remainder and an overflow-safe chunk. The SIMD and scalar results should equal the
/// scalar result on only the valid elements and, when all elements are valid and
/// `compare_unmasked_simd`, the unmasked SIMD result.
#[cfg(any(feature = "float", feature = "half"))]
#[cfg(test)]
pub(crate) fn test_adversarial_masked_argminmax<
    DType,
    SCALAR,
    SIMD,
    SV,
    SM,
    const LANE_SIZE: usize,
>(
    _scalar: SCALAR, // necessary to use SCALAR
    _simd: SIMD,     // necessary to use SIMD
    compare_unmasked_simd: bool,
) where
    DType: FloatCore + AsPrimitive<usize>,
    SV: Copy, // SIMD vector type
    SM: Copy, // SIMD mask type
    SCALAR: ScalarArgMinMax<DType>,
    SIMD: SIMDArgMinMax<DType, SV, SM, LANE_SIZE, SCALAR>,
{
    use dev_utils::utils::get_validity;
    let (zero, one) = (DType::zero(), DType::one());
    let values = [
        zero,
        -zero,
        one,
        -one,
        DType::infinity(),
        DType::neg_infinity(),
        DType::nan(),
        -DType::nan(),
        DType::min_value(),
        DType::max_value(),
    ];
    let mut state: u64 = 42; // deterministic pseudo-random numbers
    let mut random = |n: usize| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (state >> 33) as usize % n
    };
    let chunk = SIMD::_get_overflow_lane_size_limit();
    let small_chunk = (chunk < 1 << 16).then_some(chunk + 3);
    for len in (1..=2 * LANE_SIZE + 1)
        .chain([63, 64, 65, 129, 1027])
        .chain(small_chunk)
    {
        // Random values (all of them, only signed zeros, or signed zeros and ones)
        let mut datasets: Vec<Vec<DType>> = [&values[..], &values[..2], &values[..3]]
            .iter()
            .map(|palette| (0..len).map(|_| palette[random(palette.len())]).collect())
            .collect();
        // Signed zeros on both sides of a SIMD register, the remainder and a chunk
        let remainder_start = len - len % LANE_SIZE;
        for boundary in [LANE_SIZE, remainder_start, chunk]
            .into_iter()
            .filter(|&b| 0 < b && b < len)
        {
            for (zero_i, zero_j, other) in [(zero, -zero, one), (-zero, zero, -one)] {
                let mut data = vec![other; len];
                (data[boundary - 1], data[boundary]) = (zero_i, zero_j);
                datasets.push(data);
            }
        }
        for data in datasets {
            // No nulls and ~25% nulls
            for nulls in [0, 4] {
                let valid: Vec<bool> = (0..len).map(|_| nulls == 0 || random(nulls) > 0).collect();
                let validity = get_validity(len, 3, |i| valid[i]);
                let indices: Vec<usize> = (0..len).filter(|&i| valid[i]).collect();
                let valid_values: Vec<DType> = indices.iter().map(|&i| data[i]).collect();
                let (argminmax, argmin, argmax) = match valid_values.is_empty() {
                    true => (None, None, None),
                    false => {
                        let (min, max) = SCALAR::argminmax(&valid_values);
                        let (min_, max_) =
                            (SCALAR::argmin(&valid_values), SCALAR::argmax(&valid_values));
                        let index = |i: usize| indices[i];
                        (
                            Some((index(min), index(max))),
                            Some(index(min_)),
                            Some(index(max_)),
                        )
                    }
                };
                assert_eq!(argminmax, SCALAR::argminmax_masked(&data, &validity, 3));
                assert_eq!(argmin, SCALAR::argmin_masked(&data, &validity, 3));
                assert_eq!(argmax, SCALAR::argmax_masked(&data, &validity, 3));
                assert_eq!(argminmax, unsafe {
                    SIMD::argminmax_masked(&data, &validity, 3)
                });
                assert_eq!(argmin, unsafe { SIMD::argmin_masked(&data, &validity, 3) });
                assert_eq!(argmax, unsafe { SIMD::argmax_masked(&data, &validity, 3) });

                if compare_unmasked_simd && nulls == 0 {
                    let (min, max) = argminmax.unwrap();
                    assert_eq!((min, max), unsafe { SIMD::argminmax(&data) });
                    assert_eq!(min, unsafe { SIMD::argmin(&data) });
                    assert_eq!(max, unsafe { SIMD::argmax(&data) });
                }
            }
        }
    }
}

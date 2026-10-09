use argminmax::{ArgMinMax, ArgMinMaxMasked};
#[cfg(any(feature = "float", feature = "half"))]
use argminmax::{NaNArgMinMax, NaNArgMinMaxMasked};

#[cfg(feature = "half")]
use half::f16;
#[cfg(any(feature = "float", feature = "half"))]
use num_traits::float::FloatCore;
use num_traits::{AsPrimitive, FromPrimitive};

use rstest::rstest;
use rstest_reuse::{self, *};

use dev_utils::utils::{get_validity, SampleUniformFullRange};

const ARRAY_LENGTH: usize = 100_000;
const NB_RANDOM_RUNS: usize = 500;
const RANDOM_ARR_LENGTH: usize = 5_000;

// ----- dtypes_with_nan template -----

// Float and half
#[cfg(all(feature = "float", feature = "half"))]
#[template]
#[rstest]
// https://stackoverflow.com/a/3793950
#[case::float16(f16::MIN, f16::from_usize(1 << f16::MANTISSA_DIGITS).unwrap())]
#[case::float32(f32::MIN, f32::MAX)]
#[case::float64(f64::MIN, f64::MAX)]
fn dtypes_with_nan<T>(#[case] min: T, #[case] max: T) {}

// Float and not half
#[cfg(all(feature = "float", not(feature = "half")))]
#[template]
#[rstest]
#[case::float32(f32::MIN, f32::MAX)]
#[case::float64(f64::MIN, f64::MAX)]
fn dtypes_with_nan<T>(#[case] min: T, #[case] max: T) {}

// Not float and half
#[cfg(all(not(feature = "float"), feature = "half"))]
#[template]
#[rstest]
// https://stackoverflow.com/a/3793950
#[case::float16(f16::MIN, f16::from_usize(1 << f16::MANTISSA_DIGITS).unwrap())]
fn dtypes_with_nan<T>(#[case] min: T, #[case] max: T) {}

// ----- dtypes template -----

#[cfg(all(feature = "float", not(feature = "half")))]
#[template]
#[rstest]
#[case::float32(f32::MIN, f32::MAX)]
#[case::float64(f64::MIN, f64::MAX)]
#[case::int8(i8::MIN, i8::MAX)]
#[case::int16(i16::MIN, i16::MAX)]
#[case::int32(i32::MIN, i32::MAX)]
#[case::int64(i64::MIN, i64::MAX)]
#[case::int128(i128::MIN, i128::MAX)]
#[case::uint8(u8::MIN, u8::MAX)]
#[case::uint16(u16::MIN, u16::MAX)]
#[case::uint32(u32::MIN, u32::MAX)]
#[case::uint64(u64::MIN, u64::MAX)]
#[case::uint128(u128::MIN, u128::MAX)]
fn dtypes<T>(#[case] min: T, #[case] max: T) {}

#[cfg(all(feature = "float", feature = "half"))]
#[template]
#[rstest]
#[case::float16(f16::MIN, f16::from_usize(1 << f16::MANTISSA_DIGITS).unwrap())]
#[case::float32(f32::MIN, f32::MAX)]
#[case::float64(f64::MIN, f64::MAX)]
#[case::int8(i8::MIN, i8::MAX)]
#[case::int16(i16::MIN, i16::MAX)]
#[case::int32(i32::MIN, i32::MAX)]
#[case::int64(i64::MIN, i64::MAX)]
#[case::int128(i128::MIN, i128::MAX)]
#[case::uint8(u8::MIN, u8::MAX)]
#[case::uint16(u16::MIN, u16::MAX)]
#[case::uint32(u32::MIN, u32::MAX)]
#[case::uint64(u64::MIN, u64::MAX)]
#[case::uint128(u128::MIN, u128::MAX)]
fn dtypes<T>(#[case] min: T, #[case] max: T) {}

#[cfg(not(feature = "float"))]
#[template]
#[rstest]
// #[case::float16(f16::MIN, f16::MAX)] // TODO
#[case::int8(i8::MIN, i8::MAX)]
#[case::int16(i16::MIN, i16::MAX)]
#[case::int32(i32::MIN, i32::MAX)]
#[case::int64(i64::MIN, i64::MAX)]
#[case::int128(i128::MIN, i128::MAX)]
#[case::uint8(u8::MIN, u8::MAX)]
#[case::uint16(u16::MIN, u16::MAX)]
#[case::uint32(u32::MIN, u32::MAX)]
#[case::uint64(u64::MIN, u64::MAX)]
#[case::uint128(u128::MIN, u128::MAX)]
fn dtypes<T>(#[case] min: T, #[case] max: T) {}

// ----- Helpers -----

/// Returns a monotonic array of type T with length ARRAY_LENGTH and step size 1
/// The values are within the range of T and are cyclic if the range of T is smaller
/// than ARRAY_LENGTH
///
/// max_index is the max value that can be represented by T
fn get_monotonic_array<T>(n: usize, max_index: usize) -> Vec<T>
where
    T: Copy + FromPrimitive + AsPrimitive<usize>,
{
    (0..n)
        .into_iter()
        // modulo max_index to ensure that the values are within the range of T
        .map(|x| T::from_usize(x % max_index).unwrap())
        .collect::<Vec<T>>()
}

/// Returns a random validity bitmap for `len` elements that starts at bit `offset` - with
/// about `nulls` out of 256 elements null - and the validity of each element.
fn get_random_validity(len: usize, offset: usize, nulls: u16) -> (Vec<u8>, Vec<bool>) {
    let random: Vec<u8> = SampleUniformFullRange::get_random_array(len);
    let valid: Vec<bool> = random.iter().map(|&r| r as u16 >= nulls).collect();
    (get_validity(len, offset, |i| valid[i]), valid)
}

/// Straightforward reference for the masked functions: the index of the first valid value
/// that is better than all other valid values (e.g., the min for `is_better = a < b`).
/// When `ignore_nan`, NaNs are ignored (unless all valid values are NaN, then the first
/// valid index is returned), otherwise the index of the first valid NaN is returned.
fn masked_reference<T: Copy + PartialOrd>(
    data: &[T],
    valid: &[bool],
    ignore_nan: bool,
    is_better: fn(T, T) -> bool,
) -> Option<usize> {
    let is_nan = |i: &usize| data[*i].partial_cmp(&data[*i]).is_none();
    let valid_indices = (0..data.len()).filter(|&i| valid[i]);
    let first_valid = valid_indices.clone().next()?;
    if !ignore_nan {
        if let Some(first_nan) = valid_indices.clone().find(is_nan) {
            return Some(first_nan);
        }
    }
    let best = valid_indices.filter(|i| !is_nan(i)).reduce(|best, i| {
        if is_better(data[i], data[best]) {
            i
        } else {
            best
        }
    });
    Some(best.unwrap_or(first_valid))
}

// ======================================= TESTS =======================================

/// Test the ArgMinMax trait for the default implementations: slice and vec
#[cfg(test)]
#[allow(
    clippy::needless_borrow,
    clippy::unnecessary_mut_passed,
    reason = "tests the (mutably) borrowed receivers"
)]
mod default_test {
    use super::*;

    #[apply(dtypes)]
    fn test_argminmax_slice<T>(#[case] _min: T, #[case] max: T)
    where
        T: Copy + FromPrimitive + AsPrimitive<usize>,
        for<'a> &'a [T]: ArgMinMax,
    {
        // max_index is the max value that can be represented by T
        let max_index: usize = std::cmp::min(ARRAY_LENGTH, max.as_());

        let data: &[T] = &get_monotonic_array(ARRAY_LENGTH, max_index);
        // Test slice (aka the base implementation)
        let (min, max) = data.argminmax();
        assert_eq!(min, 0);
        assert_eq!(min, data.argmin());
        assert_eq!(max, max_index - 1);
        assert_eq!(max, data.argmax());
        // Borrowed slice
        let (min, max) = (&data).argminmax();
        assert_eq!(min, 0);
        assert_eq!(min, (&data).argmin());
        assert_eq!(max, max_index - 1);
        assert_eq!(max, (&data).argmax());
    }

    #[cfg(any(feature = "float", feature = "half"))]
    #[apply(dtypes_with_nan)]
    fn test_argminmax_slice_nan<T>(#[case] _min: T, #[case] max: T)
    where
        T: Copy + FromPrimitive + AsPrimitive<usize>,
        for<'a> &'a [T]: NaNArgMinMax,
    {
        // max_index is the max value that can be represented by T
        let max_index: usize = std::cmp::min(ARRAY_LENGTH, max.as_());

        let data: &[T] = &get_monotonic_array(ARRAY_LENGTH, max_index);
        // Test slice (aka the base implementation)
        let (min, max) = data.nanargminmax();
        assert_eq!(min, 0);
        assert_eq!(min, data.nanargmin());
        assert_eq!(max, max_index - 1);
        assert_eq!(max, data.nanargmax());
        // Borrowed slice
        let (min, max) = (&data).nanargminmax();
        assert_eq!(min, 0);
        assert_eq!(min, (&data).nanargmin());
        assert_eq!(max, max_index - 1);
        assert_eq!(max, (&data).nanargmax());
    }

    #[cfg(any(feature = "float", feature = "half"))]
    #[apply(dtypes_with_nan)]
    fn test_argminmax_slice_nan_and_infinities<T>(#[case] _min: T, #[case] _max: T)
    where
        T: FloatCore,
        for<'a> &'a [T]: ArgMinMax,
    {
        // The only non-NaN values are infinities: the first one is the argmin and argmax
        for inf in [T::infinity(), T::neg_infinity()] {
            let mut data: Vec<T> = vec![inf; ARRAY_LENGTH];
            data[0] = T::nan();
            let data: &[T] = &data;
            assert_eq!(data.argminmax(), (1, 1));
            assert_eq!(data.argmin(), 1);
            assert_eq!(data.argmax(), 1);
        }
    }

    // TODO: this is currently not supported yet
    // #[test]
    // fn test_argminmax_array() {
    //     // Test array
    //     let data: [f32; ARRAY_LENGTH] = (0..ARRAY_LENGTH).map(|x| x as f32).collect::<Vec<f32>>().try_into().unwrap();
    //     let (min, max) = data.argminmax();
    //     assert_eq!(min, 0);
    //     assert_eq!(max, ARRAY_LENGTH - 1);
    // }

    #[apply(dtypes)]
    fn test_argminmax_vec<T>(#[case] _min: T, #[case] max: T)
    where
        T: Copy + FromPrimitive + AsPrimitive<usize>,
        for<'a> &'a [T]: ArgMinMax,
    {
        // max_index is the max value that can be represented by T
        let max_index: usize = std::cmp::min(ARRAY_LENGTH, max.as_());

        let data: Vec<T> = get_monotonic_array(ARRAY_LENGTH, max_index);
        // Test owned vec
        let (min, max) = data.argminmax();
        assert_eq!(min, 0);
        assert_eq!(min, data.argmin());
        assert_eq!(max, max_index - 1);
        assert_eq!(max, data.argmax());
        // Test borrowed vec
        let (min, max) = (&data).argminmax();
        assert_eq!(min, 0);
        assert_eq!(min, (&data).argmin());
        assert_eq!(max, max_index - 1);
        assert_eq!(max, (&data).argmax());

        let mut data_mut: Vec<T> = get_monotonic_array(ARRAY_LENGTH, max_index);
        // Test owned mutable vec
        let (min, max) = data_mut.argminmax();
        assert_eq!(min, 0);
        assert_eq!(min, data_mut.argmin());
        assert_eq!(max, max_index - 1);
        assert_eq!(max, data_mut.argmax());
        // Test borrowed mutable vec
        let (min, max) = (&mut data_mut).argminmax();
        assert_eq!(min, 0);
        assert_eq!(min, (&mut data_mut).argmin());
        assert_eq!(max, max_index - 1);
        assert_eq!(max, (&mut data_mut).argmax());
    }

    #[cfg(any(feature = "float", feature = "half"))]
    #[apply(dtypes_with_nan)]
    fn test_argminmax_vec_nan<T>(#[case] _min: T, #[case] max: T)
    where
        T: Copy + FromPrimitive + AsPrimitive<usize>,
        for<'a> &'a [T]: NaNArgMinMax,
    {
        // max_index is the max value that can be represented by T
        let max_index: usize = std::cmp::min(ARRAY_LENGTH, max.as_());

        let data: Vec<T> = get_monotonic_array(ARRAY_LENGTH, max_index);
        // Test owned vec
        let (min, max) = data.nanargminmax();
        assert_eq!(min, 0);
        assert_eq!(min, data.nanargmin());
        assert_eq!(max, max_index - 1);
        assert_eq!(max, data.nanargmax());
        // Test borrowed vec
        let (min, max) = (&data).nanargminmax();
        assert_eq!(min, 0);
        assert_eq!(min, (&data).nanargmin());
        assert_eq!(max, max_index - 1);
        assert_eq!(max, (&data).nanargmax());
    }

    #[apply(dtypes)]
    fn test_argminmax_many_random_runs<T>(#[case] _min: T, #[case] _max: T)
    where
        T: Copy + FromPrimitive + AsPrimitive<usize> + SampleUniformFullRange,
        for<'a> &'a [T]: ArgMinMax,
    {
        for _ in 0..NB_RANDOM_RUNS {
            let data: Vec<T> = SampleUniformFullRange::get_random_array(RANDOM_ARR_LENGTH);
            // Slice
            let slice: &[T] = &data;
            let (min_slice, max_slice) = slice.argminmax();
            // Vec
            let (min_vec, max_vec) = data.argminmax();

            // Check
            assert_eq!(min_slice, min_vec);
            assert_eq!(min_slice, slice.argmin());
            assert_eq!(min_slice, data.argmin());
            assert_eq!(max_slice, max_vec);
            assert_eq!(max_slice, slice.argmax());
            assert_eq!(max_slice, data.argmax());
        }
    }
}

/// Test the ArgMinMaxMasked & NaNArgMinMaxMasked traits against a straightforward reference
#[cfg(test)]
mod masked_tests {
    use super::*;

    #[cfg(any(feature = "float", feature = "half"))]
    use num_traits::float::FloatCore;

    const LENGTHS: [usize; 7] = [0, 1, 7, 64, 65, 1_000, RANDOM_ARR_LENGTH + 3];
    const OFFSETS: [usize; 3] = [0, 3, 64 + 11];
    /// The number of nulls out of 256 elements
    const NULLS: [u16; 6] = [0, 1, 26, 128, 230, 256];

    #[apply(dtypes)]
    fn test_argminmax_masked<T>(#[case] _min: T, #[case] _max: T)
    where
        T: Copy + PartialOrd + FromPrimitive + SampleUniformFullRange,
        for<'a> &'a [T]: ArgMinMax + ArgMinMaxMasked,
    {
        for (len, offset, nulls) in LENGTHS
            .iter()
            .flat_map(|&len| OFFSETS.iter().map(move |&offset| (len, offset)))
            .flat_map(|(len, offset)| NULLS.iter().map(move |&nulls| (len, offset, nulls)))
        {
            let (validity, valid) = get_random_validity(len, offset, nulls);
            let random_data: Vec<T> = SampleUniformFullRange::get_random_array(len);
            // Many ties: the first index of the min / max should be returned
            let ties_data: Vec<T> = (0..len).map(|i| T::from_usize(i % 3).unwrap()).collect();
            for mut data in [random_data, ties_data] {
                // The null elements hold the extreme values of the data type
                for i in (0..len).filter(|&i| !valid[i]) {
                    data[i] = [T::MIN, T::MAX][i % 2];
                }
                let min = masked_reference(&data, &valid, true, |a, b| a < b);
                let max = masked_reference(&data, &valid, true, |a, b| a > b);

                let slice: &[T] = &data;
                assert_eq!(slice.argminmax_masked(&validity, offset), min.zip(max));
                assert_eq!(slice.argmin_masked(&validity, offset), min);
                assert_eq!(slice.argmax_masked(&validity, offset), max);
                assert_eq!(data.argminmax_masked(&validity, offset), min.zip(max));
                assert_eq!(data.argmin_masked(&validity, offset), min);
                assert_eq!(data.argmax_masked(&validity, offset), max);
                if nulls == 0 && len > 0 {
                    assert_eq!(min.zip(max), Some(slice.argminmax()));
                }
            }
        }
    }

    #[cfg(any(feature = "float", feature = "half"))]
    #[apply(dtypes_with_nan)]
    fn test_argminmax_masked_nans<T>(#[case] _min: T, #[case] _max: T)
    where
        T: FloatCore + SampleUniformFullRange,
        for<'a> &'a [T]: ArgMinMaxMasked + NaNArgMinMaxMasked,
    {
        let (nan, inf) = (T::nan(), T::infinity());
        for (len, offset, nulls) in LENGTHS
            .iter()
            .flat_map(|&len| OFFSETS.iter().map(move |&offset| (len, offset)))
            .flat_map(|(len, offset)| NULLS.iter().map(move |&nulls| (len, offset, nulls)))
        {
            let (validity, valid) = get_random_validity(len, offset, nulls);
            // Some NaNs, only NaNs, and only NaNs and infinities
            let some_nans: Vec<T> = (SampleUniformFullRange::get_random_array(len).into_iter())
                .enumerate()
                .map(|(i, v): (usize, T)| if i % 97 == 13 { nan } else { v })
                .collect();
            let only_nans: Vec<T> = vec![nan; len];
            let nans_and_inf: Vec<T> = (0..len).map(|i| [nan, inf][i % 2]).collect();
            // -0.0 and 0.0 as the min (max): they are equal, so the first one is returned
            let zeros_and = |other: T| -> Vec<T> {
                let values = [T::zero(), -T::zero(), other];
                (0..len).map(|i| values[(i * 7 + i / 3) % 3]).collect()
            };
            let (zeros_and_one, zeros_and_minus_one) = (zeros_and(T::one()), zeros_and(-T::one()));
            for mut data in [
                some_nans,
                only_nans,
                nans_and_inf,
                zeros_and_one,
                zeros_and_minus_one,
            ] {
                // The null elements hold NaNs, infinities and the extreme values
                for i in (0..len).filter(|&i| !valid[i]) {
                    data[i] = [nan, -inf, inf, T::MIN, T::MAX][i % 5];
                }
                let slice: &[T] = &data;

                // NaNs are ignored
                let min = masked_reference(&data, &valid, true, |a, b| a < b);
                let max = masked_reference(&data, &valid, true, |a, b| a > b);
                assert_eq!(slice.argminmax_masked(&validity, offset), min.zip(max));
                assert_eq!(slice.argmin_masked(&validity, offset), min);
                assert_eq!(slice.argmax_masked(&validity, offset), max);

                // The first NaN is returned
                let min = masked_reference(&data, &valid, false, |a, b| a < b);
                let max = masked_reference(&data, &valid, false, |a, b| a > b);
                assert_eq!(slice.nanargminmax_masked(&validity, offset), min.zip(max));
                assert_eq!(slice.nanargmin_masked(&validity, offset), min);
                assert_eq!(slice.nanargmax_masked(&validity, offset), max);
            }
        }
    }

    #[test]
    fn test_argminmax_masked_edge_cases() {
        let data: Vec<i32> = vec![5, 3, 9, 3, 9];
        // No (valid) elements
        assert_eq!((&data[..0]).argminmax_masked(&[], 0), None);
        assert_eq!(data.argminmax_masked(&[0b0000_0000], 0), None);
        // Bits after the data
        assert_eq!(data.argmin_masked(&[0b1110_0000], 0), None);
        // Bits before the data
        assert_eq!(data.argmax_masked(&[0b0000_0011], 2), None);
        // A single valid element
        assert_eq!(data.argminmax_masked(&[0b0000_0100], 0), Some((2, 2)));
        // The first index is returned on ties
        assert_eq!(data.argminmax_masked(&[0b0001_1111], 0), Some((1, 2)));
        assert_eq!(data.argminmax_masked(&[0b1111_1000, 0], 3), Some((1, 2)));
        assert_eq!(data.argminmax_masked(&[0b0001_1010], 0), Some((1, 4)));
    }

    #[test]
    #[should_panic(expected = "The validity bitmap is too short")]
    fn test_argminmax_masked_validity_too_short() {
        let data: Vec<i32> = vec![0; 9];
        data.argminmax_masked(&[0xFF], 0);
    }

    #[test]
    #[should_panic(expected = "The validity bitmap is too short")]
    fn test_argminmax_masked_offset_overflow() {
        // offset + len overflows
        let data: Vec<i32> = vec![0];
        data.argminmax_masked(&[], usize::MAX);
    }
}

/// Test the ArgMinMax trait for the ndarray implementation: Array1 and ArrayView1
#[cfg(feature = "ndarray")]
#[cfg(test)]
#[allow(
    clippy::needless_borrow,
    clippy::unnecessary_mut_passed,
    reason = "tests the (mutably) borrowed receivers"
)]
mod ndarray_tests {
    use super::*;

    use ndarray::Array1;

    #[apply(dtypes)]
    fn test_argminmax_ndarray<T>(#[case] _min: T, #[case] max: T)
    where
        T: Copy + FromPrimitive + AsPrimitive<usize>,
        for<'a> &'a [T]: ArgMinMax,
    {
        // max_index is the max value that can be represented by T
        let max_index: usize = std::cmp::min(ARRAY_LENGTH, max.as_());

        let data: Array1<T> = Array1::from(get_monotonic_array(ARRAY_LENGTH, max_index));
        // --- Array1
        // Test owned Array1
        let (min, max) = data.argminmax();
        assert_eq!(min, 0);
        assert_eq!(min, data.argmin());
        assert_eq!(max, max_index - 1);
        assert_eq!(max, data.argmax());
        // Test borrowed Array1
        let (min, max) = (&data).argminmax();
        assert_eq!(min, 0);
        assert_eq!(min, (&data).argmin());
        assert_eq!(max, max_index - 1);
        assert_eq!(max, (&data).argmax());
        // --- ArrayView1
        // Test owened ArrayView1
        let (min, max) = data.view().argminmax();
        assert_eq!(min, 0);
        assert_eq!(min, data.view().argmin());
        assert_eq!(max, max_index - 1);
        assert_eq!(max, data.view().argmax());
        // Test borrowed ArrayView1
        let (min, max) = (&data.view()).argminmax();
        assert_eq!(min, 0);
        assert_eq!(min, (&data.view()).argmin());
        assert_eq!(max, max_index - 1);
        assert_eq!(max, (&data.view()).argmax());

        let mut data_mut: Array1<T> = Array1::from(get_monotonic_array(ARRAY_LENGTH, max_index));
        // --- Array1
        // Test owned mutable Array1
        let (min, max) = data_mut.argminmax();
        assert_eq!(min, 0);
        assert_eq!(min, data_mut.argmin());
        assert_eq!(max, max_index - 1);
        assert_eq!(max, data_mut.argmax());
        // Test borrowed mutable Array1
        let (min, max) = (&mut data_mut).argminmax();
        assert_eq!(min, 0);
        assert_eq!(min, (&mut data_mut).argmin());
        assert_eq!(max, max_index - 1);
        assert_eq!(max, (&mut data_mut).argmax());
        // --- ArrayView1
        // Test owned mutable ArrayView1
        let (min, max) = data_mut.view_mut().argminmax();
        assert_eq!(min, 0);
        assert_eq!(min, data_mut.view_mut().argmin());
        assert_eq!(max, max_index - 1);
        assert_eq!(max, data_mut.view_mut().argmax());
        // Test borrowed mutable ArrayView1
        let (min, max) = (&mut data_mut.view_mut()).argminmax();
        assert_eq!(min, 0);
        assert_eq!(min, (&mut data_mut.view_mut()).argmin());
        assert_eq!(max, max_index - 1);
        assert_eq!(max, (&mut data_mut.view_mut()).argmax());
    }

    #[cfg(any(feature = "float", feature = "half"))]
    #[apply(dtypes_with_nan)]
    fn test_argminmax_ndarray_nan<T>(#[case] _min: T, #[case] max: T)
    where
        T: Copy + FromPrimitive + AsPrimitive<usize>,
        for<'a> &'a [T]: NaNArgMinMax,
    {
        // max_index is the max value that can be represented by T
        let max_index: usize = std::cmp::min(ARRAY_LENGTH, max.as_());

        let data: Array1<T> = Array1::from(get_monotonic_array(ARRAY_LENGTH, max_index));
        // --- Array1
        // Test owned Array1
        let (min, max) = data.nanargminmax();
        assert_eq!(min, 0);
        assert_eq!(min, data.nanargmin());
        assert_eq!(max, max_index - 1);
        assert_eq!(max, data.nanargmax());
        // Test borrowed Array1
        let (min, max) = (&data).nanargminmax();
        assert_eq!(min, 0);
        assert_eq!(min, (&data).nanargmin());
        assert_eq!(max, max_index - 1);
        assert_eq!(max, (&data).nanargmax());
        // --- ArrayView1
        // Test owened ArrayView1
        let (min, max) = data.view().nanargminmax();
        assert_eq!(min, 0);
        assert_eq!(min, data.view().nanargmin());
        assert_eq!(max, max_index - 1);
        assert_eq!(max, data.view().nanargmax());
        // Test borrowed ArrayView1
        let (min, max) = (&data.view()).nanargminmax();
        assert_eq!(min, 0);
        assert_eq!(min, (&data.view()).nanargmin());
        assert_eq!(max, max_index - 1);
        assert_eq!(max, (&data.view()).nanargmax());

        let mut data_mut: Array1<T> = Array1::from(get_monotonic_array(ARRAY_LENGTH, max_index));
        // --- Array1
        // Test owned mutable Array1
        let (min, max) = data_mut.nanargminmax();
        assert_eq!(min, 0);
        assert_eq!(min, data_mut.nanargmin());
        assert_eq!(max, max_index - 1);
        assert_eq!(max, data_mut.nanargmax());
        // Test borrowed mutable Array1
        let (min, max) = (&mut data_mut).nanargminmax();
        assert_eq!(min, 0);
        assert_eq!(min, (&mut data_mut).nanargmin());
        assert_eq!(max, max_index - 1);
        assert_eq!(max, (&mut data_mut).nanargmax());
        // --- ArrayView1
        // Test owned mutable ArrayView1
        let (min, max) = data_mut.view_mut().nanargminmax();
        assert_eq!(min, 0);
        assert_eq!(min, data_mut.view_mut().nanargmin());
        assert_eq!(max, max_index - 1);
        assert_eq!(max, data_mut.view_mut().nanargmax());
        // Test borrowed mutable ArrayView1
        let (min, max) = (&mut data_mut.view_mut()).nanargminmax();
        assert_eq!(min, 0);
        assert_eq!(min, (&mut data_mut.view_mut()).nanargmin());
        assert_eq!(max, max_index - 1);
        assert_eq!(max, (&mut data_mut.view_mut()).nanargmax());
    }

    #[apply(dtypes)]
    fn test_argminmax_many_random_runs_ndarray<T>(#[case] _min: T, #[case] _max: T)
    where
        T: Copy + FromPrimitive + AsPrimitive<usize> + SampleUniformFullRange,
        for<'a> &'a [T]: ArgMinMax,
    {
        for _ in 0..NB_RANDOM_RUNS {
            let data: Vec<T> = SampleUniformFullRange::get_random_array(RANDOM_ARR_LENGTH);
            // Slice
            let slice: &[T] = &data;
            let (min_slice, max_slice) = slice.argminmax();
            // Vec
            let (min_vec, max_vec) = data.argminmax();
            // Array1
            let array: Array1<T> = Array1::from_vec(slice.to_vec());
            let (min_array, max_array) = array.argminmax();

            // Check
            assert_eq!(min_slice, min_vec);
            assert_eq!(min_slice, min_array);
            assert_eq!(min_slice, slice.argmin());
            assert_eq!(min_slice, data.argmin());
            assert_eq!(min_slice, array.argmin());
            assert_eq!(max_slice, max_vec);
            assert_eq!(max_slice, max_array);
            assert_eq!(max_slice, slice.argmax());
            assert_eq!(max_slice, data.argmax());
            assert_eq!(max_slice, array.argmax());
        }
    }
}

#[cfg(feature = "arrow")]
#[cfg(test)]
#[allow(clippy::needless_borrow, reason = "tests the borrowed receivers")]
mod arrow_tests {
    use super::*;

    use arrow::array::{Array, Int32Array, PrimitiveArray};
    use arrow::buffer::NullBuffer;
    use arrow::datatypes::*;

    // The max of f16: up to 2^11 all integers are exact (https://stackoverflow.com/a/3793950)
    #[cfg(any(feature = "float", feature = "half"))]
    #[template]
    #[rstest]
    #[cfg_attr(feature = "half", case::float16(Float16Type {}, f16::MIN, f16::from_f32(2048.0)))]
    #[cfg_attr(feature = "float", case::float32(Float32Type {}, f32::MIN, f32::MAX))]
    #[cfg_attr(feature = "float", case::float64(Float64Type {}, f64::MIN, f64::MAX))]
    fn dtypes_arrow_with_nan<T, ArrowDataType>(
        #[case] _arrow_type: ArrowDataType,
        #[case] min: T,
        #[case] max: T,
    ) {
    }

    #[template]
    #[rstest]
    #[cfg_attr(feature = "half", case::float16(Float16Type {}, f16::MIN, f16::from_f32(2048.0)))]
    #[cfg_attr(feature = "float", case::float32(Float32Type {}, f32::MIN, f32::MAX))]
    #[cfg_attr(feature = "float", case::float64(Float64Type {}, f64::MIN, f64::MAX))]
    #[case::int8(Int8Type {}, i8::MIN, i8::MAX)]
    #[case::int16(Int16Type {}, i16::MIN, i16::MAX)]
    #[case::int32(Int32Type {}, i32::MIN, i32::MAX)]
    #[case::int64(Int64Type {}, i64::MIN, i64::MAX)]
    #[case::decimal128(Decimal128Type {}, i128::MIN, i128::MAX)]
    #[case::uint8(UInt8Type {}, u8::MIN, u8::MAX)]
    #[case::uint16(UInt16Type {}, u16::MIN, u16::MAX)]
    #[case::uint32(UInt32Type {}, u32::MIN, u32::MAX)]
    #[case::uint64(UInt64Type {}, u64::MIN, u64::MAX)]
    fn dtypes_arrow<T, ArrowDataType>(
        #[case] _arrow_type: ArrowDataType,
        #[case] min: T,
        #[case] max: T,
    ) {
    }

    #[apply(dtypes_arrow)]
    fn test_argminmax_arrow<T, ArrowDataType>(
        #[case] _dtype: ArrowDataType, // used to infer the arrow data type
        #[case] _min: T,
        #[case] max: T,
    ) where
        T: Copy + FromPrimitive + AsPrimitive<usize>,
        for<'a> &'a [T]: ArgMinMax + ArgMinMaxMasked,
        ArrowDataType: ArrowPrimitiveType<Native = T> + ArrowNumericType,
        PrimitiveArray<ArrowDataType>: From<Vec<T>>,
    {
        // max_index is the max value that can be represented by T
        let max_index: usize = std::cmp::min(ARRAY_LENGTH, max.as_());

        let data: PrimitiveArray<ArrowDataType> =
            PrimitiveArray::from(get_monotonic_array(ARRAY_LENGTH, max_index));
        // Test owned PrimitiveArray
        let (min, max) = data.argminmax();
        assert_eq!(min, 0);
        assert_eq!(min, data.argmin());
        assert_eq!(max, max_index - 1);
        assert_eq!(max, data.argmax());
        // Test borrowed PrimitiveArray
        let (min, max) = (&data).argminmax();
        assert_eq!(min, 0);
        assert_eq!(min, (&data).argmin());
        assert_eq!(max, max_index - 1);
        assert_eq!(max, (&data).argmax());
    }

    #[cfg(any(feature = "float", feature = "half"))]
    #[apply(dtypes_arrow_with_nan)]
    fn test_argminmax_arrow_nan<T, ArrowDataType>(
        #[case] _dtype: ArrowDataType, // used to infer the arrow data type
        #[case] _min: T,
        #[case] max: T,
    ) where
        T: Copy + FromPrimitive + AsPrimitive<usize>,
        for<'a> &'a [T]: NaNArgMinMax + NaNArgMinMaxMasked,
        ArrowDataType: ArrowPrimitiveType<Native = T> + ArrowNumericType,
        PrimitiveArray<ArrowDataType>: From<Vec<T>>,
    {
        // max_index is the max value that can be represented by T
        let max_index: usize = std::cmp::min(ARRAY_LENGTH, max.as_());

        let data: PrimitiveArray<ArrowDataType> =
            PrimitiveArray::from(get_monotonic_array(ARRAY_LENGTH, max_index));
        // Test owned PrimitiveArray
        let (min, max) = data.nanargminmax();
        assert_eq!(min, 0);
        assert_eq!(min, data.nanargmin());
        assert_eq!(max, max_index - 1);
        assert_eq!(max, data.nanargmax());
        // Test borrowed PrimitiveArray
        let (min, max) = (&data).nanargminmax();
        assert_eq!(min, 0);
        assert_eq!(min, (&data).nanargmin());
        assert_eq!(max, max_index - 1);
        assert_eq!(max, (&data).nanargmax());
    }

    #[apply(dtypes_arrow)]
    fn test_argminmax_many_random_runs_arrow<T, ArrowDataType>(
        #[case] _dtype: ArrowDataType, // used to infer the arrow data type
        #[case] _min: T,
        #[case] _max: T,
    ) where
        T: Copy + FromPrimitive + AsPrimitive<usize> + SampleUniformFullRange,
        for<'a> &'a [T]: ArgMinMax + ArgMinMaxMasked,
        ArrowDataType: ArrowPrimitiveType<Native = T> + ArrowNumericType,
        PrimitiveArray<ArrowDataType>: From<Vec<T>>,
    {
        for _ in 0..NB_RANDOM_RUNS {
            let data: Vec<T> = SampleUniformFullRange::get_random_array(RANDOM_ARR_LENGTH);
            // Slice
            let slice: &[T] = &data;
            let (min_slice, max_slice) = slice.argminmax();
            // Vec
            let (min_vec, max_vec) = data.argminmax();
            // Arrow
            let arrow: PrimitiveArray<ArrowDataType> = PrimitiveArray::from(data.clone());
            let (min_arrow, max_arrow) = arrow.argminmax();

            // Check
            assert_eq!(min_slice, min_vec);
            assert_eq!(min_slice, min_arrow);
            assert_eq!(min_slice, slice.argmin());
            assert_eq!(min_slice, data.argmin());
            assert_eq!(min_slice, arrow.argmin());
            assert_eq!(max_slice, max_vec);
            assert_eq!(max_slice, max_arrow);
            assert_eq!(max_slice, slice.argmax());
            assert_eq!(max_slice, data.argmax());
            assert_eq!(max_slice, arrow.argmax());
        }
    }

    /// Returns an array with the given values and validity. Unlike `From<Vec<Option<T>>>`,
    /// which stores 0 in the null elements, this keeps their values.
    fn with_nulls<A: ArrowPrimitiveType>(data: &[A::Native], valid: &[bool]) -> PrimitiveArray<A> {
        PrimitiveArray::new(data.to_vec().into(), Some(NullBuffer::from(valid)))
    }

    /// Asserts that `f` panics because all values are null
    fn assert_all_null_panic<R>(f: impl FnOnce() -> R) {
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f))
            .err()
            .expect("no panic");
        let message = (panic.downcast_ref::<String>().map(String::as_str))
            .or(panic.downcast_ref::<&str>().copied());
        assert_eq!(message, Some("All values are null"));
    }

    #[test]
    fn test_argminmax_arrow_edge_cases() {
        let arrow = Int32Array::from(vec![Some(1), None, Some(5), Some(3), Some(7), None]);
        // A slice keeps its null buffer, also when that has no nulls
        let no_nulls = arrow.slice(2, 3); // [5, 3, 7]
        assert!(no_nulls
            .nulls()
            .is_some_and(|nulls| nulls.null_count() == 0));
        assert_eq!(no_nulls.argminmax(), (1, 2));
        // The indices are relative to the (repeatedly) sliced array
        let sliced = arrow.slice(1, 5).slice(1, 4); // [5, 3, 7, null]
        assert_eq!(sliced.argminmax(), (1, 2));
        // An empty array panics (with or without a null buffer)
        let empty = Int32Array::from(Vec::<i32>::new());
        assert!(std::panic::catch_unwind(|| empty.argmin()).is_err());
        let empty = arrow.slice(1, 0);
        assert!(std::panic::catch_unwind(|| empty.argmin()).is_err());
    }

    #[test]
    fn test_argmin_null_count_check() {
        // The documented way to get `None` (instead of a panic) when all values are null
        let argmin =
            |arrow: &Int32Array| (arrow.null_count() < arrow.len()).then(|| arrow.argmin());
        assert_eq!(argmin(&Int32Array::from(vec![None, None])), None);
        assert_eq!(argmin(&Int32Array::from(Vec::<i32>::new())), None);
        assert_eq!(
            argmin(&Int32Array::from(vec![None, Some(3), Some(1)])),
            Some(2)
        );
    }

    #[apply(dtypes_arrow)]
    fn test_argminmax_arrow_nulls<T, ArrowDataType>(
        #[case] _dtype: ArrowDataType, // used to infer the arrow data type
        #[case] min: T,
        #[case] max: T,
    ) where
        T: Copy + PartialOrd + SampleUniformFullRange,
        for<'a> &'a [T]: ArgMinMax + ArgMinMaxMasked,
        ArrowDataType: ArrowPrimitiveType<Native = T> + ArrowNumericType,
    {
        for nulls in [1, 128, 230] {
            let mut data: Vec<T> = SampleUniformFullRange::get_random_array(RANDOM_ARR_LENGTH);
            let (_, valid) = get_random_validity(RANDOM_ARR_LENGTH, 0, nulls);
            // The null elements hold the extreme values of the data type
            for i in (0..RANDOM_ARR_LENGTH).filter(|&i| !valid[i]) {
                data[i] = [min, max][i % 2];
            }
            let arrow: PrimitiveArray<ArrowDataType> = with_nulls(&data, &valid);
            // The validity bitmap of a slice has an offset
            for offset in [0, 13] {
                let arrow = arrow.slice(offset, RANDOM_ARR_LENGTH - offset);
                let (data, valid) = (&data[offset..], &valid[offset..]);
                let min = masked_reference(data, valid, true, |a, b| a < b).unwrap();
                let max = masked_reference(data, valid, true, |a, b| a > b).unwrap();
                assert_eq!(arrow.argminmax(), (min, max));
                assert_eq!(arrow.argmin(), min);
                assert_eq!(arrow.argmax(), max);
            }
        }
        // Only nulls: panics (as for an empty array)
        let arrow: PrimitiveArray<ArrowDataType> = with_nulls(&[max; 100], &[false; 100]);
        assert_all_null_panic(|| arrow.argminmax());
        assert_all_null_panic(|| arrow.argmin());
        assert_all_null_panic(|| arrow.argmax());
    }

    #[cfg(any(feature = "float", feature = "half"))]
    #[apply(dtypes_arrow_with_nan)]
    fn test_argminmax_arrow_nulls_nan<T, ArrowDataType>(
        #[case] _dtype: ArrowDataType, // used to infer the arrow data type
        #[case] _min: T,
        #[case] _max: T,
    ) where
        T: FloatCore + SampleUniformFullRange,
        for<'a> &'a [T]: ArgMinMax + ArgMinMaxMasked + NaNArgMinMax + NaNArgMinMaxMasked,
        ArrowDataType: ArrowPrimitiveType<Native = T> + ArrowNumericType,
    {
        let (nan, inf) = (T::nan(), T::infinity());
        let mut data: Vec<T> = SampleUniformFullRange::get_random_array(RANDOM_ARR_LENGTH);
        let (_, mut valid) = get_random_validity(RANDOM_ARR_LENGTH, 0, 128);
        // The null elements hold NaNs, infinities and the extreme values
        for i in (0..RANDOM_ARR_LENGTH).filter(|&i| !valid[i]) {
            data[i] = [nan, -inf, inf, T::MIN, T::MAX][i % 5];
        }
        // A null NaN before a valid NaN
        (data[20], valid[20]) = (nan, false);
        (data[30], valid[30]) = (nan, true);
        let arrow: PrimitiveArray<ArrowDataType> = with_nulls(&data, &valid);
        for offset in [0, 13] {
            let arrow = arrow.slice(offset, RANDOM_ARR_LENGTH - offset);
            // NaNs are ignored
            let (data, valid) = (&data[offset..], &valid[offset..]);
            let min = masked_reference(data, valid, true, |a, b| a < b).unwrap();
            let max = masked_reference(data, valid, true, |a, b| a > b).unwrap();
            assert_eq!(arrow.argminmax(), (min, max));
            assert_eq!(arrow.argmin(), min);
            assert_eq!(arrow.argmax(), max);
            // The first valid NaN is returned
            assert_eq!(arrow.nanargminmax(), (30 - offset, 30 - offset));
            assert_eq!(arrow.nanargmin(), 30 - offset);
            assert_eq!(arrow.nanargmax(), 30 - offset);
        }
        // Only nulls: panics (as for an empty array)
        let arrow: PrimitiveArray<ArrowDataType> = with_nulls(&[T::nan(); 100], &[false; 100]);
        assert_all_null_panic(|| arrow.nanargminmax());
        assert_all_null_panic(|| arrow.nanargmin());
        assert_all_null_panic(|| arrow.nanargmax());
    }
}

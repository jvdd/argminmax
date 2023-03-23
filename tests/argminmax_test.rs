use argminmax::ArgMinMax;
#[cfg(any(feature = "float", feature = "half"))]
use argminmax::NaNArgMinMax;

#[cfg(feature = "half")]
use half::f16;
use num_traits::{AsPrimitive, FromPrimitive};

use rstest::rstest;
use rstest_reuse::{self, *};

use dev_utils::utils;
use rand;

const ARRAY_LENGTH: usize = 100_000;

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

#[cfg(feature = "float")]
#[template]
#[rstest]
// #[case::float16(f16::MIN, f16::MAX)] // TODO
#[case::float32(f32::MIN, f32::MAX)]
#[case::float64(f64::MIN, f64::MAX)]
#[case::int8(i8::MIN, i8::MAX)]
#[case::int16(i16::MIN, i16::MAX)]
#[case::int32(i32::MIN, i32::MAX)]
#[case::int64(i64::MIN, i64::MAX)]
#[case::uint8(u8::MIN, u8::MAX)]
#[case::uint16(u16::MIN, u16::MAX)]
#[case::uint32(u32::MIN, u32::MAX)]
#[case::uint64(u64::MIN, u64::MAX)]
fn dtypes<T>(#[case] min: T, #[case] max: T) {}

#[cfg(not(feature = "float"))]
#[template]
#[rstest]
// #[case::float16(f16::MIN, f16::MAX)] // TODO
#[case::int8(i8::MIN, i8::MAX)]
#[case::int16(i16::MIN, i16::MAX)]
#[case::int32(i32::MIN, i32::MAX)]
#[case::int64(i64::MIN, i64::MAX)]
#[case::uint8(u8::MIN, u8::MAX)]
#[case::uint16(u16::MIN, u16::MAX)]
#[case::uint32(u32::MIN, u32::MAX)]
#[case::uint64(u64::MIN, u64::MAX)]
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

// ======================================= TESTS =======================================

/// Test the ArgMinMax trait for the default implementations: slice and vec
#[cfg(test)]
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
        assert_eq!(max, max_index - 1);
        // Borrowed slice
        let (min, max) = (&data).argminmax();
        assert_eq!(min, 0);
        assert_eq!(max, max_index - 1);
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
        assert_eq!(max, max_index - 1);
        // Borrowed slice
        let (min, max) = (&data).nanargminmax();
        assert_eq!(min, 0);
        assert_eq!(max, max_index - 1);
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
        assert_eq!(max, max_index - 1);
        // Test borrowed vec
        let (min, max) = (&data).argminmax();
        assert_eq!(min, 0);
        assert_eq!(max, max_index - 1);

        let mut data_mut: Vec<T> = get_monotonic_array(ARRAY_LENGTH, max_index);
        // Test owned mutable vec
        let (min, max) = data_mut.argminmax();
        assert_eq!(min, 0);
        assert_eq!(max, max_index - 1);
        // Test borrowed mutable vec
        let (min, max) = (&mut data_mut).argminmax();
        assert_eq!(min, 0);
        assert_eq!(max, max_index - 1);
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
        assert_eq!(max, max_index - 1);
        // Test borrowed vec
        let (min, max) = (&data).nanargminmax();
        assert_eq!(min, 0);
        assert_eq!(max, max_index - 1);
    }

    #[apply(dtypes)]
    fn test_argminmax_many_random_runs<T>(#[case] min: T, #[case] max: T)
    where
        T: Copy + FromPrimitive + AsPrimitive<usize> + rand::distributions::uniform::SampleUniform,
        for<'a> &'a [T]: ArgMinMax,
    {
        for _ in 0..500 {
            let data: Vec<T> = utils::get_random_array::<T>(5_000, min, max);
            // Slice
            let slice: &[T] = &data;
            let (min_slice, max_slice) = slice.argminmax();
            // Vec
            let (min_vec, max_vec) = data.argminmax();

            // Check
            assert_eq!(min_slice, min_vec);
            assert_eq!(max_slice, max_vec);
        }
    }
}

/// Test the ArgMinMax trait for the ndarray implementation: Array1 and ArrayView1
#[cfg(feature = "ndarray")]
#[cfg(test)]
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
        assert_eq!(max, max_index - 1);
        // Test borrowed Array1
        let (min, max) = (&data).argminmax();
        assert_eq!(min, 0);
        assert_eq!(max, max_index - 1);
        // --- ArrayView1
        // Test owened ArrayView1
        let (min, max) = data.view().argminmax();
        assert_eq!(min, 0);
        assert_eq!(max, max_index - 1);
        // Test borrowed ArrayView1
        let (min, max) = (&data.view()).argminmax();
        assert_eq!(min, 0);
        assert_eq!(max, max_index - 1);

        let mut data_mut: Array1<T> = Array1::from(get_monotonic_array(ARRAY_LENGTH, max_index));
        // --- Array1
        // Test owned mutable Array1
        let (min, max) = data_mut.argminmax();
        assert_eq!(min, 0);
        assert_eq!(max, max_index - 1);
        // Test borrowed mutable Array1
        let (min, max) = (&mut data_mut).argminmax();
        assert_eq!(min, 0);
        assert_eq!(max, max_index - 1);
        // --- ArrayView1
        // Test owned mutable ArrayView1
        let (min, max) = data_mut.view_mut().argminmax();
        assert_eq!(min, 0);
        assert_eq!(max, max_index - 1);
        // Test borrowed mutable ArrayView1
        let (min, max) = (&mut data_mut.view_mut()).argminmax();
        assert_eq!(min, 0);
        assert_eq!(max, max_index - 1);
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
        assert_eq!(max, max_index - 1);
        // Test borrowed Array1
        let (min, max) = (&data).nanargminmax();
        assert_eq!(min, 0);
        assert_eq!(max, max_index - 1);
        // --- ArrayView1
        // Test owened ArrayView1
        let (min, max) = data.view().nanargminmax();
        assert_eq!(min, 0);
        assert_eq!(max, max_index - 1);
        // Test borrowed ArrayView1
        let (min, max) = (&data.view()).nanargminmax();
        assert_eq!(min, 0);
        assert_eq!(max, max_index - 1);

        let mut data_mut: Array1<T> = Array1::from(get_monotonic_array(ARRAY_LENGTH, max_index));
        // --- Array1
        // Test owned mutable Array1
        let (min, max) = data_mut.nanargminmax();
        assert_eq!(min, 0);
        assert_eq!(max, max_index - 1);
        // Test borrowed mutable Array1
        let (min, max) = (&mut data_mut).nanargminmax();
        assert_eq!(min, 0);
        assert_eq!(max, max_index - 1);
        // --- ArrayView1
        // Test owned mutable ArrayView1
        let (min, max) = data_mut.view_mut().nanargminmax();
        assert_eq!(min, 0);
        assert_eq!(max, max_index - 1);
        // Test borrowed mutable ArrayView1
        let (min, max) = (&mut data_mut.view_mut()).nanargminmax();
        assert_eq!(min, 0);
        assert_eq!(max, max_index - 1);
    }

    #[apply(dtypes)]
    fn test_argminmax_many_random_runs_ndarray<T>(#[case] min: T, #[case] max: T)
    where
        T: Copy + FromPrimitive + AsPrimitive<usize> + rand::distributions::uniform::SampleUniform,
        for<'a> &'a [T]: ArgMinMax,
    {
        for _ in 0..500 {
            let data: Vec<T> = utils::get_random_array::<T>(5_000, min, max);
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
            assert_eq!(max_slice, max_vec);
            assert_eq!(min_slice, min_array);
            assert_eq!(max_slice, max_array);
        }
    }
}

#[cfg(feature = "arrow")]
#[cfg(test)]
mod arrow_tests {
    use super::*;

    use arrow::array::PrimitiveArray;
    use arrow::datatypes::*;

    #[cfg(feature = "float")]
    #[template]
    #[rstest]
    #[case::float32(Float32Type {}, f32::MIN, f32::MAX)]
    #[case::float64(Float64Type {}, f64::MIN, f64::MAX)]
    fn dtypes_arrow_with_nan<T, ArrowDataType>(
        #[case] _arrow_type: ArrowDataType,
        #[case] min: T,
        #[case] max: T,
    ) {
    }

    #[cfg(feature = "float")]
    #[template]
    #[rstest]
    #[case::float32(Float32Type {}, f32::MIN, f32::MAX)]
    #[case::float64(Float64Type {}, f64::MIN, f64::MAX)]
    #[case::int8(Int8Type {}, i8::MIN, i8::MAX)]
    #[case::int16(Int16Type {}, i16::MIN, i16::MAX)]
    #[case::int32(Int32Type {}, i32::MIN, i32::MAX)]
    #[case::int64(Int64Type {}, i64::MIN, i64::MAX)]
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

    #[cfg(not(feature = "float"))]
    #[template]
    #[case::int8(Int8Type {}, i8::MIN, i8::MAX)]
    #[case::int16(Int16Type {}, i16::MIN, i16::MAX)]
    #[case::int32(Int32Type {}, i32::MIN, i32::MAX)]
    #[case::int64(Int64Type {}, i64::MIN, i64::MAX)]
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
        for<'a> &'a [T]: ArgMinMax,
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
        assert_eq!(max, max_index - 1);
        // Test borrowed PrimitiveArray
        let (min, max) = (&data).argminmax();
        assert_eq!(min, 0);
        assert_eq!(max, max_index - 1);
    }

    #[cfg(feature = "float")]
    #[apply(dtypes_arrow_with_nan)]
    fn test_argminmax_arrow_nan<T, ArrowDataType>(
        #[case] _dtype: ArrowDataType, // used to infer the arrow data type
        #[case] _min: T,
        #[case] max: T,
    ) where
        T: Copy + FromPrimitive + AsPrimitive<usize>,
        for<'a> &'a [T]: NaNArgMinMax,
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
        assert_eq!(max, max_index - 1);
        // Test borrowed PrimitiveArray
        let (min, max) = (&data).nanargminmax();
        assert_eq!(min, 0);
        assert_eq!(max, max_index - 1);
    }

    #[apply(dtypes_arrow)]
    fn test_argminmax_many_random_runs_arrow<T, ArrowDataType>(
        #[case] _dtype: ArrowDataType, // used to infer the arrow data type
        #[case] min: T,
        #[case] max: T,
    ) where
        T: Copy + FromPrimitive + AsPrimitive<usize> + rand::distributions::uniform::SampleUniform,
        for<'a> &'a [T]: ArgMinMax,
        ArrowDataType: ArrowPrimitiveType<Native = T> + ArrowNumericType,
        PrimitiveArray<ArrowDataType>: From<Vec<T>>,
    {
        for _ in 0..500 {
            let data: Vec<T> = utils::get_random_array::<T>(5_000, min, max);
            // Slice
            let slice: &[T] = &data;
            let (min_slice, max_slice) = slice.argminmax();
            // Vec
            let (min_vec, max_vec) = data.argminmax();
            // Arrow
            let arrow: PrimitiveArray<ArrowDataType> = PrimitiveArray::from(data);
            let (min_arrow, max_arrow) = arrow.argminmax();

            // Check
            assert_eq!(min_slice, min_vec);
            assert_eq!(max_slice, max_vec);
            assert_eq!(min_slice, min_arrow);
            assert_eq!(max_slice, max_arrow);
        }
    }
}

//! Implementation of the scalar argminmax operations for the 16-bit floats f16 and bf16.
//!
//! As these are not hardware supported on most x86 CPUs, we aim to facilitate efficient
//! implementation of argminmax operations on f16 and bf16 arrays through transforming the
//! values to i16ord. (more details in simd/simd_half_return_nan.rs)
//!
use half::{bf16, f16};
use num_traits::{float::FloatCore, AsPrimitive, Bounded};

/// The 16-bit floats of the half crate (f16 and bf16). Both store a sign bit followed by
/// the magnitude, so both map to i16ord in the same way; only the bits of +inf differ.
// Without SIMD (arm without the nightly_simd feature), INF_BITS and from_bits are unused
#[cfg_attr(
    not(any(
        target_arch = "x86",
        target_arch = "x86_64",
        all(target_arch = "arm", feature = "nightly_simd"),
        target_arch = "aarch64",
    )),
    allow(dead_code)
)]
pub(crate) trait Half: FloatCore + AsPrimitive<usize> + Bounded {
    /// The bits of +inf: magnitudes (absolute i16ord values) above this are NaN
    const INF_BITS: i16;
    fn to_bits(self) -> u16;
    fn from_bits(bits: u16) -> Self;
}

impl Half for f16 {
    const INF_BITS: i16 = 0x7C00;
    fn to_bits(self) -> u16 {
        f16::to_bits(self)
    }
    fn from_bits(bits: u16) -> Self {
        f16::from_bits(bits)
    }
}

impl Half for bf16 {
    const INF_BITS: i16 = 0x7F80;
    fn to_bits(self) -> u16 {
        bf16::to_bits(self)
    }
    fn from_bits(bits: u16) -> Self {
        bf16::from_bits(bits)
    }
}

/// The order-preserving map between the bits of a 16-bit float (sign and magnitude, as
/// i16) and i16ord: v if v >= 0, else i16::MIN - v (minus the magnitude). The map is
/// its own inverse, except that -0.0 and 0.0 are both mapped to 0 (as these are equal).
#[inline(always)]
fn i16ord_map(v: i16) -> i16 {
    if v < 0 {
        i16::MIN - v
    } else {
        v
    }
}

#[inline(always)]
fn half_to_i16ord<T: Half>(x: T) -> i16 {
    i16ord_map(x.to_bits() as i16)
}

/// The inverse of `half_to_i16ord` (0.0 is returned for -0.0)
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "arm", feature = "nightly_simd"),
    target_arch = "aarch64",
))]
#[inline(always)]
pub(crate) fn i16ord_to_half<T: Half>(ord_i16: i16) -> T {
    T::from_bits(i16ord_map(ord_i16) as u16)
}

// ------- Float Return NaN -------

// TODO: commented this (see the TODO below)
// #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
// #[inline(never)]
pub(crate) fn scalar_argminmax_half_return_nan<T: Half>(arr: &[T]) -> (usize, usize) {
    // The values are transformed to i16ord
    //   benchmarks  show:
    //     1. this is 7-10x faster than using raw f16
    //     2. this is 3x faster than transforming to f32 or f64
    assert!(!arr.is_empty());
    let mut low_index: usize = 0;
    let mut high_index: usize = 0;
    // It is remarkably faster to iterate over the index and use get_unchecked
    // than using .iter().enumerate() (with a fold).
    let mut low: i16 = half_to_i16ord(unsafe { *arr.get_unchecked(low_index) });
    let mut high: i16 = half_to_i16ord(unsafe { *arr.get_unchecked(high_index) });
    for i in 0..arr.len() {
        let v: T = unsafe { *arr.get_unchecked(i) };
        if v.is_nan() {
            // Return the index of the first NaN value
            return (i, i);
        }
        let v: i16 = half_to_i16ord(v);
        if v < low {
            low = v;
            low_index = i;
        } else if v > high {
            high = v;
            high_index = i;
        }
    }
    (low_index, high_index)
}

pub(crate) fn scalar_argmin_half_return_nan<T: Half>(arr: &[T]) -> usize {
    // The values are transformed to i16ord
    //   benchmarks  show:
    //     1. this is 7-10x faster than using raw f16
    //     2. this is 3x faster than transforming to f32 or f64
    assert!(!arr.is_empty());
    let mut low_index: usize = 0;
    // It is remarkably faster to iterate over the index and use get_unchecked
    // than using .iter().enumerate() (with a fold).
    let mut low: i16 = half_to_i16ord(unsafe { *arr.get_unchecked(low_index) });
    for i in 0..arr.len() {
        let v: T = unsafe { *arr.get_unchecked(i) };
        if v.is_nan() {
            // Return the index of the first NaN value
            return i;
        }
        let v: i16 = half_to_i16ord(v);
        if v < low {
            low = v;
            low_index = i;
        }
    }
    low_index
}

pub(crate) fn scalar_argmax_half_return_nan<T: Half>(arr: &[T]) -> usize {
    // The values are transformed to i16ord
    //   benchmarks  show:
    //     1. this is 7-10x faster than using raw f16
    //     2. this is 3x faster than transforming to f32 or f64
    assert!(!arr.is_empty());
    let mut high_index: usize = 0;
    // It is remarkably faster to iterate over the index and use get_unchecked
    // than using .iter().enumerate() (with a fold).
    let mut high: i16 = half_to_i16ord(unsafe { *arr.get_unchecked(high_index) });
    for i in 0..arr.len() {
        let v: T = unsafe { *arr.get_unchecked(i) };
        if v.is_nan() {
            // Return the index of the first NaN value
            return i;
        }
        let v: i16 = half_to_i16ord(v);
        if v > high {
            high = v;
            high_index = i;
        }
    }
    high_index
}

pub(crate) fn scalar_argminmax_half_ignore_nan<T: Half>(arr: &[T]) -> (usize, usize) {
    // The values are transformed to i16ord
    //   benchmarks  show:
    //     1. this is 7-10x faster than using raw f16
    //     2. this is 3x faster than transforming to f32 or f64
    assert!(!arr.is_empty());
    // Start from the first non-NaN value, so that the loop does not have to check
    // whether a non-NaN value was seen
    let Some(start) = arr.iter().position(|v| !v.is_nan()) else {
        return (0, 0); // all values are NaN
    };
    let mut low_index: usize = start;
    let mut high_index: usize = start;
    let mut low: i16 = half_to_i16ord(arr[start]);
    let mut high: i16 = low;
    // It is remarkably faster to iterate over the index and use get_unchecked
    // than using .iter().enumerate() (with a fold). The loop starts at `start` (not
    // `start + 1`), as LLVM does not unroll it on aarch64 otherwise.
    for i in start..arr.len() {
        let v: T = unsafe { *arr.get_unchecked(i) };
        if v.is_nan() {
            // v is NaN, ignore it (do nothing)
        } else {
            // v is not NaN
            let v: i16 = half_to_i16ord(v);
            if v < low {
                low = v;
                low_index = i;
            } else if v > high {
                high = v;
                high_index = i;
            }
        }
    }
    (low_index, high_index)
}

pub(crate) fn scalar_argmin_half_ignore_nan<T: Half>(arr: &[T]) -> usize {
    // The values are transformed to i16ord
    //   benchmarks  show:
    //     1. this is 7-10x faster than using raw f16
    //     2. this is 3x faster than transforming to f32 or f64
    assert!(!arr.is_empty());
    // Start from the first non-NaN value, so that the loop does not have to check
    // whether a non-NaN value was seen
    let Some(start) = arr.iter().position(|v| !v.is_nan()) else {
        return 0; // all values are NaN
    };
    let mut low_index: usize = start;
    let mut low: i16 = half_to_i16ord(arr[start]);
    // It is remarkably faster to iterate over the index and use get_unchecked
    // than using .iter().enumerate() (with a fold). The loop starts at `start` (not
    // `start + 1`), as LLVM does not unroll it on aarch64 otherwise.
    for i in start..arr.len() {
        let v: T = unsafe { *arr.get_unchecked(i) };
        if v.is_nan() {
            // v is NaN, ignore it (do nothing)
        } else {
            // v is not NaN
            let v: i16 = half_to_i16ord(v);
            if v < low {
                low = v;
                low_index = i;
            }
        }
    }
    low_index
}

pub(crate) fn scalar_argmax_half_ignore_nan<T: Half>(arr: &[T]) -> usize {
    // The values are transformed to i16ord
    //   benchmarks  show:
    //     1. this is 7-10x faster than using raw f16
    //     2. this is 3x faster than transforming to f32 or f64
    assert!(!arr.is_empty());
    // Start from the first non-NaN value, so that the loop does not have to check
    // whether a non-NaN value was seen
    let Some(start) = arr.iter().position(|v| !v.is_nan()) else {
        return 0; // all values are NaN
    };
    let mut high_index: usize = start;
    let mut high: i16 = half_to_i16ord(arr[start]);
    // It is remarkably faster to iterate over the index and use get_unchecked
    // than using .iter().enumerate() (with a fold). The loop starts at `start` (not
    // `start + 1`), as LLVM does not unroll it on aarch64 otherwise.
    for i in start..arr.len() {
        let v: T = unsafe { *arr.get_unchecked(i) };
        if v.is_nan() {
            // v is NaN, ignore it (do nothing)
        } else {
            // v is not NaN
            let v: i16 = half_to_i16ord(v);
            if v > high {
                high = v;
                high_index = i;
            }
        }
    }
    high_index
}

// TODO: previously we had dedicated non x86_64 code for f16 (see below)

// #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
// // #[inline(never)]
// pub(crate) fn scalar_argminmax_f16(arr: &[f16]) -> (usize, usize) {
//     // The values are transformed to i16ord
//     //   benchmarks  show:
//     //     1. this is 7-10x faster than using raw f16
//     //     2. this is 3x faster than transforming to f32 or f64
//     assert!(!arr.is_empty());
//     // This is 3% slower on x86_64, but 12% faster on aarch64.
//     let minmax_tuple: (usize, i16, usize, i16) = arr.iter().enumerate().fold(
//         (0, half_to_i16ord(arr[0]), 0, half_to_i16ord(arr[0])),
//         |(low_index, low, high_index, high), (i, item)| {
//             if item.is_nan() {
//                 // Return the index of the first NaN value
//                 return (i, i);
//             }
//             let item = half_to_i16ord(*item);
//             if item < low {
//                 (i, item, high_index, high)
//             } else if item > high {
//                 (low_index, low, i, item)
//             } else {
//                 (low_index, low, high_index, high)
//             }
//         },
//     );
//     (minmax_tuple.0, minmax_tuple.2)
// }

// ======================================= TESTS =======================================

#[cfg(all(feature = "float", feature = "half"))]
#[cfg(test)]
mod tests {
    use super::Half;
    use super::{
        scalar_argmax_half_ignore_nan, scalar_argmin_half_ignore_nan,
        scalar_argminmax_half_ignore_nan,
    };
    use super::{
        scalar_argmax_half_return_nan, scalar_argmin_half_return_nan,
        scalar_argminmax_half_return_nan,
    };
    use crate::{FloatIgnoreNaN, FloatReturnNaN, ScalarArgMinMax, SCALAR};

    use half::{bf16, f16};
    use rstest::rstest;

    use dev_utils::utils::SampleUniformFullRange;

    const ARR_LEN: usize = 1025;

    fn get_arrays<T: Half + SampleUniformFullRange>(len: usize) -> (Vec<f32>, Vec<T>) {
        let vec_half: Vec<T> = T::get_random_array(len);
        let vec_f32: Vec<f32> = vec_half.iter().map(|x| x.to_f32().unwrap()).collect();
        (vec_f32, vec_half)
    }

    #[rstest]
    #[case::f16(f16::ZERO)]
    #[case::bf16(bf16::ZERO)]
    fn test_generic_and_specific_impl_return_the_same_results<T: Half + SampleUniformFullRange>(
        #[case] _dtype: T,
    ) {
        for _ in 0..100 {
            let (vec_f32, vec_half) = get_arrays(ARR_LEN);
            let data_f32: &[f32] = &vec_f32;
            let data_half: &[T] = &vec_half;
            // Return NaN
            let (argmin_index, argmax_index) = SCALAR::<FloatReturnNaN>::argminmax(data_f32);
            let (argmin_index_half, argmax_index_half) =
                scalar_argminmax_half_return_nan(data_half);
            let argmin_index_half_single = scalar_argmin_half_return_nan(data_half);
            let argmax_index_half_single = scalar_argmax_half_return_nan(data_half);
            assert_eq!(argmin_index, argmin_index_half);
            assert_eq!(argmax_index, argmax_index_half_single);
            assert_eq!(argmax_index, argmax_index_half);
            assert_eq!(argmin_index, argmin_index_half_single);
            // Ignore NaN
            let (argmin_index, argmax_index) = SCALAR::<FloatIgnoreNaN>::argminmax(data_f32);
            let (argmin_index_half, argmax_index_half) =
                scalar_argminmax_half_ignore_nan(data_half);
            let argmin_index_half_single = scalar_argmin_half_ignore_nan(data_half);
            let argmax_index_half_single = scalar_argmax_half_ignore_nan(data_half);
            assert_eq!(argmin_index, argmin_index_half);
            assert_eq!(argmin_index, argmin_index_half_single);
            assert_eq!(argmax_index, argmax_index_half);
            assert_eq!(argmax_index, argmax_index_half_single);
        }
    }

    #[rstest]
    #[case::f16(f16::ZERO)]
    #[case::bf16(bf16::ZERO)]
    fn test_generic_and_specific_impl_return_nans<T: Half + SampleUniformFullRange>(
        #[case] _dtype: T,
    ) {
        // first, middle, last element
        let nan_pos: [usize; 3] = [0, ARR_LEN / 2, ARR_LEN - 1];
        for pos in nan_pos.iter() {
            let (vec_f32, vec_half) = get_arrays(ARR_LEN);
            let mut data_f32: Vec<f32> = vec_f32;
            let mut data_half: Vec<T> = vec_half;
            data_f32[*pos] = f32::NAN;
            data_half[*pos] = T::nan();
            let (argmin_index, argmax_index) = SCALAR::<FloatReturnNaN>::argminmax(&data_f32);
            let (argmin_index_half, argmax_index_half) =
                scalar_argminmax_half_return_nan(&data_half);
            let argmin_index_half_single = scalar_argmin_half_return_nan(&data_half);
            let argmax_index_half_single = scalar_argmax_half_return_nan(&data_half);
            assert_eq!(argmin_index, argmin_index_half);
            assert_eq!(argmin_index, argmin_index_half_single);
            assert_eq!(argmax_index, argmax_index_half);
            assert_eq!(argmax_index, argmax_index_half_single);
        }

        // all elements are NaN
        let (mut vec_f32, mut vec_half) = get_arrays(ARR_LEN);
        vec_f32.iter_mut().for_each(|x| *x = f32::NAN);
        vec_half.iter_mut().for_each(|x| *x = T::nan());
        let data_f32: &[f32] = &vec_f32;
        let data_half: &[T] = &vec_half;
        let (argmin_index, argmax_index) = SCALAR::<FloatReturnNaN>::argminmax(data_f32);
        let (argmin_index_half, argmax_index_half) = scalar_argminmax_half_return_nan(data_half);
        let argmin_index_half_single = scalar_argmin_half_return_nan(data_half);
        let argmax_index_half_single = scalar_argmax_half_return_nan(data_half);
        assert_eq!(argmin_index, argmin_index_half);
        assert_eq!(argmin_index, argmin_index_half_single);
        assert_eq!(argmax_index, argmax_index_half);
        assert_eq!(argmax_index, argmax_index_half_single);
        assert_eq!(argmin_index, 0);
        assert_eq!(argmax_index, 0);
    }

    #[rstest]
    #[case::f16(f16::ZERO)]
    #[case::bf16(bf16::ZERO)]
    fn test_generic_and_specific_impl_ignore_nans<T: Half + SampleUniformFullRange>(
        #[case] _dtype: T,
    ) {
        // first, middle, last element
        let nan_pos: [usize; 3] = [0, ARR_LEN / 2, ARR_LEN - 1];
        for pos in nan_pos.iter() {
            let (vec_f32, vec_half) = get_arrays(ARR_LEN);
            let mut data_f32: Vec<f32> = vec_f32;
            let mut data_half: Vec<T> = vec_half;
            data_f32[*pos] = f32::NAN;
            data_half[*pos] = T::nan();
            let (argmin_index, argmax_index) = SCALAR::<FloatIgnoreNaN>::argminmax(&data_f32);
            let (argmin_index_half, argmax_index_half) =
                scalar_argminmax_half_ignore_nan(&data_half);
            let argmin_index_half_single = scalar_argmin_half_ignore_nan(&data_half);
            let argmax_index_half_single = scalar_argmax_half_ignore_nan(&data_half);
            assert_eq!(argmin_index, argmin_index_half);
            assert_eq!(argmin_index, argmin_index_half_single);
            assert_eq!(argmax_index, argmax_index_half);
            assert_eq!(argmax_index, argmax_index_half_single);
        }

        // all elements are NaN
        let (mut vec_f32, mut vec_half) = get_arrays(ARR_LEN);
        vec_f32.iter_mut().for_each(|x| *x = f32::NAN);
        vec_half.iter_mut().for_each(|x| *x = T::nan());
        let data_f32: &[f32] = &vec_f32;
        let data_half: &[T] = &vec_half;
        let (argmin_index, argmax_index) = SCALAR::<FloatIgnoreNaN>::argminmax(data_f32);
        let (argmin_index_half, argmax_index_half) = scalar_argminmax_half_ignore_nan(data_half);
        let argmin_index_half_single = scalar_argmin_half_ignore_nan(data_half);
        let argmax_index_half_single = scalar_argmax_half_ignore_nan(data_half);
        assert_eq!(argmin_index, argmin_index_half);
        assert_eq!(argmin_index, argmin_index_half_single);
        assert_eq!(argmax_index, argmax_index_half);
        assert_eq!(argmax_index, argmax_index_half_single);
        assert_eq!(argmin_index, 0);
        assert_eq!(argmax_index, 0);

        // only NaNs and infinities (the first one is NaN)
        for inf in [f32::INFINITY, f32::NEG_INFINITY] {
            let data_f32: Vec<f32> = vec![f32::NAN, inf, inf, f32::NAN];
            let data_half: Vec<T> = data_f32.iter().map(|&x| T::from(x).unwrap()).collect();
            let (argmin_index, argmax_index) = SCALAR::<FloatIgnoreNaN>::argminmax(&data_f32);
            assert_eq!((argmin_index, argmax_index), (1, 1));
            assert_eq!(scalar_argminmax_half_ignore_nan(&data_half), (1, 1));
            assert_eq!(scalar_argmin_half_ignore_nan(&data_half), 1);
            assert_eq!(scalar_argmax_half_ignore_nan(&data_half), 1);
        }
    }
}

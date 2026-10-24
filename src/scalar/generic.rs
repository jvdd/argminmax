#[cfg(feature = "float")]
use num_traits::float::FloatCore;
use num_traits::PrimInt;

use super::super::dtype_strategy::Int;
/// The DTypeStrategy for which we implement the ScalarArgMinMax trait
#[cfg(any(feature = "float", feature = "half"))]
use super::super::dtype_strategy::{FloatIgnoreNaN, FloatReturnNaN};
use crate::validity::{assert_validity_len, scalar_masked};

/// Helper trait to initialize the min and max values & check if we should return
/// This will be implemented for all:
/// - ints - Int DTypeStrategy (see 1st impl block below)
/// - uints - Int DTypeStrategy (see 1st impl block below)
/// - floats: returning NaNs - FloatReturnNan DTypeStrategy (see 2nd impl block below)
/// - floats: ignoring NaNs - FloatIgnoreNaN DTypeStrategy (see 3rd impl block below)
///
trait SCALARInit<ScalarDType: Copy + PartialOrd> {
    const _RETURN_AT_NAN: bool;

    /// The index of the value that the min and max values start from (below `arr.len()`
    /// for a non-empty `arr`: the loops read it unchecked)
    fn _start_index(arr: &[ScalarDType]) -> usize;

    /// Nan check
    fn _nan_check(v: ScalarDType) -> bool;
}

/// A trait providing the scalar implementation of the argminmax operations.
///
// This trait will be implemented for the different DTypeStrategy
pub trait ScalarArgMinMax<ScalarDType: Copy + PartialOrd> {
    /// Get the index of the minimum and maximum values in the slice.
    ///
    /// # Arguments
    /// - `data` - the slice of data.
    ///
    /// # Returns
    /// A tuple of the index of the minimum and maximum values in the slice
    /// `(min_index, max_index)`.
    ///
    fn argminmax(data: &[ScalarDType]) -> (usize, usize);

    /// Get the index of the minimum value in the slice.
    ///
    /// # Arguments
    /// - `data` - the slice of data.
    ///
    /// # Returns
    /// The index of the minimum value in the slice.
    ///
    fn argmin(data: &[ScalarDType]) -> usize;

    /// Get the index of the maximum value in the slice.
    ///
    /// # Arguments
    /// - `data` - the slice of data.
    ///
    /// # Returns
    /// The index of the maximum value in the slice.
    ///
    fn argmax(data: &[ScalarDType]) -> usize;

    /// Get the index of the minimum and maximum values in the slice, skipping the null
    /// elements.
    ///
    /// # Arguments
    /// - `data` - the slice of data.
    /// - `validity` - the validity bitmap: element `i` is valid (not null) iff bit
    ///   `offset + i` is set (in the [Arrow format](https://arrow.apache.org/docs/format/Columnar.html#validity-bitmaps)).
    /// - `offset` - the bit offset of the first element in the validity bitmap.
    ///
    /// # Returns
    /// A tuple of the index of the minimum and maximum valid values in the slice
    /// `(min_index, max_index)`, or `None` when there are no valid values.
    ///
    /// # Panics
    /// When the validity bitmap has less than `offset + data.len()` bits.
    ///
    fn argminmax_masked(
        data: &[ScalarDType],
        validity: &[u8],
        offset: usize,
    ) -> Option<(usize, usize)>
    where
        Self: Sized,
    {
        assert_validity_len(validity, offset, data.len());
        let (min, max) = scalar_masked::<_, Self, true, true>(data, validity, offset);
        Some((min?.0, max?.0))
    }

    /// Get the index of the minimum value in the slice, skipping the null elements.
    ///
    /// See [`argminmax_masked`](ScalarArgMinMax::argminmax_masked) for the arguments and
    /// the panics.
    ///
    /// # Returns
    /// The index of the minimum valid value in the slice, or `None` when there are no
    /// valid values.
    ///
    fn argmin_masked(data: &[ScalarDType], validity: &[u8], offset: usize) -> Option<usize>
    where
        Self: Sized,
    {
        assert_validity_len(validity, offset, data.len());
        let (min, _) = scalar_masked::<_, Self, true, false>(data, validity, offset);
        Some(min?.0)
    }

    /// Get the index of the maximum value in the slice, skipping the null elements.
    ///
    /// See [`argminmax_masked`](ScalarArgMinMax::argminmax_masked) for the arguments and
    /// the panics.
    ///
    /// # Returns
    /// The index of the maximum valid value in the slice, or `None` when there are no
    /// valid values.
    ///
    fn argmax_masked(data: &[ScalarDType], validity: &[u8], offset: usize) -> Option<usize>
    where
        Self: Sized,
    {
        assert_validity_len(validity, offset, data.len());
        let (_, max) = scalar_masked::<_, Self, false, true>(data, validity, offset);
        Some(max?.0)
    }
}

/// Type that implements the [ScalarArgMinMax](crate::ScalarArgMinMax) trait.
///
/// This struct implements the ScalarArgMinMax trait for the different data types and their [datatype strategies](crate::dtype_strategy).
///
// See the impl_scalar! macro below for the implementation of the ScalarArgMinMax trait
pub struct SCALAR<DTypeStrategy> {
    pub(crate) _dtype_strategy: std::marker::PhantomData<DTypeStrategy>,
}

// ------- Implement the SCALARInit trait for the different DTypeStrategy -------

impl<ScalarDType> SCALARInit<ScalarDType> for SCALAR<Int>
where
    ScalarDType: PrimInt,
{
    const _RETURN_AT_NAN: bool = false;

    #[inline(always)]
    fn _start_index(_arr: &[ScalarDType]) -> usize {
        0
    }

    #[inline(always)]
    fn _nan_check(_v: ScalarDType) -> bool {
        false
    }
}

#[cfg(feature = "float")]
impl<ScalarDType> SCALARInit<ScalarDType> for SCALAR<FloatReturnNaN>
where
    ScalarDType: FloatCore,
{
    const _RETURN_AT_NAN: bool = true;

    #[inline(always)]
    fn _start_index(_arr: &[ScalarDType]) -> usize {
        0
    }

    #[inline(always)]
    fn _nan_check(v: ScalarDType) -> bool {
        v.is_nan()
    }
}

#[cfg(feature = "float")]
impl<ScalarDType> SCALARInit<ScalarDType> for SCALAR<FloatIgnoreNaN>
where
    ScalarDType: FloatCore,
{
    const _RETURN_AT_NAN: bool = false;

    /// The index of the first non-NaN value (0 if all values are NaN). From there, NaNs are
    /// never lower or higher, so the loops skip them without a NaN check (that keeps the
    /// argmin and argmax updates free of branches, which mispredict on random data)
    #[inline(always)]
    fn _start_index(arr: &[ScalarDType]) -> usize {
        arr.iter().position(|v| !v.is_nan()).unwrap_or(0)
    }

    #[inline(always)]
    fn _nan_check(v: ScalarDType) -> bool {
        v.is_nan()
    }
}

// ------- Implement the ScalarArgMinMax trait for the different DTypeStrategy -------

macro_rules! impl_scalar {
    ($dtype_strategy:ty, $($dtype:ty),*) => {
        $(
            impl ScalarArgMinMax<$dtype> for SCALAR<$dtype_strategy>
            {
                #[inline(always)]
                fn argminmax(arr: &[$dtype]) -> (usize, usize) {
                    assert!(!arr.is_empty());
                    let start: usize = Self::_start_index(arr);
                    let mut low_index: usize = start;
                    let mut high_index: usize = start;
                    // It is remarkably faster to iterate over the index and use get_unchecked
                    // than using .iter().enumerate() (with a fold).
                    let mut low: $dtype = unsafe { *arr.get_unchecked(start) };
                    let mut high: $dtype = low;
                    for i in start..arr.len() {
                        let v: $dtype = unsafe { *arr.get_unchecked(i) };
                        if <Self as SCALARInit<$dtype>>::_RETURN_AT_NAN && Self::_nan_check(v) {
                            // When _RETURN_AT_NAN is true and we encounter a NaN
                            return (i, i); // -> return the index
                        }
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

                #[inline(always)]
                fn argmin(arr: &[$dtype]) -> usize {
                    assert!(!arr.is_empty());
                    let start: usize = Self::_start_index(arr);
                    let mut low_index: usize = start;
                    // It is remarkably faster to iterate over the index and use get_unchecked
                    // than using .iter().enumerate() (with a fold).
                    let mut low: $dtype = unsafe { *arr.get_unchecked(start) };
                    for i in start..arr.len() {
                        let v: $dtype = unsafe { *arr.get_unchecked(i) };
                        if <Self as SCALARInit<$dtype>>::_RETURN_AT_NAN && Self::_nan_check(v) {
                            // When _RETURN_AT_NAN is true and we encounter a NaN
                            return i; // -> return the index
                        }
                        if v < low {
                            low = v;
                            low_index = i;
                        }
                    }
                    low_index
                }

                #[inline(always)]
                fn argmax(arr: &[$dtype]) -> usize {
                    assert!(!arr.is_empty());
                    let start: usize = Self::_start_index(arr);
                    let mut high_index: usize = start;
                    // It is remarkably faster to iterate over the index and use get_unchecked
                    // than using .iter().enumerate() (with a fold).
                    let mut high: $dtype = unsafe { *arr.get_unchecked(start) };
                    for i in start..arr.len() {
                        let v: $dtype = unsafe { *arr.get_unchecked(i) };
                        if <Self as SCALARInit<$dtype>>::_RETURN_AT_NAN && Self::_nan_check(v) {
                            // When _RETURN_AT_NAN is true and we encounter a NaN
                            return i; // -> return the index
                        }
                        if v > high {
                            high = v;
                            high_index = i;
                        }
                    }
                    high_index
                }
            }
        )*
    };
}

impl_scalar!(Int, i8, i16, i32, i64, i128, u8, u16, u32, u64, u128);
#[cfg(feature = "float")]
impl_scalar!(FloatReturnNaN, f32, f64);
#[cfg(feature = "float")]
impl_scalar!(FloatIgnoreNaN, f32, f64);

// --- Optional data types

#[cfg(feature = "half")]
use super::scalar_half::{
    scalar_argmax_half_ignore_nan, scalar_argmax_half_return_nan, scalar_argmin_half_ignore_nan,
    scalar_argmin_half_return_nan, scalar_argminmax_half_ignore_nan,
    scalar_argminmax_half_return_nan, Half,
};

/// For the 16-bit floats `f16` and `bf16`
#[cfg(feature = "half")]
impl<T: Half> ScalarArgMinMax<T> for SCALAR<FloatReturnNaN> {
    #[inline(always)]
    fn argminmax(arr: &[T]) -> (usize, usize) {
        scalar_argminmax_half_return_nan(arr)
    }

    #[inline(always)]
    fn argmin(arr: &[T]) -> usize {
        scalar_argmin_half_return_nan(arr)
    }

    #[inline(always)]
    fn argmax(arr: &[T]) -> usize {
        scalar_argmax_half_return_nan(arr)
    }
}

/// For the 16-bit floats `f16` and `bf16`
#[cfg(feature = "half")]
impl<T: Half> ScalarArgMinMax<T> for SCALAR<FloatIgnoreNaN> {
    #[inline(always)]
    fn argminmax(arr: &[T]) -> (usize, usize) {
        scalar_argminmax_half_ignore_nan(arr)
    }

    #[inline(always)]
    fn argmin(arr: &[T]) -> usize {
        scalar_argmin_half_ignore_nan(arr)
    }

    #[inline(always)]
    fn argmax(arr: &[T]) -> usize {
        scalar_argmax_half_ignore_nan(arr)
    }
}

// ======================================= TESTS =======================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Compare SCALAR<Int> with the first index of the min / max value, for every
    /// rotation of `data` (so the extremes and ties occur at every position).
    fn check_all_rotations<T: Ord + Copy>(mut data: Vec<T>)
    where
        SCALAR<Int>: ScalarArgMinMax<T>,
    {
        for _ in 0..data.len() {
            let min = data.iter().min().unwrap();
            let max = data.iter().max().unwrap();
            let argmin = data.iter().position(|v| v == min).unwrap();
            let argmax = data.iter().position(|v| v == max).unwrap();
            assert_eq!(SCALAR::<Int>::argminmax(&data), (argmin, argmax));
            assert_eq!(SCALAR::<Int>::argmin(&data), argmin);
            assert_eq!(SCALAR::<Int>::argmax(&data), argmax);
            data.rotate_left(1);
        }
    }

    #[test]
    fn test_i128_extremes_and_ties() {
        // Values that only differ in the high or in the low 64 bits + duplicate MIN/MAX
        check_all_rotations(vec![
            0,
            -1,
            1,
            i64::MIN as i128,
            i64::MAX as i128,
            u64::MAX as i128,
            1 << 64,
            -(1 << 64),
            i128::MIN,
            i128::MAX,
            i128::MIN + 1,
            i128::MAX - 1,
            i128::MIN,
            i128::MAX,
        ]);
        check_all_rotations(vec![i128::MIN; 3]);
    }

    #[test]
    fn test_u128_extremes_and_ties() {
        // Values that only differ in the high or in the low 64 bits + duplicate MIN/MAX
        check_all_rotations(vec![
            1,
            u64::MAX as u128,
            1 << 64,
            (1 << 64) + 1,
            (1 << 127) - 1,
            1 << 127,
            u128::MIN,
            u128::MAX,
            u128::MAX - 1,
            u128::MIN,
            u128::MAX,
        ]);
        check_all_rotations(vec![u128::MAX; 3]);
    }
}

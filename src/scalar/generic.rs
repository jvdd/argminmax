#[cfg(feature = "float")]
use num_traits::float::FloatCore;
use num_traits::PrimInt;

use super::super::dtype_strategy::Int;
/// The DTypeStrategy for which we implement the ScalarArgMinMax trait
#[cfg(any(feature = "float", feature = "half"))]
use super::super::dtype_strategy::{FloatIgnoreNaN, FloatReturnNaN};

/// Helper trait to initialize the min and max values & check if we should return
/// This will be implemented for all:
/// - ints - Int DTypeStrategy (see 1st impl block below)
/// - uints - Int DTypeStrategy (see 1st impl block below)
/// - floats: returning NaNs - FloatReturnNan DTypeStrategy (see 2nd impl block below)
/// - floats: ignoring NaNs - FloatIgnoreNaN DTypeStrategy (see 3rd impl block below)
///
trait SCALARInit<ScalarDType: Copy + PartialOrd> {
    /// Initialize the initial value for the min and max values

    fn _init_min(arr: &[ScalarDType]) -> ScalarDType;

    fn _init_max(arr: &[ScalarDType]) -> ScalarDType;

    /// Check if we should return at the current value

    fn _return_check(v: ScalarDType) -> bool;
}

/// The ScalarArgMinMax trait that should be implemented for the different DTypeStrategy
/// This trait contains the argminmax operations.
///
pub trait ScalarArgMinMax<ScalarDType: Copy + PartialOrd> {
    fn argminmax(data: &[ScalarDType]) -> (usize, usize);
}

/// SCALAR struct that will implement the ScalarArgMinMax trait for the
/// different DTypeStrategy
///
/// See the impl_scalar! macro below for the implementation of the ScalarArgMinMax trait
///
pub struct SCALAR<DTypeStrategy> {
    pub(crate) _dtype_strategy: std::marker::PhantomData<DTypeStrategy>,
}

/// ------- Implement the SCALARInit trait for the different DTypeStrategy -------

impl<ScalarDType> SCALARInit<ScalarDType> for SCALAR<Int>
where
    ScalarDType: PrimInt,
{
    #[inline(always)]
    fn _init_min(arr: &[ScalarDType]) -> ScalarDType {
        unsafe { *arr.get_unchecked(0) }
    }

    #[inline(always)]
    fn _init_max(arr: &[ScalarDType]) -> ScalarDType {
        unsafe { *arr.get_unchecked(0) }
    }

    #[inline(always)]
    fn _return_check(_v: ScalarDType) -> bool {
        false
    }
}

#[cfg(feature = "float")]
impl<ScalarDType> SCALARInit<ScalarDType> for SCALAR<FloatReturnNaN>
where
    ScalarDType: FloatCore,
{
    #[inline(always)]
    fn _init_min(arr: &[ScalarDType]) -> ScalarDType {
        unsafe { *arr.get_unchecked(0) }
    }

    #[inline(always)]
    fn _init_max(arr: &[ScalarDType]) -> ScalarDType {
        unsafe { *arr.get_unchecked(0) }
    }

    #[inline(always)]
    fn _return_check(v: ScalarDType) -> bool {
        v.is_nan()
    }
}

#[cfg(feature = "float")]
impl<ScalarDType> SCALARInit<ScalarDType> for SCALAR<FloatIgnoreNaN>
where
    ScalarDType: FloatCore,
{
    #[inline(always)]
    fn _init_min(arr: &[ScalarDType]) -> ScalarDType {
        let start_value: ScalarDType = unsafe { *arr.get_unchecked(0) };
        if start_value.is_nan() {
            ScalarDType::infinity()
        } else {
            start_value
        }
    }

    #[inline(always)]
    fn _init_max(arr: &[ScalarDType]) -> ScalarDType {
        let start_value: ScalarDType = unsafe { *arr.get_unchecked(0) };
        if start_value.is_nan() {
            ScalarDType::neg_infinity()
        } else {
            start_value
        }
    }

    #[inline(always)]
    fn _return_check(_v: ScalarDType) -> bool {
        false
    }
}

/// ------- Implement the ScalarArgMinMax trait for the different DTypeStrategy -------

macro_rules! impl_scalar {
    ($dtype_strategy:ty, $($dtype:ty),*) => {
        $(
            impl ScalarArgMinMax<$dtype> for SCALAR<$dtype_strategy> {
                #[inline(always)]
                fn argminmax(arr: &[$dtype]) -> (usize, usize) {
                    assert!(!arr.is_empty());
                    let mut low_index: usize = 0;
                    let mut high_index: usize = 0;
                    // It is remarkably faster to iterate over the index and use get_unchecked
                    // than using .iter().enumerate() (with a fold).
                    let mut low: $dtype = Self::_init_min(arr);
                    let mut high: $dtype = Self::_init_max(arr);
                    for i in 0..arr.len() {
                        let v: $dtype = unsafe { *arr.get_unchecked(i) };
                        if Self::_return_check(v) {
                            return (i, i);
                        }
                        if v < low {
                            low = v;
                            low_index = i;
                        }
                        if v > high {
                            high = v;
                            high_index = i;
                        }
                    }
                    (low_index, high_index)
                }
            }
        )*
    };
}

impl_scalar!(Int, i8, i16, i32, i64, u8, u16, u32, u64);
#[cfg(feature = "float")]
impl_scalar!(FloatReturnNaN, f32, f64);
#[cfg(feature = "float")]
impl_scalar!(FloatIgnoreNaN, f32, f64);

// --- Optional data types

#[cfg(feature = "half")]
use super::scalar_f16::scalar_argminmax_f16_return_nan;
#[cfg(feature = "half")]
use half::f16;

#[cfg(feature = "half")]
impl ScalarArgMinMax<f16> for SCALAR<FloatReturnNaN> {
    #[inline(always)]
    fn argminmax(arr: &[f16]) -> (usize, usize) {
        scalar_argminmax_f16_return_nan(arr)
    }
}

#[cfg(feature = "half")]
impl ScalarArgMinMax<f16> for SCALAR<FloatIgnoreNaN> {
    // TODO: implement this correctly
    #[inline(always)]
    fn argminmax(arr: &[f16]) -> (usize, usize) {
        scalar_argminmax_f16_return_nan(arr)
    }
}

use num_traits::float::FloatCore;
use num_traits::AsPrimitive;

use super::config::SIMDInstructionSet;
use super::task::*;
use crate::scalar::{SCALARIgnoreNaN, ScalarArgMinMax, SCALAR};

// ---------------------------------- SIMD operations ----------------------------------

/// Core SIMD operations
/// These operations are used by the SIMD algorithm and have to be implemented for each
/// data type - SIMD instruction set combination.
/// The operations are implemented in the `simd_*.rs` files.
///
/// Note that for floating point dataypes two implementations are required:
/// - one for the ignore NaN case (uses a floating point SIMDVecDtype)
///   (see the `simd_f*_ignore_nan.rs` files)
/// - one for the return NaN case (uses an integer SIMDVecDtype - as we use the
///   ord_transform to view the floating point data as ordinal integer data).
///   (see the `simd_f*_return_nan.rs` files)
pub trait SIMDOps<ScalarDType, SIMDVecDtype, SIMDMaskDtype, const LANE_SIZE: usize>
where
    ScalarDType: Copy + PartialOrd + AsPrimitive<usize>,
    SIMDVecDtype: Copy,
    SIMDMaskDtype: Copy,
{
    /// Integers > this value **cannot** be accurately represented in SIMDVecDtype
    const MAX_INDEX: usize;
    /// Initial index value for the SIMD vector
    const INITIAL_INDEX: SIMDVecDtype;
    /// Increment value for the SIMD vector
    const INDEX_INCREMENT: SIMDVecDtype;

    /// Convert a SIMD register to array
    unsafe fn _reg_to_arr(reg: SIMDVecDtype) -> [ScalarDType; LANE_SIZE];

    /// Load a SIMD register from memory
    unsafe fn _mm_loadu(data: *const ScalarDType) -> SIMDVecDtype;

    /// Add two SIMD registers
    unsafe fn _mm_add(a: SIMDVecDtype, b: SIMDVecDtype) -> SIMDVecDtype;

    /// Compare two SIMD registers for greater-than (gt): a > b
    /// Returns a SIMD mask
    unsafe fn _mm_cmpgt(a: SIMDVecDtype, b: SIMDVecDtype) -> SIMDMaskDtype;

    /// Compare two SIMD registers for less-than (lt): a < b
    unsafe fn _mm_cmplt(a: SIMDVecDtype, b: SIMDVecDtype) -> SIMDMaskDtype;

    /// Blend two SIMD registers using a SIMD mask (selects elements from a or b)
    unsafe fn _mm_blendv(a: SIMDVecDtype, b: SIMDVecDtype, mask: SIMDMaskDtype) -> SIMDVecDtype;

    /// Horizontal min: get the minimum value from the value SIMD register and its
    /// corresponding index from the index SIMD register
    #[inline(always)]
    unsafe fn _horiz_min(index: SIMDVecDtype, value: SIMDVecDtype) -> (usize, ScalarDType) {
        // This becomes the bottleneck when using 8-bit data types, as for  every 2**7
        // or 2**8 elements, the SIMD inner loop is executed (& thus also terminated)
        // to avoid overflow.
        // To tackle this bottleneck, we use a different approach for 8-bit data types:
        // -> we overwrite this method to perform (in SIMD) the horizontal min
        //    see: https://stackoverflow.com/a/9798369
        // Note: this is not a bottleneck for 16-bit data types, as the termination of
        // the SIMD inner loop is 2**8 times less frequent.
        let index_arr = Self::_reg_to_arr(index);
        let value_arr = Self::_reg_to_arr(value);
        let (min_index, min_value) = min_index_value(&index_arr, &value_arr);
        (min_index.as_(), min_value)
    }

    /// Horizontal max: get the maximum value from the value SIMD register and its
    /// corresponding index from the index SIMD register
    #[inline(always)]
    unsafe fn _horiz_max(index: SIMDVecDtype, value: SIMDVecDtype) -> (usize, ScalarDType) {
        // This becomes the bottleneck when using 8-bit data types, as for  every 2**7
        // or 2**8 elements, the SIMD inner loop is executed (& thus also terminated)
        // to avoid overflow.
        // To tackle this bottleneck, we use a different approach for 8-bit data types:
        // -> we overwrite this method to perform (in SIMD) the horizontal max
        //    see: https://stackoverflow.com/a/9798369
        // Note: this is not a bottleneck for 16-bit data types, as the termination of
        // the SIMD inner loop is 2**8 times less frequent.
        let index_arr = Self::_reg_to_arr(index);
        let value_arr = Self::_reg_to_arr(value);
        let (max_index, max_value) = max_index_value(&index_arr, &value_arr);
        (max_index.as_(), max_value)
    }

    /// Get the largest multiple of LANE_SIZE that is <= MAX_INDEX
    #[inline(always)]
    fn _get_overflow_lane_size_limit() -> usize {
        Self::MAX_INDEX - Self::MAX_INDEX % LANE_SIZE
    }
}

// ---------------------------------- SIMD algorithm -----------------------------------

// --------------- Default

/// The default SIMDCore trait (for all data types)
///
/// This trait is auto-implemented below for:
/// - ints
/// - uints
/// - floats: returning NaN
/// => this corresponds to structs that implement SIMDInstructionSet (see `config.rs`)
///    (thus also for example `SSE` for float returning NaN)
pub trait SIMDCore<ScalarDType, SIMDVecDtype, SIMDMaskDtype, const LANE_SIZE: usize>:
    SIMDOps<ScalarDType, SIMDVecDtype, SIMDMaskDtype, LANE_SIZE>
where
    ScalarDType: Copy + PartialOrd + AsPrimitive<usize>,
    SIMDVecDtype: Copy,
    SIMDMaskDtype: Copy,
{
    /// Core argminmax algorithm - returns (argmin, min, argmax, max)
    ///
    /// This method asserts:
    /// - the array length is a multiple of LANE_SIZE
    /// This method assumes:
    /// - the array length is <= MAX_INDEX
    ///
    /// Note that this method is not overflow safe, as it assumes that the array length
    /// is <= MAX_INDEX. The `_overflow_safe_core_argminmax method` is overflow safe.
    ///
    /// Note that this method is leveraged by the return NaN implementation (as the
    /// float values - including NaNs - are mapped to ordinal integers).
    #[inline(always)]
    unsafe fn _core_argminmax(arr: &[ScalarDType]) -> (usize, ScalarDType, usize, ScalarDType) {
        assert_eq!(arr.len() % LANE_SIZE, 0);
        // Efficient calculation of argmin and argmax together
        let mut new_index = Self::INITIAL_INDEX;
        let mut index_low = Self::INITIAL_INDEX;
        let mut index_high = Self::INITIAL_INDEX;

        let mut arr_ptr = arr.as_ptr(); // Array pointer we will increment in the loop
        let mut values_low = Self::_mm_loadu(arr_ptr);
        let mut values_high = Self::_mm_loadu(arr_ptr);

        // This is (40%-5%) slower than the loop below (depending on the data type)
        // arr.chunks_exact(LANE_SIZE)
        //     .into_iter()
        //     .skip(1)
        //     .for_each(|step| {
        //         new_index = Self::_mm_add(new_index, increment);

        //         let new_values = Self::_mm_loadu(step.as_ptr());

        //         let lt_mask = Self::_mm_cmplt(new_values, values_low);
        //         let gt_mask = Self::_mm_cmpgt(new_values, values_high);

        //         index_low = Self::_mm_blendv(index_low, new_index, lt_mask);
        //         index_high = Self::_mm_blendv(index_high, new_index, gt_mask);

        //         values_low = Self::_mm_blendv(values_low, new_values, lt_mask);
        //         values_high = Self::_mm_blendv(values_high, new_values, gt_mask);
        //     });

        for _ in 0..arr.len() / LANE_SIZE - 1 {
            // Increment the index
            new_index = Self::_mm_add(new_index, Self::INDEX_INCREMENT);
            // Load the next chunk of data
            arr_ptr = arr_ptr.add(LANE_SIZE);
            let new_values = Self::_mm_loadu(arr_ptr);

            // Update the lowest values and index
            let mask_low = Self::_mm_cmplt(new_values, values_low);
            values_low = Self::_mm_blendv(values_low, new_values, mask_low);
            index_low = Self::_mm_blendv(index_low, new_index, mask_low);

            // Update the highest values and index
            let mask_high = Self::_mm_cmpgt(new_values, values_high);
            values_high = Self::_mm_blendv(values_high, new_values, mask_high);
            index_high = Self::_mm_blendv(index_high, new_index, mask_high);
        }

        // Get the min/max index and corresponding value from the SIMD vectors and return
        let (min_index, min_value) = Self::_horiz_min(index_low, values_low);
        let (max_index, max_value) = Self::_horiz_max(index_high, values_high);
        (min_index, min_value, max_index, max_value)
    }

    /// Overflow-safe core argminmax algorithm - returns (argmin, min, argmax, max)
    ///
    /// This method asserts:
    /// - the array is not empty
    /// - the array length is a multiple of LANE_SIZE
    ///
    /// Note that this method checks for nans by comparing v != v (is true for nans)
    /// -> returns once `_core_argminmax` returns a NaN value
    #[inline(always)]
    unsafe fn _overflow_safe_core_argminmax(
        arr: &[ScalarDType],
    ) -> (usize, ScalarDType, usize, ScalarDType) {
        assert!(!arr.is_empty());
        assert_eq!(arr.len() % LANE_SIZE, 0);
        // 0. Get the max value of the data type - which needs to be divided by LANE_SIZE
        let dtype_max = Self::_get_overflow_lane_size_limit();

        // 1. Determine the number of loops needed
        // let n_loops = (arr.len() + dtype_max - 1) / dtype_max; // ceil division
        let n_loops = arr.len() / dtype_max; // floor division

        // 2. Perform overflow-safe _core_argminmax
        let mut min_index: usize = 0;
        let mut min_value: ScalarDType = unsafe { *arr.get_unchecked(0) };
        let mut max_index: usize = 0;
        let mut max_value: ScalarDType = unsafe { *arr.get_unchecked(0) };
        let mut start: usize = 0;
        // 2.0 Perform the full loops
        for _ in 0..n_loops {
            if min_value != min_value || max_value != max_value {
                // If min_value or max_value is NaN, we can return immediately
                return (min_index, min_value, max_index, max_value);
            }
            let (min_index_, min_value_, max_index_, max_value_) =
                Self::_core_argminmax(&arr[start..start + dtype_max]);
            if min_value_ < min_value || min_value_ != min_value_ {
                min_index = start + min_index_;
                min_value = min_value_;
            }
            if max_value_ > max_value || max_value_ != max_value_ {
                max_index = start + max_index_;
                max_value = max_value_;
            }
            start += dtype_max;
        }
        // 2.1 Handle the remainder
        if start < arr.len() {
            if min_value != min_value || max_value != max_value {
                // If min_value or max_value is NaN, we can return immediately
                return (min_index, min_value, max_index, max_value);
            }
            let (min_index_, min_value_, max_index_, max_value_) =
                Self::_core_argminmax(&arr[start..]);
            if min_value_ < min_value || min_value_ != min_value_ {
                min_index = start + min_index_;
                min_value = min_value_;
            }
            if max_value_ > max_value || max_value_ != max_value_ {
                max_index = start + max_index_;
                max_value = max_value_;
            }
        }

        // 3. Return the min/max index and corresponding value
        (min_index, min_value, max_index, max_value)
    }
}

// Implement SIMDCore where SIMDOps is implemented (for the SIMDIstructionSet structs)
impl<T, ScalarDType, SIMDVecDtype, SIMDMaskDtype, const LANE_SIZE: usize>
    SIMDCore<ScalarDType, SIMDVecDtype, SIMDMaskDtype, LANE_SIZE> for T
where
    ScalarDType: Copy + PartialOrd + AsPrimitive<usize>,
    SIMDVecDtype: Copy,
    SIMDMaskDtype: Copy,
    T: SIMDOps<ScalarDType, SIMDVecDtype, SIMDMaskDtype, LANE_SIZE> + SIMDInstructionSet,
{
    // Use the implementation
}

// --------------- Float Ignore NaNs

/// SIMD operations for setting a SIMD vector to a scalar value (only required for floats)
pub trait SIMDSetOps<ScalarDType, SIMDVecDtype>
where
    ScalarDType: FloatCore,
{
    /// Set a SIMD vector to a scalar value (each lane is set to the scalar value)
    unsafe fn _mm_set1(a: ScalarDType) -> SIMDVecDtype;
}

/// SIMDCore trait that ignore NaNs (for float types)
///
/// This trait is auto-implemented below for:
/// - floats: ignoring NaN
/// => this corresponds to the IgnoreNan structs (see `config.rs`)
///    (for example `SSEIgnoreNaN`)
pub trait SIMDCoreIgnoreNaN<ScalarDType, SIMDVecDtype, SIMDMaskDtype, const LANE_SIZE: usize>:
    SIMDOps<ScalarDType, SIMDVecDtype, SIMDMaskDtype, LANE_SIZE> + SIMDSetOps<ScalarDType, SIMDVecDtype>
where
    ScalarDType: FloatCore + AsPrimitive<usize>,
    SIMDVecDtype: Copy,
    SIMDMaskDtype: Copy,
{
    /// Core argminmax algorithm - returns (argmin, min, argmax, max)
    ///
    /// This method asserts:
    /// - the array length is a multiple of LANE_SIZE
    /// This method assumes:
    /// - the array length is <= MAX_INDEX
    ///
    /// Note that this method is not overflow safe, as it assumes that the array length
    /// is <= MAX_INDEX. The `_overflow_safe_core_argminmax method` is overflow safe.
    #[inline(always)]
    unsafe fn _core_argminmax(arr: &[ScalarDType]) -> (usize, ScalarDType, usize, ScalarDType) {
        assert_eq!(arr.len() % LANE_SIZE, 0);
        // Efficient calculation of argmin and argmax together
        let mut new_index = Self::INITIAL_INDEX;

        let mut arr_ptr = arr.as_ptr(); // Array pointer we will increment in the loop
        let new_values = Self::_mm_loadu(arr_ptr);

        // Update the lowest values and index
        let mask_low = Self::_mm_cmplt(new_values, Self::_mm_set1(ScalarDType::infinity()));
        let mut values_low = Self::_mm_blendv(
            Self::_mm_set1(ScalarDType::infinity()),
            new_values,
            mask_low,
        );
        let mut index_low =
            Self::_mm_blendv(Self::_mm_set1(ScalarDType::zero()), new_index, mask_low);

        // Update the highest values and index
        let mask_high = Self::_mm_cmpgt(new_values, Self::_mm_set1(ScalarDType::neg_infinity()));
        let mut values_high = Self::_mm_blendv(
            Self::_mm_set1(ScalarDType::neg_infinity()),
            new_values,
            mask_high,
        );
        let mut index_high =
            Self::_mm_blendv(Self::_mm_set1(ScalarDType::zero()), new_index, mask_high);

        for _ in 0..arr.len() / LANE_SIZE - 1 {
            // Increment the index
            new_index = Self::_mm_add(new_index, Self::INDEX_INCREMENT);
            // Load the next chunk of data
            arr_ptr = arr_ptr.add(LANE_SIZE);
            let new_values = Self::_mm_loadu(arr_ptr);

            // Update the lowest values and index
            let mask_low = Self::_mm_cmplt(new_values, values_low);
            values_low = Self::_mm_blendv(values_low, new_values, mask_low);
            index_low = Self::_mm_blendv(index_low, new_index, mask_low);

            // Update the highest values and index
            let mask_high = Self::_mm_cmpgt(new_values, values_high);
            values_high = Self::_mm_blendv(values_high, new_values, mask_high);
            index_high = Self::_mm_blendv(index_high, new_index, mask_high);
        }

        // Get the min/max index and corresponding value from the SIMD vectors and return
        let (min_index, min_value) = Self::_horiz_min(index_low, values_low);
        let (max_index, max_value) = Self::_horiz_max(index_high, values_high);
        (min_index, min_value, max_index, max_value)
    }

    /// Overflow-safe core argminmax algorithm - returns (argmin, min, argmax, max)
    ///
    /// This method asserts:
    /// - the array is not empty
    /// - the array length is a multiple of LANE_SIZE
    ///
    /// Note that this method ignores nans by assuring that no NaN values are inserted
    /// in the initial min / max SIMD vectors. Since comparing a value to NaN always
    /// returns false, the NaN values will never be selected as the min / max values.
    #[inline(always)]
    unsafe fn _overflow_safe_core_argminmax(
        arr: &[ScalarDType],
    ) -> (usize, ScalarDType, usize, ScalarDType) {
        assert!(!arr.is_empty());
        // 0. Get the max value of the data type - which needs to be divided by LANE_SIZE
        let dtype_max = Self::_get_overflow_lane_size_limit();

        // 1. Determine the number of loops needed
        // let n_loops = (arr.len() + dtype_max - 1) / dtype_max; // ceil division
        let n_loops = arr.len() / dtype_max; // floor division

        // 2. Perform overflow-safe _core_argminmax
        let mut min_index: usize = 0;
        let mut min_value: ScalarDType = ScalarDType::infinity();
        let mut max_index: usize = 0;
        let mut max_value: ScalarDType = ScalarDType::neg_infinity();
        let mut start: usize = 0;
        // 2.0 Perform the full loops
        for _ in 0..n_loops {
            let (min_index_, min_value_, max_index_, max_value_) =
                Self::_core_argminmax(&arr[start..start + dtype_max]);
            if min_value_ < min_value {
                min_index = start + min_index_;
                min_value = min_value_;
            }
            if max_value_ > max_value {
                max_index = start + max_index_;
                max_value = max_value_;
            }
            start += dtype_max;
        }
        // 2.1 Handle the remainder
        if start < arr.len() {
            let (min_index_, min_value_, max_index_, max_value_) =
                Self::_core_argminmax(&arr[start..]);
            if min_value_ < min_value {
                min_index = start + min_index_;
                min_value = min_value_;
            }
            if max_value_ > max_value {
                max_index = start + max_index_;
                max_value = max_value_;
            }
        }

        // 3. Return the min/max index and corresponding value
        (min_index, min_value, max_index, max_value)
    }
}

// Implement SIMDCoreIgnoreNaNs where SIMDOps + SIMDSetOps is implemented for floats
impl<T, ScalarDType, SIMDVecDtype, SIMDMaskDtype, const LANE_SIZE: usize>
    SIMDCoreIgnoreNaN<ScalarDType, SIMDVecDtype, SIMDMaskDtype, LANE_SIZE> for T
where
    ScalarDType: FloatCore + AsPrimitive<usize>,
    SIMDVecDtype: Copy,
    SIMDMaskDtype: Copy,
    T: SIMDOps<ScalarDType, SIMDVecDtype, SIMDMaskDtype, LANE_SIZE>
        + SIMDSetOps<ScalarDType, SIMDVecDtype>,
{
    // Use the implementation
}

// -------------------------------- ArgMinMax SIMD TRAIT -------------------------------

// --------------- Default

/// Trait for SIMD argminmax operations
///
/// This trait its `argminmax` method should be implemented for all structs that
/// implement `SIMDOps` for the same generics.
/// This trait is implemented for:
/// - ints (see, the simd_i*.rs files)
/// - uints (see, the simd_u*.rs files)
/// - floats: returning NaNs (see, the simd_f*_return_nan.rs files)
#[allow(clippy::missing_safety_doc)] // TODO: add safety docs?
pub trait SIMDArgMinMax<ScalarDType, SIMDVecDtype, SIMDMaskDtype, const LANE_SIZE: usize>:
    SIMDCore<ScalarDType, SIMDVecDtype, SIMDMaskDtype, LANE_SIZE>
where
    ScalarDType: Copy + PartialOrd + AsPrimitive<usize>,
    SIMDVecDtype: Copy,
    SIMDMaskDtype: Copy,
{
    /// Returns the index of the minimum and maximum value in the array
    unsafe fn argminmax(data: &[ScalarDType]) -> (usize, usize);

    // Is necessary to have a separate function for this so we can call it in the
    // argminmax function when we add the target feature to the function.
    #[inline(always)]
    unsafe fn _argminmax(data: &[ScalarDType]) -> (usize, usize)
    where
        SCALAR: ScalarArgMinMax<ScalarDType>,
    {
        argminmax_generic(
            data,
            LANE_SIZE,
            Self::_overflow_safe_core_argminmax,
            false,
            SCALAR::argminmax,
        )
    }
}

// --------------- Float Return NaN

// This is the same code as the default trait - thus we can just use the default trait.

// --------------- Float Ignore NaN

/// Trait for SIMD argminmax operations that ignore NaNs
///
/// This trait its `argminmax` method should be implemented for all structs that
/// implement `SIMDOps` and `SIMDSetOps` for the same generics.
/// This trait is implemented for:
/// - floats: ignoring NaNs (see, the simd_f*_ignore_nan.rs files)
#[allow(clippy::missing_safety_doc)] // TODO: add safety docs?
pub trait SIMDArgMinMaxIgnoreNaN<ScalarDType, SIMDVecDtype, SIMDMaskDtype, const LANE_SIZE: usize>:
    SIMDCoreIgnoreNaN<ScalarDType, SIMDVecDtype, SIMDMaskDtype, LANE_SIZE>
where
    ScalarDType: FloatCore + AsPrimitive<usize>,
    SIMDVecDtype: Copy,
    SIMDMaskDtype: Copy,
{
    /// Returns the index of the minimum and maximum value in the array
    unsafe fn argminmax(data: &[ScalarDType]) -> (usize, usize);

    // Is necessary to have a separate function for this so we can call it in the
    // argminmax function when we add the target feature to the function.
    #[inline(always)]
    unsafe fn _argminmax(data: &[ScalarDType]) -> (usize, usize)
    where
        SCALARIgnoreNaN: ScalarArgMinMax<ScalarDType>,
    {
        argminmax_generic(
            data,
            LANE_SIZE,
            Self::_overflow_safe_core_argminmax,
            true,
            SCALARIgnoreNaN::argminmax,
        )
    }
}

// --------------------------------- Unimplement Macros --------------------------------

// TODO: temporarily removed the target_arch specification bc we currently do not
// ArgMinMaxIgnoreNan for f16 ignore nan

// #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
macro_rules! unimpl_SIMDOps {
    ($scalar_type:ty, $reg:ty, $simd_instructionset:ident) => {
        impl SIMDOps<$scalar_type, $reg, $reg, 0> for $simd_instructionset {
            const INITIAL_INDEX: $reg = 0;
            const INDEX_INCREMENT: $reg = 0;
            const MAX_INDEX: usize = 0;

            unsafe fn _reg_to_arr(_reg: $reg) -> [$scalar_type; 0] {
                unimplemented!()
            }

            unsafe fn _mm_loadu(_data: *const $scalar_type) -> $reg {
                unimplemented!()
            }

            unsafe fn _mm_add(_a: $reg, _b: $reg) -> $reg {
                unimplemented!()
            }

            unsafe fn _mm_cmpgt(_a: $reg, _b: $reg) -> $reg {
                unimplemented!()
            }

            unsafe fn _mm_cmplt(_a: $reg, _b: $reg) -> $reg {
                unimplemented!()
            }

            unsafe fn _mm_blendv(_a: $reg, _b: $reg, _mask: $reg) -> $reg {
                unimplemented!()
            }
        }
    };
}

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
macro_rules! unimpl_SIMDArgMinMax {
    ($scalar_type:ty, $reg:ty, $simd_instructionset:ident) => {
        impl SIMDArgMinMax<$scalar_type, $reg, $reg, 0> for $simd_instructionset {
            unsafe fn argminmax(_data: &[$scalar_type]) -> (usize, usize) {
                unimplemented!()
            }
        }
    };
}

// #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
macro_rules! unimpl_SIMDArgMinMaxIgnoreNaN {
    ($scalar_type:ty, $reg:ty, $simd_instructionset:ident) => {
        impl SIMDSetOps<$scalar_type, $reg> for $simd_instructionset {
            unsafe fn _mm_set1(_a: $scalar_type) -> $reg {
                unimplemented!()
            }
        }
        impl SIMDArgMinMaxIgnoreNaN<$scalar_type, $reg, $reg, 0> for $simd_instructionset {
            unsafe fn argminmax(_data: &[$scalar_type]) -> (usize, usize) {
                unimplemented!()
            }
        }
    };
}

// TODO: temporarily removed the target_arch until we implement f16_ignore_nans
#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
pub(crate) use unimpl_SIMDArgMinMax; // Now classic paths Just Work™
                                     // #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
pub(crate) use unimpl_SIMDArgMinMaxIgnoreNaN; // Now classic paths Just Work™
                                              // #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
pub(crate) use unimpl_SIMDOps; // Now classic paths Just Work™

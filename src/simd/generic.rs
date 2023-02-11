use num_traits::{AsPrimitive, Float};

use super::config::SIMDInstructionSet;
use super::task::*;
use crate::scalar::{ScalarArgMinMax, SCALAR};

// ---------------------------------- SIMD operations ----------------------------------

/// Core SIMD operations
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

    unsafe fn _reg_to_arr(reg: SIMDVecDtype) -> [ScalarDType; LANE_SIZE];

    unsafe fn _mm_loadu(data: *const ScalarDType) -> SIMDVecDtype;

    unsafe fn _mm_add(a: SIMDVecDtype, b: SIMDVecDtype) -> SIMDVecDtype;

    unsafe fn _mm_cmpgt(a: SIMDVecDtype, b: SIMDVecDtype) -> SIMDMaskDtype;

    unsafe fn _mm_cmplt(a: SIMDVecDtype, b: SIMDVecDtype) -> SIMDMaskDtype;

    unsafe fn _mm_blendv(a: SIMDVecDtype, b: SIMDVecDtype, mask: SIMDMaskDtype) -> SIMDVecDtype;

    // TODO: remove this?
    #[inline(always)]
    unsafe fn _mm_prefetch(data: *const ScalarDType) {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            #[cfg(target_arch = "x86")]
            use std::arch::x86::_mm_prefetch;
            #[cfg(target_arch = "x86_64")]
            use std::arch::x86_64::_mm_prefetch;

            _mm_prefetch(data as *const i8, 0); // 0=NTA
        }
        #[cfg(target_arch = "aarch64")]
        {
            use std::arch::aarch64::_prefetch;

            _prefetch(data as *const i8, 0, 0); // 0=READ, 0=NTA
        }
    }

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

// ---------------------------------- SIMD algorithm ----------------------------------

// --------------- Default

/// The default SIMDCore trait (for all data types)
pub trait SIMDCore<ScalarDType, SIMDVecDtype, SIMDMaskDtype, const LANE_SIZE: usize>:
    SIMDOps<ScalarDType, SIMDVecDtype, SIMDMaskDtype, LANE_SIZE>
where
    ScalarDType: Copy + PartialOrd + AsPrimitive<usize>,
    SIMDVecDtype: Copy,
    SIMDMaskDtype: Copy,
{
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
            // Self::_mm_prefetch(arr_ptr); // Hint to the CPU to prefetch the next chunk of data
            let new_values = Self::_mm_loadu(arr_ptr);

            // Update the lowest values and index
            let mask = Self::_mm_cmplt(new_values, values_low);
            values_low = Self::_mm_blendv(values_low, new_values, mask);
            index_low = Self::_mm_blendv(index_low, new_index, mask);

            // Update the highest values and index
            let mask = Self::_mm_cmpgt(new_values, values_high);
            values_high = Self::_mm_blendv(values_high, new_values, mask);
            index_high = Self::_mm_blendv(index_high, new_index, mask);

            // 25 is a non-scientific number, but seems to work overall
            //  => TODO: probably this should be in function of the data type
            // Self::_mm_prefetch(arr_ptr.add(LANE_SIZE * 25)); // Hint to the CPU to prefetch upcoming data
        }

        // Get the min/max index and corresponding value from the SIMD vectors and return
        let (min_index, min_value) = Self::_horiz_min(index_low, values_low);
        let (max_index, max_value) = Self::_horiz_max(index_high, values_high);
        (min_index, min_value, max_index, max_value)
    }

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
        let mut min_value: ScalarDType = unsafe { *arr.get_unchecked(0) };
        let mut max_index: usize = 0;
        let mut max_value: ScalarDType = unsafe { *arr.get_unchecked(0) };
        let mut start: usize = 0;
        // 2.0 Perform the full loops
        for _ in 0..n_loops {
            // Self::_mm_prefetch(arr.as_ptr().add(start));
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
            // Self::_mm_prefetch(arr.as_ptr().add(start));
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
    ScalarDType: Float,
{
    unsafe fn _mm_set1(a: ScalarDType) -> SIMDVecDtype;
}

/// SIMDCore trait that ignore NaNs (for float types)
pub trait SIMDCoreFloatIgnoreNaN<ScalarDType, SIMDVecDtype, SIMDMaskDtype, const LANE_SIZE: usize>:
    SIMDOps<ScalarDType, SIMDVecDtype, SIMDMaskDtype, LANE_SIZE> + SIMDSetOps<ScalarDType, SIMDVecDtype>
where
    ScalarDType: Float + AsPrimitive<usize>,
    SIMDVecDtype: Copy,
    SIMDMaskDtype: Copy,
{
    #[inline(always)]
    unsafe fn _core_argminmax(arr: &[ScalarDType]) -> (usize, ScalarDType, usize, ScalarDType) {
        assert_eq!(arr.len() % LANE_SIZE, 0);
        // Efficient calculation of argmin and argmax together
        let mut new_index = Self::INITIAL_INDEX;

        let mut arr_ptr = arr.as_ptr(); // Array pointer we will increment in the loop
        let new_values = Self::_mm_loadu(arr_ptr);

        // Update the lowest values and index
        let mask = Self::_mm_cmplt(new_values, Self::_mm_set1(ScalarDType::infinity()));
        let mut values_low =
            Self::_mm_blendv(Self::_mm_set1(ScalarDType::infinity()), new_values, mask);
        let mut index_low = Self::_mm_blendv(Self::_mm_set1(ScalarDType::zero()), new_index, mask);

        // Update the highest values and index
        let mask = Self::_mm_cmpgt(new_values, Self::_mm_set1(ScalarDType::neg_infinity()));
        let mut values_high = Self::_mm_blendv(
            Self::_mm_set1(ScalarDType::neg_infinity()),
            new_values,
            mask,
        );
        let mut index_high = Self::_mm_blendv(Self::_mm_set1(ScalarDType::zero()), new_index, mask);

        for _ in 0..arr.len() / LANE_SIZE - 1 {
            // Increment the index
            new_index = Self::_mm_add(new_index, Self::INDEX_INCREMENT);
            // Load the next chunk of data
            arr_ptr = arr_ptr.add(LANE_SIZE);
            let new_values = Self::_mm_loadu(arr_ptr);

            // Update the lowest values and index
            let mask = Self::_mm_cmplt(new_values, values_low);
            values_low = Self::_mm_blendv(values_low, new_values, mask);
            index_low = Self::_mm_blendv(index_low, new_index, mask);

            // Update the highest values and index
            let mask = Self::_mm_cmpgt(new_values, values_high);
            values_high = Self::_mm_blendv(values_high, new_values, mask);
            index_high = Self::_mm_blendv(index_high, new_index, mask);
        }

        // Get the min/max index and corresponding value from the SIMD vectors and return
        let (min_index, min_value) = Self::_horiz_min(index_low, values_low);
        let (max_index, max_value) = Self::_horiz_max(index_high, values_high);
        (min_index, min_value, max_index, max_value)
    }

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
            // Self::_mm_prefetch(arr.as_ptr().add(start));
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
            // Self::_mm_prefetch(arr.as_ptr().add(start));
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

// Implement SIMDCoreFloatIgnoreNaNs where SIMDOps + SIMDSetOps is implemented for floats
impl<T, ScalarDType, SIMDVecDtype, SIMDMaskDtype, const LANE_SIZE: usize>
    SIMDCoreFloatIgnoreNaN<ScalarDType, SIMDVecDtype, SIMDMaskDtype, LANE_SIZE> for T
where
    ScalarDType: Float + AsPrimitive<usize>,
    SIMDVecDtype: Copy,
    SIMDMaskDtype: Copy,
    T: SIMDOps<ScalarDType, SIMDVecDtype, SIMDMaskDtype, LANE_SIZE>
        + SIMDSetOps<ScalarDType, SIMDVecDtype>,
{
    // Use the implementation
}

// --------------- Float Return NaNs

// IDEA: make SIMDOps extend this trait & provide empty implementations

// pub trait SIMDOrdTransformOps<FloatDType, IntDType, SIMDVecDTtype, const LANE_SIZE: usize>
// where
//     FloatDType: Copy + Float,
//     IntDType: Copy,
//     SIMDVecDTtype: Copy,
// {
//     const BIT_SHIFT: i32;
//     const SIGN_BIT_MASK: SIMDVecDTtype;
//     fn _reg_to_int_arr(reg: SIMDVecDTtype) -> [IntDType; LANE_SIZE];
//     fn _mm_srai(a: SIMDVecDTtype, imm8: i32) -> SIMDVecDTtype;
//     fn _mm_and(a: SIMDVecDTtype, b: SIMDVecDTtype) -> SIMDVecDTtype;
//     fn _mm_xor(a: SIMDVecDTtype, b: SIMDVecDTtype) -> SIMDVecDTtype;
//     #[inline(always)]
//     fn _mm_ord_transform(a: SIMDVecDTtype) -> SIMDVecDTtype {
//         let sign_bit_shifted = Self::_mm_srai(a, Self::BIT_SHIFT);
//         let sign_bit_masked = Self::_mm_and(sign_bit_shifted, Self::SIGN_BIT_MASK);
//         Self::_mm_xor(a, sign_bit_masked)
//     }
// }

// ------------------------------- ArgMinMax SIMD TRAIT ------------------------------

// --------------- Default

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
        argminmax_generic(data, LANE_SIZE, Self::_overflow_safe_core_argminmax)
    }
}

// --------------- Float Ignore NaNs

#[allow(clippy::missing_safety_doc)] // TODO: add safety docs?
pub trait SIMDArgMinMaxFloatIgnoreNaN<
    ScalarDType,
    SIMDVecDtype,
    SIMDMaskDtype,
    const LANE_SIZE: usize,
>: SIMDCoreFloatIgnoreNaN<ScalarDType, SIMDVecDtype, SIMDMaskDtype, LANE_SIZE> where
    ScalarDType: Copy + PartialOrd + AsPrimitive<usize> + Float,
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
        argminmax_generic(data, LANE_SIZE, Self::_overflow_safe_core_argminmax)
    }
}

// TODO: update this to use the new SIMD trait
#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
macro_rules! unimplement_simd {
    ($scalar_type:ty, $reg:ty, $simd_type:ident, $zero:literal) => {
        impl SIMD<$scalar_type, $reg, $reg, 0> for $simd_type {
            const INITIAL_INDEX: $reg = 0;
            const MAX_INDEX: usize = 0;
            
            const INDEX_INCREMENT: $reg = 0;
            const MIN_VALUE: $scalar_type = $zero;
            const MAX_VALUE: $scalar_type = $zero;


            unsafe fn _reg_to_arr(_reg: $reg) -> [$scalar_type; 0] {
                unimplemented!()
            }

            unsafe fn _mm_loadu(_data: *const $scalar_type) -> $reg {
                unimplemented!()
            }

            unsafe fn _mm_set1(_a: $scalar_type) -> $reg {
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

            unsafe fn argminmax(_data: &[$scalar_type]) -> (usize, usize) {
                unimplemented!()
            }
        }
    };
}
#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
pub(crate) use unimplement_simd; // Now classic paths Just Work™

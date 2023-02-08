use num_traits::{AsPrimitive, Zero};

use super::task::*;
use crate::scalar::{ScalarArgMinMax, SCALAR};

// TODO: other potential generic SIMDIndexDtype: Copy
#[allow(clippy::missing_safety_doc)] // TODO: add safety docs?
pub trait SIMD<
    ScalarDType: Copy + PartialOrd + AsPrimitive<usize> + Zero,
    SIMDVecDtype: Copy,
    SIMDMaskDtype: Copy,
    const LANE_SIZE: usize,
>
{
    /// Initial index value for the SIMD vector
    const INITIAL_INDEX: SIMDVecDtype;
    /// Increment value for the SIMD vector
    const INDEX_INCREMENT: SIMDVecDtype;

    /// Integers > this value **cannot** be accurately represented in SIMDVecDtype
    const MAX_INDEX: usize;

    /// Smallest value that can be represented in ScalarDType
    const MIN_VALUE: ScalarDType;
    /// Largest value that can be represented in ScalarDType
    const MAX_VALUE: ScalarDType;

    #[inline(always)]
    fn _find_largest_lower_multiple_of_lane_size(n: usize) -> usize {
        n - n % LANE_SIZE
    }

    // ------------------------------------ SIMD HELPERS --------------------------------------

    unsafe fn _reg_to_arr(reg: SIMDVecDtype) -> [ScalarDType; LANE_SIZE];

    unsafe fn _mm_loadu(data: *const ScalarDType) -> SIMDVecDtype;

    unsafe fn _mm_set1(a: ScalarDType) -> SIMDVecDtype;

    unsafe fn _mm_add(a: SIMDVecDtype, b: SIMDVecDtype) -> SIMDVecDtype;

    unsafe fn _mm_cmpgt(a: SIMDVecDtype, b: SIMDVecDtype) -> SIMDMaskDtype;

    unsafe fn _mm_cmplt(a: SIMDVecDtype, b: SIMDVecDtype) -> SIMDMaskDtype;

    unsafe fn _mm_blendv(a: SIMDVecDtype, b: SIMDVecDtype, mask: SIMDMaskDtype) -> SIMDVecDtype;

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

    // ------------------------------------ ARGMINMAX --------------------------------------

    unsafe fn argminmax(data: &[ScalarDType]) -> (usize, usize);

    #[inline(always)]
    unsafe fn _argminmax(data: &[ScalarDType]) -> (usize, usize)
    where
        SCALAR: ScalarArgMinMax<ScalarDType>,
    {
        argminmax_generic(data, LANE_SIZE, Self::_overflow_safe_core_argminmax)
    }

    #[inline(always)]
    unsafe fn _overflow_safe_core_argminmax(
        arr: &[ScalarDType],
    ) -> (usize, ScalarDType, usize, ScalarDType) {
        assert!(!arr.is_empty());
        // 0. Get the max value of the data type - which needs to be divided by LANE_SIZE
        let dtype_max = Self::_find_largest_lower_multiple_of_lane_size(Self::MAX_INDEX);

        // 1. Determine the number of loops needed
        // let n_loops = (arr.len() + dtype_max - 1) / dtype_max; // ceil division
        let n_loops = arr.len() / dtype_max; // floor division

        // 2. Perform overflow-safe _core_argminmax
        let mut min_index: usize = 0;
        let mut min_value: ScalarDType = Self::MAX_VALUE;
        let mut max_index: usize = 0;
        let mut max_value: ScalarDType = Self::MIN_VALUE;
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

    // TODO: can be cleaner (perhaps?)
    #[inline(always)]
    unsafe fn _get_min_max_index_value(
        index_low: SIMDVecDtype,
        values_low: SIMDVecDtype,
        index_high: SIMDVecDtype,
        values_high: SIMDVecDtype,
    ) -> (usize, ScalarDType, usize, ScalarDType) {
        let (min_index, min_value) = Self::_horiz_min(index_low, values_low);
        let (max_index, max_value) = Self::_horiz_max(index_high, values_high);
        (min_index, min_value, max_index, max_value)
    }

    #[inline(always)]
    unsafe fn _core_argminmax(arr: &[ScalarDType]) -> (usize, ScalarDType, usize, ScalarDType) {
        assert_eq!(arr.len() % LANE_SIZE, 0);
        // Efficient calculation of argmin and argmax together
        let mut new_index = Self::INITIAL_INDEX;

        let mut arr_ptr = arr.as_ptr(); // Array pointer we will increment in the loop
        let new_values = Self::_mm_loadu(arr_ptr);

        // Update the lowest values and index
        let mask = Self::_mm_cmplt(new_values, Self::_mm_set1(Self::MAX_VALUE));
        let mut values_low = Self::_mm_blendv(Self::_mm_set1(Self::MAX_VALUE), new_values, mask);
        let mut index_low = Self::_mm_blendv(Self::_mm_set1(ScalarDType::zero()), new_index, mask);

        // Update the highest values and index
        let mask = Self::_mm_cmpgt(new_values, Self::_mm_set1(Self::MIN_VALUE));
        let mut values_high = Self::_mm_blendv(Self::_mm_set1(Self::MIN_VALUE), new_values, mask);
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

        Self::_get_min_max_index_value(index_low, values_low, index_high, values_high)
    }
}

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
macro_rules! unimplement_simd {
    ($scalar_type:ty, $reg:ty, $simd_type:ident) => {
        impl SIMD<$scalar_type, $reg, $reg, 0> for $simd_type {
            const INITIAL_INDEX: $reg = 0;
            const MAX_INDEX: usize = 0;

            unsafe fn _reg_to_arr(_reg: $reg) -> [$scalar_type; 0] {
                unimplemented!()
            }

            unsafe fn _mm_loadu(_data: *const $scalar_type) -> $reg {
                unimplemented!()
            }

            unsafe fn _mm_set1(_a: usize) -> $reg {
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

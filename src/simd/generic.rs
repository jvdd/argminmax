use num_traits::AsPrimitive;

use super::task::*;
use crate::scalar::{ScalarArgMinMax, SCALAR};

// TODO: other potential generic SIMDIndexDtype: Copy
#[allow(clippy::missing_safety_doc)] // TODO: add safety docs?
pub trait SIMD<
    ScalarDType: Copy + PartialOrd + AsPrimitive<usize>,
    SIMDVecDtype: Copy,
    SIMDMaskDtype: Copy,
    const LANE_SIZE: usize,
>
{
    const INITIAL_INDEX: SIMDVecDtype;
    const MAX_INDEX: usize; // Integers > this value **cannot** be accurately represented in SIMDVecDtype

    #[inline(always)]
    fn _find_largest_lower_multiple_of_lane_size(n: usize) -> usize {
        n - n % LANE_SIZE
    }

    // ------------------------------------ SIMD HELPERS --------------------------------------

    unsafe fn _reg_to_arr(reg: SIMDVecDtype) -> [ScalarDType; LANE_SIZE];

    unsafe fn _mm_loadu(data: *const ScalarDType) -> SIMDVecDtype;

    unsafe fn _mm_set1(a: usize) -> SIMDVecDtype;

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
        let mut min_value: ScalarDType = unsafe { *arr.get_unchecked(0) };
        let mut max_index: usize = 0;
        let mut max_value: ScalarDType = unsafe { *arr.get_unchecked(0) };
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
        // let mut start_: usize = 0;
        // let (mut min_index, mut min_value, mut max_index, mut max_value) =
        //     (0..n_loops).fold(
        //         (0, unsafe { *arr.get_unchecked(0) }, 0, unsafe { *arr.get_unchecked(0) }),
        //         |(min_idx, min_val, max_idx, max_val), i| {
        //             let start = start_;
        //             let mut end = start + dtype_max;
        //             if i == n_loops - 1 {
        //                 end = arr.len();
        //             }
        //             let (min_idx_, min_val_, max_idx_, max_val_) =
        //                 Self::_core_argminmax(&arr[start..end]);
        //             let cmp1 = min_val_.lt(&min_val);
        //             let cmp2 = max_val_.gt(&max_val);
        //             start_ += dtype_max;
        //             (
        //                 if cmp1 { start + min_idx_ } else { min_idx },
        //                 if cmp1 { min_val_ } else { min_val },
        //                 if cmp2 { start + max_idx_ } else { max_idx },
        //                 if cmp2 { max_val_ } else { max_val },
        //             )
        //         },
        //     );
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
        let mut index_low = Self::INITIAL_INDEX;
        let mut index_high = Self::INITIAL_INDEX;

        let increment = Self::_mm_set1(LANE_SIZE);

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
            new_index = Self::_mm_add(new_index, increment);
            // Load the next chunk of data
            arr_ptr = arr_ptr.add(LANE_SIZE);
            // Self::_mm_prefetch(arr_ptr); // Hint to the CPU to prefetch the next chunk of data
            let new_values = Self::_mm_loadu(arr_ptr);

            let lt_mask = Self::_mm_cmplt(new_values, values_low);
            let gt_mask = Self::_mm_cmpgt(new_values, values_high);

            // Update the highest and lowest values
            values_low = Self::_mm_blendv(values_low, new_values, lt_mask);
            values_high = Self::_mm_blendv(values_high, new_values, gt_mask);

            // TODO: check impact of adding index increment here
            // Update the index if the new value is lower/higher
            index_low = Self::_mm_blendv(index_low, new_index, lt_mask);
            index_high = Self::_mm_blendv(index_high, new_index, gt_mask);

            // 25 is a non-scientific number, but seems to work overall
            //  => TODO: probably this should be in function of the data type
            // Self::_mm_prefetch(arr_ptr.add(LANE_SIZE * 25)); // Hint to the CPU to prefetch upcoming data
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

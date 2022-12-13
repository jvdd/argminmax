use crate::task::argminmax_generic;
use crate::utils::{max_index_value, min_index_value};
use ndarray::{s, ArrayView1};
use num_traits::AsPrimitive;

use crate::scalar::{ScalarArgMinMax, SCALAR};

// TODO: other potential generic SIMDIndexDtype: Copy
#[allow(clippy::missing_safety_doc)]  // TODO: add safety docs?
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

    // ------------------------------------ ARGMINMAX --------------------------------------

    unsafe fn argminmax(data: ArrayView1<ScalarDType>) -> (usize, usize);

    #[inline(always)]
    unsafe fn _argminmax(data: ArrayView1<ScalarDType>) -> (usize, usize)
    where
        SCALAR: ScalarArgMinMax<ScalarDType>,
    {
        argminmax_generic(data, LANE_SIZE, Self::_overflow_safe_core_argminmax)
    }

    #[inline(always)]
    unsafe fn _overflow_safe_core_argminmax(
        arr: ArrayView1<ScalarDType>,
    ) -> (usize, ScalarDType, usize, ScalarDType) {
        // 0. Get the max value of the data type - which needs to be divided by LANE_SIZE
        let dtype_max = Self::_find_largest_lower_multiple_of_lane_size(Self::MAX_INDEX);

        // 1. Determine the number of loops needed
        let n_loops = arr.len() / dtype_max;
        // 2. Perform overflow-safe _core_argminmax
        let (mut min_index, mut min_value, mut max_index, mut max_value) =
            arr.exact_chunks(dtype_max).into_iter().enumerate().fold(
                (0, arr[0], 0, arr[0]),
                |(min_index, min_value, max_index, max_value), (i, chunk)| {
                    let (min_index_, min_value_, max_index_, max_value_) =
                        Self::_core_argminmax(chunk);
                    let start = i * dtype_max;
                    let cmp1 = min_value_ < min_value;
                    let cmp2 = max_value_ > max_value;
                    (
                        if cmp1 { min_index_ + start } else { min_index },
                        if cmp1 { min_value_ } else { min_value },
                        if cmp2 { max_index_ + start } else { max_index },
                        if cmp2 { max_value_ } else { max_value },
                    )
                },
            );
        // 3. Handle the remainder
        if n_loops * dtype_max < arr.len() {
            let (min_index_, min_value_, max_index_, max_value_) =
                Self::_core_argminmax(arr.slice(s![n_loops * dtype_max..arr.len()]));
            if min_value_ < min_value {
                min_index = min_index_ + n_loops * dtype_max;
                min_value = min_value_;
            }
            if max_value_ > max_value {
                max_index = max_index_ + n_loops * dtype_max;
                max_value = max_value_;
            }
        }

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
    unsafe fn _core_argminmax(
        arr: ArrayView1<ScalarDType>,
    ) -> (usize, ScalarDType, usize, ScalarDType) {
        assert_eq!(arr.len() % LANE_SIZE, 0);
        // Efficient calculation of argmin and argmax together
        let mut new_index = Self::INITIAL_INDEX;
        let mut index_low = Self::INITIAL_INDEX;
        let mut index_high = Self::INITIAL_INDEX;

        let increment = Self::_mm_set1(LANE_SIZE);

        let new_values = Self::_mm_loadu(arr.as_ptr());
        let mut values_low = new_values;
        let mut values_high = new_values;

        arr.exact_chunks(LANE_SIZE)
            .into_iter()
            .skip(1)
            .for_each(|step| {
                new_index = Self::_mm_add(new_index, increment);

                let new_values = Self::_mm_loadu(step.as_ptr());

                let lt_mask = Self::_mm_cmplt(new_values, values_low);
                let gt_mask = Self::_mm_cmpgt(new_values, values_high);

                index_low = Self::_mm_blendv(index_low, new_index, lt_mask);
                index_high = Self::_mm_blendv(index_high, new_index, gt_mask);

                values_low = Self::_mm_blendv(values_low, new_values, lt_mask);
                values_high = Self::_mm_blendv(values_high, new_values, gt_mask);
            });

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

            unsafe fn argminmax(_data: ArrayView1<$scalar_type>) -> (usize, usize) {
                unimplemented!()
            }
        }
    };
}
#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
pub(crate) use unimplement_simd; // Now classic paths Just Work™

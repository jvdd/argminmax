use crate::task::argminmax_generic;
use crate::utils::{max_index_value, min_index_value};
use ndarray::{s, ArrayView1};
use num_traits::AsPrimitive;

use crate::scalar::{ScalarArgMinMax, SCALAR};

// TODO: other potential generic SIMDIndexDtype: Copy
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
        let n_loops = arr.len().div_ceil(dtype_max);

        // 2. Peform overflow-safe _core_argminmax
        let (mut min_index, mut min_value, mut max_index, mut max_value) = (0, arr[0], 0, arr[0]);
        for i in 0..n_loops {
            let start = i * dtype_max;
            // Unbranch the min using the loop iterator
            let end = start
                + (i == n_loops - 1) as usize * (arr.len() - start)
                + (i != n_loops - 1) as usize * dtype_max;
            // Perform overflow-safe _core_argminmax
            let (min_index_, min_value_, max_index_, max_value_) =
                Self::_core_argminmax(arr.slice(s![start..end]));
            // Update the min and max values
            if min_value_ < min_value {
                min_index = min_index_ + start;
                min_value = min_value_;
            }
            if max_value_ > max_value {
                max_index = max_index_ + start;
                max_value = max_value_;
            }
        }
        (min_index, min_value, max_index, max_value)
    }

    #[inline(always)]
    unsafe fn _get_min_max_index_value(
        index_low: SIMDVecDtype,
        values_low: SIMDVecDtype,
        index_high: SIMDVecDtype,
        values_high: SIMDVecDtype,
    ) -> (usize, ScalarDType, usize, ScalarDType) {
        let values_low_arr = Self::_reg_to_arr(values_low);
        let index_low_arr = Self::_reg_to_arr(index_low);
        let values_high_arr = Self::_reg_to_arr(values_high);
        let index_high_arr = Self::_reg_to_arr(index_high);
        let (min_index, min_value) = min_index_value(&index_low_arr, &values_low_arr);
        let (max_index, max_value) = max_index_value(&index_high_arr, &values_high_arr);
        (min_index.as_(), min_value, max_index.as_(), max_value)
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

        // (min_value, min_index, max_value, max_index)
        Self::_get_min_max_index_value(index_low, values_low, index_high, values_high)
    }
}

#[cfg(target_arch = "arm")]
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
#[cfg(target_arch = "arm")]
pub(crate) use unimplement_simd; // Now classic paths Just Work™

use crate::task::argminmax_generic;
use crate::utils::{max_index_value, min_index_value};
use ndarray::{ArrayView1,s};
use num_traits::{AsPrimitive, Bounded};

// TODO: handle overflow!!

pub trait SIMD<
    ScalarDType: Copy + PartialOrd + AsPrimitive<usize> + Bounded,
    SIMDVecDtype: Copy,
    SIMDMaskDtype: Copy,
    const LANE_SIZE: usize,
>
{
    const INITIAL_INDEX: SIMDVecDtype;

    // ------------------------------------ SIMD HELPERS --------------------------------------

    // TODO: make these unsafe?
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
    unsafe fn _argminmax(data: ArrayView1<ScalarDType>) -> (usize, usize) {
        argminmax_generic(data, LANE_SIZE, Self::_overflow_safe_core_argminmax)
    }

    #[inline(always)]
    unsafe fn _overflow_safe_core_argminmax(
        data: ArrayView1<ScalarDType>,
    ) -> (usize, ScalarDType, usize, ScalarDType) {
        let max = ScalarDType::max_value();
        let n_loops = data.len().div_ceil(max.as_());
        let first_max = std::cmp::min(data.len(), max.as_());
        let (mut min_index, mut min, mut max_index, mut max) = Self::_core_argminmax(data.slice(s![0..first_max]));
        for i in 1..n_loops {
            let start = i * max.as_();
            let end = std::cmp::min(start + max.as_(), data.len());
            let (min_index_i, min_i, max_index_i, max_i) = Self::_core_argminmax(data.slice(s![start..end]));
            if min_i < min {
                min = min_i;
                min_index = min_index_i + start;
            }
            if max_i > max {
                max = max_i;
                max_index = max_index_i + start;
            }
        }
        (min_index, min, max_index, max)
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

macro_rules! unimplement_simd {
    ($scalar_type:ty, $reg:ty, $simd_type:ident) => {
        impl SIMD<$scalar_type, $reg, $reg, 0> for $simd_type {
            const INITIAL_INDEX: $reg = 0;

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
pub(crate) use unimplement_simd; // Now classic paths Just Work™

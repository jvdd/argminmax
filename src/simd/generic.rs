use crate::task::argminmax_generic;
use crate::utils::{max_index_value, min_index_value};
use ndarray::ArrayView1;
use num_traits::AsPrimitive;

// TODO: handle overflow!!

pub trait SIMD<
    DType: Copy + PartialOrd + AsPrimitive<usize>,
    SIMDDtype: Copy,
    const LANE_SIZE: usize,
>
{
    // fn _initial_index() -> SIMDDtype;
    const INITIAL_INDEX: SIMDDtype;

    // ------------------------------------ SIMD HELPERS --------------------------------------

    // TODO: make these unsafe?
    unsafe fn _reg_to_arr(reg: SIMDDtype) -> [DType; LANE_SIZE];

    unsafe fn _mm_load(data: *const DType) -> SIMDDtype;

    unsafe fn _mm_set1(a: usize) -> SIMDDtype;

    unsafe fn _mm_add(a: SIMDDtype, b: SIMDDtype) -> SIMDDtype;

    unsafe fn _mm_cmpgt(a: SIMDDtype, b: SIMDDtype) -> SIMDDtype;

    unsafe fn _mm_cmplt(a: SIMDDtype, b: SIMDDtype) -> SIMDDtype;

    unsafe fn _mm_blendv(a: SIMDDtype, b: SIMDDtype, mask: SIMDDtype) -> SIMDDtype;

    // ------------------------------------ ARGMINMAX --------------------------------------

    unsafe fn argminmax(data: ArrayView1<DType>) -> (usize, usize);

    #[inline(always)]
    unsafe fn _argminmax(data: ArrayView1<DType>) -> (usize, usize) {
        argminmax_generic(data, LANE_SIZE, Self::_core_argminmax)
    }

    #[inline(always)]
    unsafe fn _get_min_max_index_value(index_low: SIMDDtype, values_low: SIMDDtype, index_high: SIMDDtype, values_high: SIMDDtype) -> (usize, DType, usize, DType) {
        let values_low_arr = Self::_reg_to_arr(values_low);
        let index_low_arr = Self::_reg_to_arr(index_low);
        let values_high_arr = Self::_reg_to_arr(values_high);
        let index_high_arr = Self::_reg_to_arr(index_high);
        let (min_index, min_value) = min_index_value(&index_low_arr, &values_low_arr);
        let (max_index, max_value) = max_index_value(&index_high_arr, &values_high_arr);
        (min_index.as_(), min_value, max_index.as_(), max_value)
    }

    // TODO: how to handle the target feature better -> needs to move up
    #[inline(always)]
    // #[target_feature(enable = "avx2")]  // TODO: better inlining when moving this to higher level call (argminmax_generic)
    unsafe fn _core_argminmax(arr: ArrayView1<DType>, offset: usize) -> (usize, DType, usize, DType) {
        // Efficient calculation of argmin and argmax together
        let offset = Self::_mm_set1(offset);
        let mut new_index = Self::_mm_add(Self::INITIAL_INDEX, offset);
        let mut index_low = new_index;
        let mut index_high = new_index;

        let increment = Self::_mm_set1(LANE_SIZE);

        let new_values = Self::_mm_load(arr.as_ptr());
        let mut values_low = new_values;
        let mut values_high = new_values;

        arr.exact_chunks(LANE_SIZE)
            .into_iter()
            .skip(1)
            .for_each(|step| {
                new_index = Self::_mm_add(new_index, increment);

                let new_values = Self::_mm_load(step.as_ptr());

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

use crate::task::argminmax_generic;
use crate::utils::{max_index_value, min_index_value};
use ndarray::ArrayView1;
use num_traits::AsPrimitive;

pub trait SIMD<
    DType: Copy + PartialOrd + AsPrimitive<usize>,
    SIMDDtype: Copy,
    const LANE_SIZE: usize,
>
{
    fn _initial_index() -> SIMDDtype;

    // ------------------------------------ SIMD HELPERS --------------------------------------

    fn _reg_to_arr(reg: SIMDDtype) -> [DType; LANE_SIZE];

    fn _mm_load(data: *const DType) -> SIMDDtype;

    fn _mm_set1(a: usize) -> SIMDDtype;

    fn _mm_add(a: SIMDDtype, b: SIMDDtype) -> SIMDDtype;

    fn _mm_cmpgt(a: SIMDDtype, b: SIMDDtype) -> SIMDDtype;

    fn _mm_cmplt(a: SIMDDtype, b: SIMDDtype) -> SIMDDtype;

    fn _mm_blendv(a: SIMDDtype, b: SIMDDtype, mask: SIMDDtype) -> SIMDDtype;

    // ------------------------------------ ARGMINMAX --------------------------------------

    fn argminmax(data: ArrayView1<DType>) -> (usize, usize) {
        argminmax_generic(data, LANE_SIZE, Self::_core_argminmax)
    }

    fn _core_argminmax(arr: ArrayView1<DType>, offset: usize) -> (DType, usize, DType, usize) {
        // Efficient calculation of argmin and argmax together
        let offset = Self::_mm_set1(offset);
        let mut new_index = Self::_mm_add(Self::_initial_index(), offset);
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

        // Select the min_index and min_value
        let value_array = Self::_reg_to_arr(values_low);
        let index_array = Self::_reg_to_arr(index_low);
        let (min_index, min_value) = min_index_value(&value_array, &index_array);

        // Select the max_index and max_value
        let value_array = Self::_reg_to_arr(values_high);
        let index_array = Self::_reg_to_arr(index_high);
        let (max_index, max_value) = max_index_value(&value_array, &index_array);

        let min_index: usize = min_index.as_();
        let max_index: usize = max_index.as_();

        (min_value, min_index as usize, max_value, max_index as usize)
    }
}

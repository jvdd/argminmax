// ------ On &[T]

// Note: these two functions are necessary because in the final SIMD registers the
// indexes are not in sorted order - this means that the first index (in the SIMD
// registers) is not necessarily the lowest min / max index when the min / max value
// occurs multiple times.

// #[inline(always)]
// pub(crate) fn min_index_value<T: Copy + PartialOrd>(index: &[T], values: &[T]) -> (T, T) {
//     assert_eq!(index.len(), values.len());
//     let mut min_index: usize = 0;
//     let mut min_value = values[min_index];
//     for (i, value) in values.iter().enumerate() {
//         // No skip(1) here bc 5-7% slower
//         if *value < min_value {
//             min_value = *value;
//             min_index = i;
//         } else if *value == min_value && index[i] < index[min_index] {
//             min_index = i;
//         }
//     }
//     (index[min_index], min_value)
// }

#[inline(always)]
pub(crate) fn min_index_value<T: Copy + PartialOrd>(index: &[T], values: &[T]) -> (T, T) {
    assert_eq!(index.len(), values.len());
    values
        .iter()
        .enumerate()
        .fold((index[0], values[0]), |(min_idx, min), (idx, item)| {
            if *item < min || (*item == min && index[idx] < min_idx) {
                (index[idx], *item)
            } else {
                (min_idx, min)
            }
        })
}

// #[inline(always)]
// pub(crate) fn max_index_value__<T: Copy + PartialOrd>(index: &[T], values: &[T]) -> (T, T) {
//     assert_eq!(index.len(), values.len());
//     let mut max_index: usize = 0;
//     let mut max_value = values[max_index];
//     for (i, value) in values.iter().enumerate() {
//         // No skip(1) here bc 5-7% slower
//         if *value > max_value {
//             max_value = *value;
//             max_index = i;
//         } else if *value == max_value && index[i] < index[max_index] {
//             max_index = i;
//         }

//         // // Unbranching version of the above
//         // let gt_mask = *value > values[max_index];
//         // let smaller_index_mask = *value == values[max_index] && index[i] < index[max_index];
//         // let cond = (gt_mask | smaller_index_mask) as usize;
//         // let not_cond = (cond + 1) % 2;
//         // max_index = (max_index * not_cond) + (i * cond);
//     }
//     (index[max_index], max_value)
// }

#[inline(always)]
pub(crate) fn max_index_value<T: Copy + PartialOrd>(index: &[T], values: &[T]) -> (T, T) {
    assert_eq!(index.len(), values.len());
    values
        .iter()
        .enumerate()
        .fold((index[0], values[0]), |(max_idx, max), (idx, item)| {
            if *item > max || (*item == max && index[idx] < max_idx) {
                (index[idx], *item)
            } else {
                (max_idx, max)
            }
        })
}

// ------ On &[T]

// Note: these two functions are necessary because in the final SIMD registers the
// indexes are not in sorted order - this means that the first index (in the SIMD
// registers) is not necessarily the lowest min / max index when the min / max value
// occurs multiple times.

#[inline(always)]
pub(crate) fn min_index_value<T: Copy + PartialOrd>(index: &[T], values: &[T]) -> (T, T) {
    assert_eq!(index.len(), values.len());
    let mut min_index: usize = 0;
    let mut min_value = values[min_index];
    for (i, value) in values.iter().skip(1).enumerate() {
        if *value < min_value {
            min_value = *value;
            min_index = i + 1;
        } else if *value == min_value && index[i + 1] < index[min_index] {
            min_index = i + 1;
        }
    }
    (index[min_index], min_value)
}

#[inline(always)]
pub(crate) fn max_index_value<T: Copy + PartialOrd>(index: &[T], values: &[T]) -> (T, T) {
    assert_eq!(index.len(), values.len());
    let mut max_index: usize = 0;
    let mut max_value = values[max_index];
    for (i, value) in values.iter().skip(1).enumerate() {
        if *value > max_value {
            max_value = *value;
            max_index = i + 1;
        } else if *value == max_value && index[i + 1] < index[max_index] {
            max_index = i + 1;
        }
    }
    (index[max_index], max_value)
}

use ndarray::ArrayView1;

// ------ On ArrayView1

#[inline]
pub fn simple_argmin<T: Copy + PartialOrd>(arr: ArrayView1<T>) -> usize {
    let mut low_index = 0usize;
    let mut low = arr[low_index];
    for (i, item) in arr.iter().enumerate() {
        if *item < low {
            low = *item;
            low_index = i;
        }
    }
    low_index
}

#[inline]
pub fn simple_argmax<T: Copy + PartialOrd>(arr: ArrayView1<T>) -> usize {
    let mut high_index = 0usize;
    let mut high = arr[high_index];
    for (i, item) in arr.iter().enumerate() {
        if *item > high {
            high = *item;
            high_index = i;
        }
    }
    high_index
}

#[inline]
pub fn simple_argminmax<T: Copy + PartialOrd>(arr: ArrayView1<T>) -> (usize, usize) {
    let mut low_index: usize = 0;
    let mut high_index: usize = 0;
    let mut low = arr[low_index];
    let mut high = arr[high_index];
    for (i, item) in arr.iter().enumerate() {
        if *item < low {
            low = *item;
            low_index = i;
        } else if *item > high {
            high = *item;
            high_index = i;
        }
    }
    (low_index, high_index)
}

// ------ On &[T]

// Note: these two functions are necessary because in the final SIMD registers the
// indexes are not in sorted order - this means that the first index (in the SIMD
// registers) is not necessarily the lowest min / max index when the min / max value
// occurs multiple times.

#[inline]
pub fn min_index_value<T: Copy + PartialOrd>(index: &[T], values: &[T]) -> (T, T) {
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

#[inline]
pub fn max_index_value<T: Copy + PartialOrd>(index: &[T], values: &[T]) -> (T, T) {
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

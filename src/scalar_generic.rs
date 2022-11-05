use ndarray::ArrayView1;

// ------ On ArrayView1

#[inline]
pub fn scalar_argmin<T: Copy + PartialOrd>(arr: ArrayView1<T>) -> usize {
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
pub fn scalar_argmax<T: Copy + PartialOrd>(arr: ArrayView1<T>) -> usize {
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
pub fn scalar_argminmax<T: Copy + PartialOrd>(arr: ArrayView1<T>) -> (usize, usize) {
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

// Note: 5-7% faster than the above implementation (for floats)
// #[inline]
// pub fn scalar_argminmax_fold<T: Copy + PartialOrd>(arr: ArrayView1<T>) -> (usize, usize) {
//     let minmax_tuple: (usize, T, usize, T) = arr.iter().enumerate().fold(
//         (0usize, arr[0], 0usize, arr[0]),
//         |(min_idx, min, max_idx, max), (idx, item)| {
//             if *item < min {
//                 (idx, *item, max_idx, max)
//             } else if *item > max {
//                 (min_idx, min, idx, *item)
//             } else {
//                 (min_idx, min, max_idx, max)
//             }
//         },
//     );
//     (minmax_tuple.0, minmax_tuple.2)
// }
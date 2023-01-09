#[cfg(feature = "half")]
use super::scalar_f16::scalar_argminmax_f16;
#[cfg(feature = "half")]
use half::f16;

// ------ On ArrayView1

// #[inline]
// pub fn scalar_argmin<T: Copy + PartialOrd>(arr: ArrayView1<T>) -> usize {
//     let mut low_index = 0usize;
//     let mut low = arr[low_index];
//     for (i, item) in arr.iter().enumerate() {
//         if *item < low {
//             low = *item;
//             low_index = i;
//         }
//     }
//     low_index
// }

// #[inline]
// pub fn scalar_argmax<T: Copy + PartialOrd>(arr: ArrayView1<T>) -> usize {
//     let mut high_index = 0usize;
//     let mut high = arr[high_index];
//     for (i, item) in arr.iter().enumerate() {
//         if *item > high {
//             high = *item;
//             high_index = i;
//         }
//     }
//     high_index
// }

pub trait ScalarArgMinMax<ScalarDType: Copy + PartialOrd> {
    fn argminmax(data: &[ScalarDType]) -> (usize, usize);
}

pub struct SCALAR;

#[inline(always)]
pub fn scalar_argminmax<T: Copy + PartialOrd>(arr: &[T]) -> (usize, usize) {
    let minmax_tuple: (usize, T, usize, T) = arr.iter().enumerate().fold(
        (0usize, arr[0], 0usize, arr[0]),
        |(min_idx, min, max_idx, max), (idx, item)| {
            if *item < min {
                (idx, *item, max_idx, max)
            } else if *item > max {
                (min_idx, min, idx, *item)
            } else {
                (min_idx, min, max_idx, max)
            }
        },
    );
    (minmax_tuple.0, minmax_tuple.2)
}

macro_rules! impl_scalar {
    ($func:ident, $($t:ty),*) =>
    {
        $(
            impl ScalarArgMinMax<$t> for SCALAR {
                fn argminmax(data: &[$t]) -> (usize, usize) {
                    $func(data)
                }
            }
        )*
    };
}

impl_scalar!(
    scalar_argminmax,
    i8,
    i16,
    i32,
    i64,
    u8,
    u16,
    u32,
    u64,
    f32,
    f64
);
#[cfg(feature = "half")]
impl_scalar!(scalar_argminmax_f16, f16);

use ndarray::ArrayView1;
#[cfg(feature = "half")]
use half::f16;

// ------ On ArrayView1

#[cfg(feature = "half")]
#[inline]
fn f16_to_i16ord(x: f16) -> i16 {
    let x = unsafe { std::mem::transmute::<f16, i16>(x) };
    ((x >> 15) & 0x7FFF) ^ x
}

#[cfg(feature = "half")]
#[inline]
pub fn simple_argminmax_f16(arr: ArrayView1<f16>) -> (usize, usize) {
    // f16 is transformed to i16ord
    //   benchmarks  show:
    //     1. this is 7-10x faster than using raw f16
    //     2. this is 3x faster than transforming to f32 or f64
    let mut low_index: usize = 0;
    let mut high_index: usize = 0;
    let mut low = f16_to_i16ord(arr[low_index]);
    let mut high = f16_to_i16ord(arr[high_index]);
    for (i, item) in arr.iter().enumerate() {
        let item = f16_to_i16ord(*item);
        if item < low {
            low = item;
            low_index = i;
        } else if item > high {
            high = item;
            high_index = i;
        }
    }
    (low_index, high_index)
}

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
pub fn scalar_argminmax_f16(arr: ArrayView1<f16>) -> (usize, usize) {
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

#[cfg(feature = "half")]
#[cfg(test)]
mod tests {
    use super::scalar_argminmax_f16;
    use crate::scalar_generic::scalar_argminmax;

    use half::f16;
    use ndarray::Array1;

    extern crate dev_utils;
    use dev_utils::utils;

    fn get_array_f16(n: usize) -> Array1<f16> {
        let arr = utils::get_random_array(n, i16::MIN, i16::MAX);
        let arr = arr.mapv(|x| f16::from_f32(x as f32));
        Array1::from(arr)
    }

    #[test]
    fn test_generic_and_specific_impl_return_the_same_results() {
        for _ in 0..100 {
            let data = get_array_f16(1025);
            let (argmin_index, argmax_index) = scalar_argminmax(data.view());
            let (argmin_simd_index, argmax_simd_index) = scalar_argminmax_f16(data.view());
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }
    }
}

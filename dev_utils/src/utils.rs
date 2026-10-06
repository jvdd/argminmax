use half::f16;
use num_traits::Zero;

use std::ops::{Add, Sub};

use rand::distr::Uniform;
use rand::rngs::ThreadRng;
use rand::RngExt;

// worst case array that alternates between increasing max and decreasing min values
pub fn get_worst_case_array<T>(n: usize, step: T) -> Vec<T>
where
    T: Copy + Default + Sub<Output = T> + Add<Output = T>,
{
    let mut arr: Vec<T> = Vec::with_capacity(n);
    let mut min_value: T = Default::default();
    let mut max_value: T = Default::default();
    for i in 0..n {
        if i % 2 == 0 {
            arr.push(min_value);
            min_value = min_value - step;
        } else {
            arr.push(max_value);
            max_value = max_value + step;
        }
    }
    arr
}

pub trait SampleUniformFullRange: Sized {
    const MIN: Self;
    const MAX: Self;

    fn get_random_array(n: usize) -> Vec<Self>;
}

macro_rules! impl_full_range_uniform {
    ($($t:ty),*) => {
        $(
            impl SampleUniformFullRange for $t {
                const MIN: Self = <$t>::MIN;
                const MAX: Self = <$t>::MAX;

                fn get_random_array(n: usize) -> Vec<Self> {
                    let rng = ThreadRng::default();
                    let uni = Uniform::new_inclusive(Self::MIN, Self::MAX).unwrap();
                    rng.sample_iter(uni).take(n).collect()
                }
            }
        )*
    };
}

macro_rules! impl_full_range_uniform_float {
    ($($t:ty, $t_int:ty),*) => {
        $(
            impl SampleUniformFullRange for $t {
                const MIN: Self = <$t>::MIN;
                const MAX: Self = <$t>::MAX;

                fn get_random_array(n: usize) -> Vec<Self> {
                    // Generate random integers and transmute to floats to avoid
                    // range overflow issues with Uniform distribution for floats.
                    // NaNs and infinities are replaced by 0 to stay within [MIN, MAX].
                    let rand_arr_int: Vec<$t_int> = <$t_int>::get_random_array(n);
                    let rand_arr_float: Vec<Self> = unsafe { std::mem::transmute(rand_arr_int) };
                    rand_arr_float.iter().map(|x| if x.is_finite() { *x } else { <$t>::zero() }).collect()
                }
            }
        )*
    };
}

impl_full_range_uniform!(i8, i16, i32, i64, u8, u16, u32, u64);
// f16, f32, f64 use integer transmutation to avoid Uniform range overflow / SampleUniform dependency
impl_full_range_uniform_float!(f16, i16, f32, i32, f64, i64);

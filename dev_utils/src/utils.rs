use half::f16;
use num_traits::Zero;

use std::ops::{Add, Sub};

use rand::distr::Uniform;
use rand::{rng, Rng};

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

pub trait SampleUniformFullRange: rand::distr::uniform::SampleUniform {
    const MIN: Self;
    const MAX: Self;

    // random array that samples between min and max of Self
    fn get_random_array(n: usize) -> Vec<Self>
    where
        Self: Copy + rand::distr::uniform::SampleUniform,
    {
        let rng = rng();
        let uni = Uniform::new_inclusive(Self::MIN, Self::MAX).unwrap();
        rng.sample_iter(uni).take(n).collect()
    }
}

macro_rules! impl_full_range_uniform {
    ($($t:ty),*) => {
        $(
            impl SampleUniformFullRange for $t {
                const MIN: Self = <$t>::MIN;
                const MAX: Self = <$t>::MAX;
            }
        )*
    };
}

macro_rules! impl_full_range_uniform_float {
    ($($t:ty, $t_int:ty),*) => {
        $(
            impl SampleUniformFullRange for $t {
                // These 2 are not used, but are required by the trait
                const MIN: Self = <$t>::MIN;
                const MAX: Self = <$t>::MAX;

                fn get_random_array(n: usize) -> Vec<Self>
                where
                    Self: Copy + rand::distr::uniform::SampleUniform,
                {
                    // Get a uniform random array of integers
                    let rand_arr_int: Vec<$t_int> = <$t_int>::get_random_array(n);
                    // Transmute the integers to floats
                    let rand_arr_float: Vec<Self> = unsafe { std::mem::transmute(rand_arr_int) };
                    // Replace the NaNs with 0.0
                    rand_arr_float.iter().map(|x| if x.is_nan() { <$t>::zero() } else { *x }).collect()
                }
            }
        )*
    };
}

impl_full_range_uniform!(i8, i16, i32, i64, u8, u16, u32, u64);
// f16 does not suffer from range overflow panick as upcast to f32 is used to generate
// the random numbers
impl_full_range_uniform!(f16);
// Workaround for f32 and f64 as these suffer from range overflow in the rand crate
impl_full_range_uniform_float!(f32, i32, f64, i64);

mod task;
mod utils;
mod scalar;
mod simd;

pub use scalar::scalar_generic::*;
pub use scalar::scalar_f16::*;
// TODO: fix simd package private vs pub crate etc.
// pub use simd::{simd_f32, simd_f64, simd_i16, simd_i32, simd_i64};

pub use simd::{SIMD,AVX2};
use ndarray::ArrayView1;

pub trait ArgMinMax {
    // TODO: future work implement these other functions
    // fn min(self) -> Self::Item;
    // fn max(self) -> Self::Item;
    // fn minmax(self) -> (T, T);

    // fn argmin(self) -> usize;
    // fn argmax(self) -> usize;
    fn argminmax(self) -> (usize, usize);
}

macro_rules! impl_argminmax {
    ($($t:ty),*) => {
        $(
            impl ArgMinMax for ArrayView1<'_, $t> {
                fn argminmax(self) -> (usize, usize) {
                    AVX2::argminmax(self)
                }
            }
        )*
    };
}

// Implement ArgMinMax for the rust primitive types
impl_argminmax!(i16, i32, i64, f32, f64);

// Implement ArgMinMax for other data types
#[cfg(feature = "half")]
use half::f16;
#[cfg(feature = "half")]
impl_argminmax!(f16);

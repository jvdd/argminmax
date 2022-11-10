mod scalar;
mod simd;
mod task;
mod utils;

pub use scalar::scalar_f16::*;
pub use scalar::scalar_generic::*;
// TODO: fix simd package private vs pub crate etc.
// pub use simd::{simd_f32, simd_f64, simd_i16, simd_i32, simd_i64};

use ndarray::ArrayView1;
pub use simd::{AVX2, AVX512, SIMD, SSE};

pub trait ArgMinMax {
    // TODO: future work implement these other functions?
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
                    // select avx2 if available
                    if is_x86_feature_detected!("avx2") {
                        return unsafe { AVX2::argminmax(self) };
                    // } else if is_x86_feature_detected!("avx512f") {
                        // unsafe { AVX512::argminmax(self) }
                    } else if is_x86_feature_detected!("sse4.1") {
                        return unsafe { SSE::argminmax(self) };
                    } else {
                        return scalar_argminmax(self);
                    }
                }
            }
        )*
    };
}

// Implement ArgMinMax for the rust primitive types
impl_argminmax!(i16, i32, i64, f32, f64, u16, u32, u64);

macro_rules! impl_argminmax_simd {
    ($simd:ident, $($t:ty),*) => {
        $(
            impl ArgMinMax for ArrayView1<'_, $t> {
                fn argminmax(self) -> (usize, usize) {
                    unsafe { $simd::argminmax(self) }
                }
            }
        )*
    };
}

// #[cfg(
//     all(
//         any(target_arch = "x86", target_arch = "x86_64"),
//         target_feature = "avx2"
//     )
// )]
// impl_argminmax_simd!(AVX2, i16, i32, i64, f32, f64, u16, u32, u64);

// #[cfg(
//     all(
//         any(target_arch = "x86", target_arch = "x86_64"),
//         target_feature = "sse4.1",
//         not(target_feature = "avx2")
//     )
// )]
// impl_argminmax_simd!(SSE, i16, i32, i64, f32, f64, u16, u32, u64);

// Implement ArgMinMax for other data types
#[cfg(feature = "half")]
use half::f16;
#[cfg(feature = "half")]
impl_argminmax!(f16);

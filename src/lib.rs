#![feature(stdsimd)]
#![feature(avx512_target_feature)]

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

trait NbBits {
    const NB_BITS: usize;
}

pub trait ArgMinMax {
    // TODO: future work implement these other functions?
    // fn min(self) -> Self::Item;
    // fn max(self) -> Self::Item;
    // fn minmax(self) -> (T, T);

    // fn argmin(self) -> usize;
    // fn argmax(self) -> usize;
    fn argminmax(self) -> (usize, usize);
}

macro_rules! impl_nb_bits {
    ($($t:ty)*) => ($(
        impl NbBits for $t {
            const NB_BITS: usize = std::mem::size_of::<$t>() * 8;
        }
    )*)
}

macro_rules! impl_argminmax {
    ($($t:ty),*) => {
        $(
            impl ArgMinMax for ArrayView1<'_, $t> {
                fn argminmax(self) -> (usize, usize) {
                    if is_x86_feature_detected!("avx512f") & (<$t>::NB_BITS > 16){
                        // for some reason is avx512f a lot slower than avx2 for 16 bit numbers
                        return unsafe { AVX512::argminmax(self) }
                    } else if is_x86_feature_detected!("avx2") {
                        return unsafe { AVX2::argminmax(self) }
                    } else if is_x86_feature_detected!("sse4.1") & (<$t>::NB_BITS < 64) {
                        // for some reason is sse4.1 a lot slower than scalar for 64 bit numbers
                        return unsafe { SSE::argminmax(self) }
                    } else {
                        return scalar_argminmax(self);
                    }
                }
            }
        )*
    };
}

// Implement ArgMinMax for the rust primitive types
impl_nb_bits!(u16 u32 u64 i16 i32 i64 f32 f64);
impl_argminmax!(i16, i32, i64, f32, f64, u16, u32, u64);

// macro_rules! impl_argminmax_simd {
//     ($simd:ident, $($t:ty),*) => {
//         $(
//             impl ArgMinMax for ArrayView1<'_, $t> {
//                 fn argminmax(self) -> (usize, usize) {
//                     unsafe { $simd::argminmax(self) }
//                 }
//             }
//         )*
//     };
// }

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

// #[cfg(
//     all(
//         not(target_feature = "sse4.1"),
//         not(target_feature = "avx2")
//     )
// )]
// impl_argminmax!(i16, i32, i64, f32, f64, u16, u32, u64);

// Implement ArgMinMax for other data types
#[cfg(feature = "half")]
mod half_impl {
    use super::*;
    use half::f16;

    impl_nb_bits!(f16);
    impl_argminmax!(f16);
}

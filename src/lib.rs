#![feature(stdsimd)]
#![feature(avx512_target_feature)]
#![feature(arm_target_feature)]
#![feature(int_roundings)]

// #[macro_use]
// extern crate lazy_static;

mod scalar;
mod simd;
mod task;
mod utils;

use ndarray::ArrayView1;
pub use scalar::{ScalarArgMinMax, SCALAR};
pub use simd::{AVX2, AVX512, NEON, SIMD, SSE};

trait DTypeInfo {
    const NB_BITS: usize;
    const IS_FLOAT: bool;
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
    ($is_float:expr, $($t:ty)*) => ($(
        impl DTypeInfo for $t {
            const NB_BITS: usize = std::mem::size_of::<$t>() * 8;
            const IS_FLOAT: bool = $is_float;
        }
    )*)
}

// use once_cell::sync::Lazy;

// #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
// static AVX512BW_DETECTED: Lazy<bool> = Lazy::new(|| is_x86_feature_detected!("avx512bw"));
// #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
// static AVX512F_DETECTED: Lazy<bool> = Lazy::new(|| is_x86_feature_detected!("avx512f"));
// #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
// static AVX2_DETECTED: Lazy<bool> = Lazy::new(|| is_x86_feature_detected!("avx2"));
// #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
// static AVX_DETECTED: Lazy<bool> = Lazy::new(|| is_x86_feature_detected!("avx"));
// #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
// static SSE_DETECTED: Lazy<bool> = Lazy::new(|| is_x86_feature_detected!("sse4.1"));
// #[cfg(target_arch = "arm")]
// static NEON_DETECTED: Lazy<bool> = Lazy::new(|| std::arch::is_arm_feature_detected!("neon"));

// macro_rules! impl_argminmax {
//     ($($t:ty),*) => {
//         $(
//             impl ArgMinMax for ArrayView1<'_, $t> {
//                 fn argminmax(self) -> (usize, usize) {
//                     #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
//                     {
//                         if *AVX512BW_DETECTED & (<$t>::NB_BITS <= 16) {
//                             // BW (ByteWord) instructions are needed for 16-bit avx512
//                             return unsafe { AVX512::argminmax(self) }
//                         } else if *AVX512F_DETECTED {  // TODO: check if avx512bw is included in avx512f
//                             return unsafe { AVX512::argminmax(self) }
//                         } else if *AVX2_DETECTED {
//                             return unsafe { AVX2::argminmax(self) }
//                         } else if *AVX_DETECTED & (<$t>::NB_BITS >= 32) & (<$t>::IS_FLOAT == true) {
//                             // f32 and f64 do not require avx2
//                             return unsafe { AVX2::argminmax(self) }
//                         // SKIP SSE4.2 bc scalar is faster or equivalent for 64 bit numbers
//                         // // } else if is_x86_feature_detected!("sse4.2") & (<$t>::NB_BITS == 64) & (<$t>::IS_FLOAT == false) {
//                         //     // SSE4.2 is needed for comparing 64-bit integers
//                         //     return unsafe { SSE::argminmax(self) }
//                         } else if *SSE_DETECTED & (<$t>::NB_BITS < 64) {
//                             // Scalar is faster for 64-bit numbers
//                             return unsafe { SSE::argminmax(self) }
//                         }
//                     }
//                     #[cfg(target_arch = "aarch64")]
//                     {
//                         // TODO: support aarch64
//                     }
//                     #[cfg(target_arch = "arm")]
//                     {
//                         if *NEON_DETECTED & (<$t>::NB_BITS < 32) {
//                             // TODO: requires v7?
//                             // We miss some NEON instructions for 64-bit numbers
//                             return unsafe { NEON::argminmax(self) }
//                         }
//                     }
//                     SCALAR::argminmax(self)
//                 }
//             }
//         )*
//     };
// }

macro_rules! impl_argminmax {
    ($($t:ty),*) => {
        $(
            impl ArgMinMax for ArrayView1<'_, $t> {
                fn argminmax(self) -> (usize, usize) {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    {
                        if is_x86_feature_detected!("avx512bw") & (<$t>::NB_BITS <= 16) {
                            // BW (ByteWord) instructions are needed for 16-bit avx512
                            return unsafe { AVX512::argminmax(self) }
                        } else if is_x86_feature_detected!("avx512f") {  // TODO: check if avx512bw is included in avx512f
                            return unsafe { AVX512::argminmax(self) }
                        } else if is_x86_feature_detected!("avx2") {
                            return unsafe { AVX2::argminmax(self) }
                        } else if is_x86_feature_detected!("avx")  & (<$t>::NB_BITS >= 32) & (<$t>::IS_FLOAT == true) {
                            // f32 and f64 do not require avx2
                            return unsafe { AVX2::argminmax(self) }
                        // SKIP SSE4.2 bc scalar is faster or equivalent for 64 bit numbers
                        // // } else if is_x86_feature_detected!("sse4.2") & (<$t>::NB_BITS == 64) & (<$t>::IS_FLOAT == false) {
                        //     // SSE4.2 is needed for comparing 64-bit integers
                        //     return unsafe { SSE::argminmax(self) }
                        } else if is_x86_feature_detected!("sse4.1") & (<$t>::NB_BITS < 64) {
                            // Scalar is faster for 64-bit numbers
                            return unsafe { SSE::argminmax(self) }
                        }
                    }
                    #[cfg(target_arch = "aarch64")]
                    {
                        if std::arch::is_aarch64_feature_detected!("neon") & (<$t>::NB_BITS < 64) {
                            // We miss some NEON instructions for 64-bit numbers
                            return unsafe { NEON::argminmax(self) }
                        }
                    }
                    #[cfg(target_arch = "arm")]
                    {
                        if std::arch::is_arm_feature_detected!("neon") & (<$t>::NB_BITS < 32) {
                            // TODO: requires v7?
                            // We miss some NEON instructions for 64-bit numbers
                            return unsafe { NEON::argminmax(self) }
                        }
                    }
                    SCALAR::argminmax(self)
                }
            }
        )*
    };
}

// Implement ArgMinMax for the rust primitive types
impl_nb_bits!(false, u16 u32 u64 i16 i32 i64);
impl_nb_bits!(true, f32 f64);
impl_argminmax!(i16, i32, i64, f32, f64, u16, u32, u64);

// Implement ArgMinMax for other data types
#[cfg(feature = "half")]
mod half_impl {
    use super::*;
    use half::f16;

    impl_nb_bits!(true, f16);
    impl_argminmax!(f16);
}

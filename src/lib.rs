#![feature(stdsimd)]
#![feature(avx512_target_feature)]
#![feature(arm_target_feature)]

// It is necessary to import this at the root of the crate
// See: https://github.com/la10736/rstest/tree/master/rstest_reuse#use-rstest_resuse-at-the-top-of-your-crate
#[cfg(test)]
use rstest_reuse;

// #[macro_use]
// extern crate lazy_static;

mod dtype_strategy;
mod scalar;
mod simd;

pub use dtype_strategy::Int;
#[cfg(any(feature = "float", feature = "half"))]
pub use dtype_strategy::{FloatIgnoreNaN, FloatReturnNaN};
pub use scalar::{ScalarArgMinMax, SCALAR};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use simd::{SIMDArgMinMax, AVX2, AVX512, SSE};
#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
pub use simd::{SIMDArgMinMax, NEON};

#[cfg(feature = "half")]
use half::f16;

/// Trait for finding the minimum and maximum values in an array.
/// This trait is implemented for slices (or other array-like) of integers and floats*.
///
/// *Note that the trait is only implemented for floats when the default "float" feature
/// is enabled.
/// If you want to use the `argminmax` function with f16, you need to enable the "half"
/// feature.
///
pub trait ArgMinMax {
    // TODO: future work implement these other functions?
    // fn min(self) -> Self::Item;
    // fn max(self) -> Self::Item;
    // fn minmax(self) -> (T, T);

    // fn argmin(self) -> usize;
    // fn argmax(self) -> usize;

    /// Get the index of the minimum and maximum values in the array.
    ///
    /// When dealing with floats, NaNs are ignored.
    /// Note that this differs from numpy, where the `argmin` and `argmax` functions
    /// return the index of the first NaN (which is the behavior of our nanargminmax
    /// function).
    ///
    /// # Caution
    ///  When a float array contains *only* NaNs and / or infinities unexpected behavior
    ///  may occur (in which case index 0 is returned for both).
    ///
    fn argminmax(&self) -> (usize, usize);
}

/// Trait for finding the minimum and maximum values in an array.
/// This trait is implemented for slices (or other array-like) of floats*.
///
/// *Note that the trait is only implemented for floats when the default "float" feature
/// is enabled.
///
#[cfg(any(feature = "float", feature = "half"))]
pub trait NaNArgMinMax {
    /// Get the index of the minimum and maximum values in the array.
    ///
    /// When dealing with floats, NaNs are propagated (i.e. returned) - in other words,
    /// the index of the first NaN is returned.
    /// Note that this differs from numpy, where the `nanargmin` and `nanargmax`
    /// functions ignore NaNs (which is the behavior of our argminmax function).
    ///
    /// # Caution
    ///  When multiple bit-representations for NaNs are used, no guarantee is made
    ///  that the first NaN is returned.
    ///
    fn nanargminmax(&self) -> (usize, usize);
}

// ---- Helper macros ----

trait DTypeInfo {
    const NB_BITS: usize;
}

/// Macro for implementing DTypeInfo for the passed data types (uints, ints, floats)
macro_rules! impl_nb_bits {
    // $data_type is the data type (e.g. i32)
    // you can pass multiple types (separated by commas) to this macro
    ($($data_type:ty)*) => ($(
        impl DTypeInfo for $data_type {
            const NB_BITS: usize = std::mem::size_of::<$data_type>() * 8;
        }
    )*)
}

impl_nb_bits!(i8 i16 i32 i64 u8 u16 u32 u64);
#[cfg(feature = "float")]
impl_nb_bits!(f32 f64);
#[cfg(feature = "half")]
impl_nb_bits!(f16);

// ------------------------------ &[T] ------------------------------

/// Macro for implementing ArgMinMax for signed and unsigned integers
macro_rules! impl_argminmax_int {
    // $int_type is the integer data type of the array (e.g. i32)
    // you can pass multiple types (separated by commas) to this macro
    ($($int_type:ty),*) => {
        $(
            impl ArgMinMax for &[$int_type] {
                fn argminmax(&self) -> (usize, usize) {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    {
                        if is_x86_feature_detected!("sse4.1") & (<$int_type>::NB_BITS == 8) {
                            // 8-bit numbers are best handled by SSE4.1
                            return unsafe { SSE::<Int>::argminmax(self) }
                        } else if is_x86_feature_detected!("avx512bw") & (<$int_type>::NB_BITS <= 16) {
                            // BW (ByteWord) instructions are needed for 8 or 16-bit avx512
                            return unsafe { AVX512::<Int>::argminmax(self) }
                        } else if is_x86_feature_detected!("avx512f") {  // TODO: check if avx512bw is included in avx512f
                            return unsafe { AVX512::<Int>::argminmax(self) }
                        } else if is_x86_feature_detected!("avx2") {
                            return unsafe { AVX2::<Int>::argminmax(self) }
                        // SKIP SSE4.2 bc scalar is faster or equivalent for 64 bit numbers
                        // // } else if is_x86_feature_detected!("sse4.2") & (<$int_type>::NB_BITS == 64) & (<$int_type>::IS_FLOAT == false) {
                        //     // SSE4.2 is needed for comparing 64-bit integers
                        //     return unsafe { SSE::argminmax(self) }
                        } else if is_x86_feature_detected!("sse4.1") & (<$int_type>::NB_BITS < 64) {
                            // Scalar is faster for 64-bit numbers
                            return unsafe { SSE::<Int>::argminmax(self) }
                        }
                    }
                    #[cfg(target_arch = "aarch64")]
                    {
                        if std::arch::is_aarch64_feature_detected!("neon") & (<$int_type>::NB_BITS < 64) {
                            // We miss some NEON instructions for 64-bit numbers
                            return unsafe { NEON::<Int>::argminmax(self) }
                        }
                    }
                    #[cfg(target_arch = "arm")]
                    {
                        if std::arch::is_arm_feature_detected!("neon") & (<$int_type>::NB_BITS < 64) {
                            // TODO: requires v7?
                            // We miss some NEON instructions for 64-bit numbers
                            return unsafe { NEON::<Int>::argminmax(self) }
                        }
                    }
                    SCALAR::<Int>::argminmax(self)
                }
            }
        )*
    };
}

/// Macro for implementing ArgMinMax for floats
#[cfg(any(feature = "float", feature = "half"))]
macro_rules! impl_argminmax_float {
    // $float_type is the float data type of the array (e.g. f32)
    // you can pass multiple types (separated by commas) to this macro
    ($($float_type:ty),*) => {
        $(
            impl ArgMinMax for &[$float_type] {
                fn argminmax(&self) -> (usize, usize) {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    {
                        if <$float_type>::NB_BITS <= 16 {
                            // TODO: f16 IgnoreNaN is not yet SIMD-optimized
                            // do nothing (defaults to scalar)
                        } else if is_x86_feature_detected!("avx512f") {
                            return unsafe { AVX512::<FloatIgnoreNaN>::argminmax(self) }
                        } else if is_x86_feature_detected!("avx") {
                            // f32 and f64 do not require avx2
                            return unsafe { AVX2::<FloatIgnoreNaN>::argminmax(self) }
                        } else if is_x86_feature_detected!("sse4.1") & (<$float_type>::NB_BITS < 64) {
                            // Scalar is faster for 64-bit numbers
                            return unsafe { SSE::<FloatIgnoreNaN>::argminmax(self) }
                        }
                    }
                    #[cfg(target_arch = "aarch64")]
                    {
                        if std::arch::is_aarch64_feature_detected!("neon") & (<$float_type>::NB_BITS < 64) {
                            // We miss some NEON instructions for 64-bit numbers
                            return unsafe { NEON::<FloatIgnoreNaN>::argminmax(self) }
                        }
                    }
                    #[cfg(target_arch = "arm")]
                    {
                        if std::arch::is_arm_feature_detected!("neon") & (<$float_type>::NB_BITS < 64) {
                            // TODO: requires v7?
                            // We miss some NEON instructions for 64-bit numbers
                            return unsafe { NEON::<FloatIgnoreNaN>::argminmax(self) }
                        }
                    }
                    SCALAR::<FloatIgnoreNaN>::argminmax(self)
                }
            }
            impl NaNArgMinMax for &[$float_type] {
                fn nanargminmax(&self) -> (usize, usize) {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    {
                        if is_x86_feature_detected!("sse4.1") & (<$float_type>::NB_BITS == 8) {
                            // 8-bit numbers are best handled by SSE4.1
                            return unsafe { SSE::<FloatReturnNaN>::argminmax(self) }
                        } else if is_x86_feature_detected!("avx512bw") & (<$float_type>::NB_BITS <= 16) {
                            // BW (ByteWord) instructions are needed for 8 or 16-bit avx512
                            return unsafe { AVX512::<FloatReturnNaN>::argminmax(self) }
                        } else if is_x86_feature_detected!("avx512f") {  // TODO: check if avx512bw is included in avx512f
                            return unsafe { AVX512::<FloatReturnNaN>::argminmax(self) }
                        } else if is_x86_feature_detected!("avx2") {
                            return unsafe { AVX2::<FloatReturnNaN>::argminmax(self) }
                        // SKIP SSE4.2 bc scalar is faster or equivalent for 64 bit numbers
                        } else if is_x86_feature_detected!("sse4.1") & (<$float_type>::NB_BITS < 64) {
                            // Scalar is faster for 64-bit numbers
                            // TODO: double check this (observed different things for new float implementation)
                            return unsafe { SSE::<FloatReturnNaN>::argminmax(self) }
                        }
                    }
                    #[cfg(target_arch = "aarch64")]
                    {
                        if std::arch::is_aarch64_feature_detected!("neon") & (<$float_type>::NB_BITS < 64) {
                            // We miss some NEON instructions for 64-bit numbers
                            return unsafe { NEON::<FloatReturnNaN>::argminmax(self) }
                        }
                    }
                    #[cfg(target_arch = "arm")]
                    {
                        if std::arch::is_arm_feature_detected!("neon") & (<$float_type>::NB_BITS < 64) {
                            // TODO: requires v7?
                            // We miss some NEON instructions for 64-bit numbers
                            return unsafe { NEON::<FloatReturnNaN>::argminmax(self) }
                        }
                    }
                    SCALAR::<FloatReturnNaN>::argminmax(self)
                }
            }
        )*
    };
}

// Implement ArgMinMax for the rust primitive types
impl_argminmax_int!(i8, i16, i32, i64, u8, u16, u32, u64);
#[cfg(feature = "float")]
impl_argminmax_float!(f32, f64);
// Implement ArgMinMax for other data types
#[cfg(feature = "half")]
impl_argminmax_float!(f16);

// ------------------------------ [T] ------------------------------

// impl<T> ArgMinMax for [T]
// where
//     for<'a> &'a [T]: ArgMinMax,
// {
//     fn argminmax(&self) -> (usize, usize) {
//         // TODO: use the slice implementation without having stack-overflow
//     }
// }

// ------------------------------ Vec ------------------------------

impl<T> ArgMinMax for Vec<T>
where
    for<'a> &'a [T]: ArgMinMax,
{
    fn argminmax(&self) -> (usize, usize) {
        self.as_slice().argminmax()
    }
}

#[cfg(any(feature = "float", feature = "half"))]
impl<T> NaNArgMinMax for Vec<T>
where
    for<'a> &'a [T]: NaNArgMinMax,
{
    fn nanargminmax(&self) -> (usize, usize) {
        self.as_slice().nanargminmax()
    }
}

// ----------------------- (optional) ndarray ----------------------

#[cfg(feature = "ndarray")]
mod ndarray_impl {
    use super::*;
    use ndarray::{ArrayBase, Data, Ix1};

    // Use the slice implementation
    // -> implement for S where slice implementation available for S::Elem
    // ArrayBase instead of Array1 or ArrayView1 -> https://github.com/rust-ndarray/ndarray/issues/1059
    impl<S> ArgMinMax for ArrayBase<S, Ix1>
    where
        S: Data,
        for<'a> &'a [S::Elem]: ArgMinMax,
    {
        fn argminmax(&self) -> (usize, usize) {
            self.as_slice().unwrap().argminmax()
        }
    }

    #[cfg(any(feature = "float", feature = "half"))]
    impl<S> NaNArgMinMax for ArrayBase<S, Ix1>
    where
        S: Data,
        for<'a> &'a [S::Elem]: NaNArgMinMax,
    {
        fn nanargminmax(&self) -> (usize, usize) {
            self.as_slice().unwrap().nanargminmax()
        }
    }
}

// ----------------------- (optional) arrow ----------------------

#[cfg(feature = "arrow")]
mod arrow_impl {
    use super::*;
    use arrow::array::PrimitiveArray;

    // Use the slice implementation
    // -> implement for T where slice implementation available for T::Native
    impl<T> ArgMinMax for PrimitiveArray<T>
    where
        T: arrow::datatypes::ArrowNumericType,
        for<'a> &'a [T::Native]: ArgMinMax,
    {
        fn argminmax(&self) -> (usize, usize) {
            self.values().argminmax()
        }
    }

    #[cfg(any(feature = "float", feature = "half"))]
    impl<T> NaNArgMinMax for PrimitiveArray<T>
    where
        T: arrow::datatypes::ArrowNumericType,
        for<'a> &'a [T::Native]: NaNArgMinMax,
    {
        fn nanargminmax(&self) -> (usize, usize) {
            self.values().nanargminmax()
        }
    }
}

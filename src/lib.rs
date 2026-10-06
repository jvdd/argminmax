//! A crate for finding the index of the minimum and maximum values in an array.
//!
//! These operations are optimized for speed using [SIMD](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data) instructions (when available).  
//! The SIMD implementation is branchless, ensuring that there is no best case / worst case.
//! Furthermore, runtime CPU feature detection is used to choose the fastest implementation for the current CPU (with a scalar fallback).
//!
//! The SIMD implementation is enabled for the following architectures:
//! - `x86` / `x86_64`: [`SSE`](https://en.wikipedia.org/wiki/Streaming_SIMD_Extensions), [`AVX2`](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#Advanced_Vector_Extensions_2), [`AVX512`](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#AVX-512)
//! - `arm` / `aarch64`: [`NEON`](https://en.wikipedia.org/wiki/ARM_architecture#Advanced_SIMD_(Neon))
//!
//! `i128` and `u128` always use the scalar implementation.
//!
//! # Description
//!
//! This crate provides two traits: [`ArgMinMax`](trait.ArgMinMax.html) and [`NaNArgMinMax`](trait.NaNArgMinMax.html).
//!
//! These traits are implemented for [`slice`](https://doc.rust-lang.org/std/primitive.slice.html) and [`Vec`](https://doc.rust-lang.org/std/vec/struct.Vec.html).  
//! - For [`ArgMinMax`](trait.ArgMinMax.html) the supported data types are
//!   - ints: `i8`, `i16`, `i32`, `i64`, `i128`
//!   - uints: `u8`, `u16`, `u32`, `u64`, `u128`
//!   - floats: `f16`, `f32`, `f64` (see [Features](#features))
//! - For [`NaNArgMinMax`](trait.NaNArgMinMax.html) the supported data types are
//!   - floats: `f16`, `f32`, `f64` (see [Features](#features))
//!
//! Both traits differ in how they handle NaNs:
//! - [`ArgMinMax`](trait.ArgMinMax.html) ignores NaNs and returns the index of the minimum and maximum values in an array.
//! - [`NaNArgMinMax`](trait.NaNArgMinMax.html) returns the index of the first NaN in an array if there is one, otherwise it returns the index of the minimum and maximum values in an array.
//!
//! Both traits have a counterpart that skips the null elements, which an [Arrow validity bitmap](https://arrow.apache.org/docs/format/Columnar.html#validity-bitmaps) marks:
//! [`ArgMinMaxMasked`](trait.ArgMinMaxMasked.html) and [`NaNArgMinMaxMasked`](trait.NaNArgMinMaxMasked.html).
//!
//! ### Caution
//! When dealing with floats and you are sure that there are no NaNs in the array, you should use [`ArgMinMax`](trait.ArgMinMax.html) instead of [`NaNArgMinMax`](trait.NaNArgMinMax.html) for performance reasons. The former is 5%-30% faster than the latter.
//!
//!
//! # Features
//! This crate has several features.
//!
//! - **`float`** *(default)* - enables the traits for floats (`f32` and `f64`).
//! - **`nightly_simd`** - enables NEON SIMD instructions on 32-bit ARM (requires a nightly compiler; no effect on other architectures).
//! - **`half`** - enables the traits for `f16` (requires the [`half`](https://crates.io/crates/half) crate).
//! - **`ndarray`** - adds the traits to [`ndarray::ArrayBase`](https://docs.rs/ndarray/latest/ndarray/struct.ArrayBase.html) (requires the `ndarray` crate).
//! - **`arrow`** - adds the traits to [`arrow::array::PrimitiveArray`](https://docs.rs/arrow/latest/arrow/array/struct.PrimitiveArray.html) (requires the `arrow` crate).
//!
//!
//! # Examples
//!
//! Three examples are provided below.
//!
//! ## Example with integers
//! ```
//! use argminmax::ArgMinMax;
//!
//! let a: Vec<i32> = vec![0, 1, 2, 3, 4, 5];
//! let (imin, imax) = a.argminmax();
//! assert_eq!(imin, 0);
//! assert_eq!(imax, 5);
//! ```
//!
//! ## Example with NaNs (default `float` feature)
//! ```ignore
//! use argminmax::ArgMinMax; // argminmax ignores NaNs
//! use argminmax::NaNArgMinMax; // nanargminmax returns index of first NaN
//!
//! let a: Vec<f32> = vec![f32::NAN, 1.0, f32::NAN, 3.0, 4.0, 5.0];
//! let (imin, imax) = a.argminmax(); // ArgMinMax::argminmax
//! assert_eq!(imin, 1);
//! assert_eq!(imax, 5);
//! let (imin, imax) = a.nanargminmax(); // NaNArgMinMax::nanargminmax
//! assert_eq!(imin, 0);
//! assert_eq!(imax, 0);
//!```
//!
//! ## Example with nulls
//! ```
//! use argminmax::ArgMinMaxMasked;
//!
//! let a: Vec<i32> = vec![-5, 1, 2, 3, 4, 9];
//! // The validity bitmap: bit 1 + i is set iff element i is valid (thus, elements 0 and 5 are null)
//! let validity: Vec<u8> = vec![0b0011_1100];
//! let (imin, imax) = a.argminmax_masked(&validity, 1).unwrap();
//! assert_eq!(imin, 1);
//! assert_eq!(imax, 4);
//! // There are no valid elements
//! assert_eq!(a.argminmax_masked(&[0], 0), None);
//! ```
//!

// NEON on 32-bit ARM is still unstable (AVX512 & aarch64 NEON are stable)
#![cfg_attr(
    all(feature = "nightly_simd", target_arch = "arm"),
    feature(
        stdarch_arm_neon_intrinsics,
        stdarch_arm_feature_detection,
        arm_target_feature
    )
)]

// #[macro_use]
// extern crate lazy_static;

pub mod dtype_strategy;
pub mod scalar;
pub mod simd;
mod validity;

pub(crate) use dtype_strategy::Int;
#[cfg(any(feature = "float", feature = "half"))]
pub(crate) use dtype_strategy::{FloatIgnoreNaN, FloatReturnNaN};
pub(crate) use scalar::{ScalarArgMinMax, SCALAR};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) use simd::{SIMDArgMinMax, AVX2, AVX512, SSE};
#[cfg(any(
    all(target_arch = "arm", feature = "nightly_simd"),
    target_arch = "aarch64"
))]
pub(crate) use simd::{SIMDArgMinMax, NEON};

#[cfg(feature = "half")]
use half::f16;

/// Trait for finding the minimum and maximum values in an array. For floats, NaNs are ignored.  
///
/// This trait is implemented for slices (or other array-like) of integers and floats.
///  
/// See the [feature documentation](index.html#features) for more information on the supported data types and array types.
///
pub trait ArgMinMax {
    // TODO: future work implement these other functions?
    // fn min(self) -> Self::Item;
    // fn max(self) -> Self::Item;
    // fn minmax(self) -> (T, T);

    /// Get the index of the minimum and maximum values in the array.
    ///
    /// When dealing with floats, NaNs are ignored.  
    /// Note that this differs from numpy, where the `argmin` and `argmax` functions
    /// return the index of the first NaN (which is the behavior of our nanargminmax
    /// function).
    ///
    /// # Returns
    /// A tuple of the index of the minimum and maximum values in the array
    /// `(min_index, max_index)`. For a float array with only NaNs, this is `(0, 0)`.
    ///
    fn argminmax(&self) -> (usize, usize);

    /// Get the index of the minimum value in the array.
    ///
    /// When dealing with floats, NaNs are ignored.
    /// Note that this differs from numpy, where the `argmin` function returns the index
    /// of the first NaN (which is the behavior of our nanargmin function).
    ///
    /// # Returns
    /// The index of the minimum value in the array (0 for a float array with only NaNs).
    ///
    fn argmin(&self) -> usize;

    /// Get the index of the maximum value in the array.
    ///
    /// When dealing with floats, NaNs are ignored.
    /// Note that this differs from numpy, where the `argmax` function returns the index
    /// of the first NaN (which is the behavior of our nanargmax function).
    ///
    /// # Returns
    /// The index of the maximum value in the array (0 for a float array with only NaNs).
    ///
    fn argmax(&self) -> usize;
}

/// Trait for finding the minimum and maximum values in an array. For floats, NaNs are propagated - index of the first NaN is returned.  
///
/// This trait is implemented for slices (or other array-like) of floats.
///  
/// See the [feature documentation](index.html#features) for more information on the supported data types and array types.
///
#[cfg(any(feature = "float", feature = "half"))]
pub trait NaNArgMinMax {
    /// Get the index of the minimum and maximum values in the array.
    ///
    /// When dealing with floats, NaNs are propagated - index of the first NaN is
    /// returned.  
    /// Note that this differs from numpy, where the `nanargmin` and `nanargmax`
    /// functions ignore NaNs (which is the behavior of our argminmax function).
    ///
    /// # Returns
    /// A tuple of the index of the minimum and maximum values in the array
    /// `(min_index, max_index)`.
    ///
    /// # Caution
    /// When multiple bit-representations for NaNs are used, no guarantee is made
    /// that the first NaN is returned.
    ///
    fn nanargminmax(&self) -> (usize, usize);

    /// Get the index of the minimum value in the array.
    ///
    /// When dealing with floats, NaNs are propagated - index of the first NaN is
    /// returned.
    /// Note that this differs from numpy, where the `nanargmin` function ignores
    /// NaNs (which is the behavior of our argmin function).
    ///
    /// # Returns
    /// The index of the minimum value in the array.
    ///
    /// # Caution
    /// When multiple bit-representations for NaNs are used, no guarantee is made
    /// that the first NaN is returned.
    ///
    fn nanargmin(&self) -> usize;

    /// Get the index of the maximum value in the array.
    ///
    /// When dealing with floats, NaNs are propagated - index of the first NaN is
    /// returned.
    /// Note that this differs from numpy, where the `nanargmax` function ignores
    /// NaNs (which is the behavior of our argmax function).
    ///
    /// # Returns
    /// The index of the maximum value in the array.
    ///
    /// # Caution
    /// When multiple bit-representations for NaNs are used, no guarantee is made
    /// that the first NaN is returned.
    ///
    fn nanargmax(&self) -> usize;
}

/// Trait for finding the minimum and maximum values in an array, skipping the null
/// elements. For floats, NaNs are ignored.
///
/// This trait is implemented for slices (and `Vec`) of the same data types as
/// [`ArgMinMax`](trait.ArgMinMax.html).
///
/// The null elements are given by a validity bitmap in the [Arrow format](https://arrow.apache.org/docs/format/Columnar.html#validity-bitmaps):
/// element `i` is valid (not null) iff bit `offset + i` of `validity` is set, where the
/// bits are numbered from the least significant bit of the first byte. E.g.,
/// - arrow: `NullBuffer::validity()` and `NullBuffer::offset()`
/// - polars-arrow: the bytes and the offset of `Bitmap::as_slice()`
///
/// The result is that of [`ArgMinMax`](trait.ArgMinMax.html) on only the valid elements
/// (thus, e.g., the first index is returned on ties), or `None` when there are no valid
/// elements. For floats, NaNs are ignored and infinities are compared as other values,
/// also when the valid values are only NaNs and/or infinities (unlike `ArgMinMax`): when
/// all valid values are NaN, the index of the first valid value is returned.
///
/// # Panics
/// When `validity` has less than `offset + len` bits.
///
pub trait ArgMinMaxMasked {
    /// Get the index of the minimum and maximum valid values in the array.
    ///
    /// When dealing with floats, NaNs are ignored. When all valid values are NaN, the
    /// index of the first valid value is returned.
    ///
    /// # Returns
    /// A tuple of the index of the minimum and maximum valid values in the array
    /// `(min_index, max_index)`, or `None` when there are no valid values.
    ///
    fn argminmax_masked(&self, validity: &[u8], offset: usize) -> Option<(usize, usize)>;

    /// Get the index of the minimum valid value in the array.
    ///
    /// When dealing with floats, NaNs are ignored. When all valid values are NaN, the
    /// index of the first valid value is returned.
    ///
    /// # Returns
    /// The index of the minimum valid value in the array, or `None` when there are no
    /// valid values.
    ///
    fn argmin_masked(&self, validity: &[u8], offset: usize) -> Option<usize>;

    /// Get the index of the maximum valid value in the array.
    ///
    /// When dealing with floats, NaNs are ignored. When all valid values are NaN, the
    /// index of the first valid value is returned.
    ///
    /// # Returns
    /// The index of the maximum valid value in the array, or `None` when there are no
    /// valid values.
    ///
    fn argmax_masked(&self, validity: &[u8], offset: usize) -> Option<usize>;
}

/// Trait for finding the minimum and maximum values in an array, skipping the null
/// elements. For floats, NaNs are propagated - index of the first (valid) NaN is
/// returned.
///
/// This trait is implemented for slices (and `Vec`) of floats.
///
/// See [`ArgMinMaxMasked`](trait.ArgMinMaxMasked.html) for the validity bitmap. The
/// result is the same as [`NaNArgMinMax`](trait.NaNArgMinMax.html) on only the valid
/// elements, or `None` when there are no valid elements.
///
/// # Panics
/// When `validity` has less than `offset + len` bits.
///
#[cfg(any(feature = "float", feature = "half"))]
pub trait NaNArgMinMaxMasked {
    /// Get the index of the minimum and maximum valid values in the array.
    ///
    /// NaNs are propagated - index of the first valid NaN is returned.
    ///
    /// # Returns
    /// A tuple of the index of the minimum and maximum valid values in the array
    /// `(min_index, max_index)`, or `None` when there are no valid values.
    ///
    /// # Caution
    /// When multiple bit-representations for NaNs are used, no guarantee is made
    /// that the first NaN is returned.
    ///
    fn nanargminmax_masked(&self, validity: &[u8], offset: usize) -> Option<(usize, usize)>;

    /// Get the index of the minimum valid value in the array.
    ///
    /// NaNs are propagated - index of the first valid NaN is returned.
    ///
    /// # Returns
    /// The index of the minimum valid value in the array, or `None` when there are no
    /// valid values.
    ///
    /// # Caution
    /// When multiple bit-representations for NaNs are used, no guarantee is made
    /// that the first NaN is returned.
    ///
    fn nanargmin_masked(&self, validity: &[u8], offset: usize) -> Option<usize>;

    /// Get the index of the maximum valid value in the array.
    ///
    /// NaNs are propagated - index of the first valid NaN is returned.
    ///
    /// # Returns
    /// The index of the maximum valid value in the array, or `None` when there are no
    /// valid values.
    ///
    /// # Caution
    /// When multiple bit-representations for NaNs are used, no guarantee is made
    /// that the first NaN is returned.
    ///
    fn nanargmax_masked(&self, validity: &[u8], offset: usize) -> Option<usize>;
}

// ---- Helper macros ----

// Only used for the SIMD dispatch below
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "arm", feature = "nightly_simd"),
    target_arch = "aarch64",
))]
trait DTypeInfo {
    const NB_BITS: usize;
}

/// Macro for implementing DTypeInfo for the passed data types (uints, ints, floats)
macro_rules! impl_nb_bits {
    // $data_type is the data type (e.g. i32)
    // you can pass multiple types (separated by commas) to this macro
    ($($data_type:ty)*) => ($(
        #[cfg(any(
            target_arch = "x86",
            target_arch = "x86_64",
            all(target_arch = "arm", feature = "nightly_simd"),
            target_arch = "aarch64",
        ))]
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

/// Returns whether the CPU supports the AVX512 implementation for `T`:
/// 8 and 16-bit data types need AVX512BW, 32 and 64-bit data types need AVX512F.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
fn avx512_supported<T: DTypeInfo>() -> bool {
    if T::NB_BITS <= 16 {
        is_x86_feature_detected!("avx512bw")
    } else {
        is_x86_feature_detected!("avx512f")
    }
}

// ------------------------------ &[T] ------------------------------

/// Macro that calls `$method($args)` of the fastest implementation (SIMD instruction set
/// or scalar) that the CPU supports, for the `$dtype` data type and the given
/// DTypeStrategy (`Int`, `FloatIgnoreNaN` or `FloatReturnNaN`).
///
/// Use it only as the tail expression of a function: when a SIMD implementation is
/// selected, the macro returns its result from the enclosing function.
macro_rules! dispatch {
    ($dtype:ty, Int, $method:ident($($arg:expr),*)) => {{
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("sse4.1") & (<$dtype>::NB_BITS == 8) {
                // 8-bit numbers are best handled by SSE4.1
                return unsafe { SSE::<Int>::$method($($arg),*) };
            }
            if avx512_supported::<$dtype>() {
                return unsafe { AVX512::<Int>::$method($($arg),*) };
            }
            if is_x86_feature_detected!("avx2") {
                return unsafe { AVX2::<Int>::$method($($arg),*) };
            // SKIP SSE4.2 bc scalar is faster or equivalent for 64 bit numbers
            } else if is_x86_feature_detected!("sse4.1") & (<$dtype>::NB_BITS < 64) {
                // Scalar is faster for 64-bit numbers
                return unsafe { SSE::<Int>::$method($($arg),*) };
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                return unsafe { NEON::<Int>::$method($($arg),*) };
            }
        }
        #[cfg(all(target_arch = "arm", feature = "nightly_simd"))]
        {
            if std::arch::is_arm_feature_detected!("neon") & (<$dtype>::NB_BITS < 64) {
                // TODO: requires v7?
                // We miss some NEON instructions for 64-bit numbers
                return unsafe { NEON::<Int>::$method($($arg),*) };
            }
        }
        SCALAR::<Int>::$method($($arg),*)
    }};
    ($dtype:ty, FloatIgnoreNaN, $method:ident($($arg:expr),*)) => {{
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if avx512_supported::<$dtype>() {
                return unsafe { AVX512::<FloatIgnoreNaN>::$method($($arg),*) };
            }
            if is_x86_feature_detected!("avx2") {
                // f16 requires avx2
                return unsafe { AVX2::<FloatIgnoreNaN>::$method($($arg),*) };
            } else if is_x86_feature_detected!("avx") & (<$dtype>::NB_BITS > 16) {
                // f32 and f64 do not require avx2
                return unsafe { AVX2::<FloatIgnoreNaN>::$method($($arg),*) };
            } else if is_x86_feature_detected!("sse4.1") & (<$dtype>::NB_BITS < 64) {
                // Scalar is faster for 64-bit numbers
                return unsafe { SSE::<FloatIgnoreNaN>::$method($($arg),*) };
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                return unsafe { NEON::<FloatIgnoreNaN>::$method($($arg),*) };
            }
        }
        #[cfg(all(target_arch = "arm", feature = "nightly_simd"))]
        {
            if std::arch::is_arm_feature_detected!("neon") & (<$dtype>::NB_BITS < 64) {
                // We miss some NEON instructions for 64-bit numbers
                return unsafe { NEON::<FloatIgnoreNaN>::$method($($arg),*) };
            }
        }
        SCALAR::<FloatIgnoreNaN>::$method($($arg),*)
    }};
    ($dtype:ty, FloatReturnNaN, $method:ident($($arg:expr),*)) => {{
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if avx512_supported::<$dtype>() {
                return unsafe { AVX512::<FloatReturnNaN>::$method($($arg),*) };
            }
            if is_x86_feature_detected!("avx2") {
                return unsafe { AVX2::<FloatReturnNaN>::$method($($arg),*) };
            // SKIP SSE4.2 bc scalar is faster or equivalent for 64 bit numbers
            } else if is_x86_feature_detected!("sse4.1") & (<$dtype>::NB_BITS < 64) {
                // Scalar is faster for 64-bit numbers
                // TODO: double check this (observed different things for new float implementation)
                return unsafe { SSE::<FloatReturnNaN>::$method($($arg),*) };
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                return unsafe { NEON::<FloatReturnNaN>::$method($($arg),*) };
            }
        }
        #[cfg(all(target_arch = "arm", feature = "nightly_simd"))]
        {
            if std::arch::is_arm_feature_detected!("neon") & (<$dtype>::NB_BITS < 64) {
                // We miss some NEON instructions for 64-bit numbers
                return unsafe { NEON::<FloatReturnNaN>::$method($($arg),*) };
            }
        }
        SCALAR::<FloatReturnNaN>::$method($($arg),*)
    }};
}

/// Macro for implementing ArgMinMax for signed and unsigned integers
macro_rules! impl_argminmax_int {
    // $int_type is the integer data type of the array (e.g. i32)
    // you can pass multiple types (separated by commas) to this macro
    ($($int_type:ty),*) => {
        $(
            impl ArgMinMax for &[$int_type] {
                fn argminmax(&self) -> (usize, usize) {
                    #[cfg(target_arch = "aarch64")]
                    {
                        if <$int_type>::NB_BITS == 64 {
                            // Scalar is faster for 64-bit numbers
                            return SCALAR::<Int>::argminmax(self);
                        }
                    }
                    dispatch!($int_type, Int, argminmax(self))
                }

                fn argmin(&self) -> usize {
                    dispatch!($int_type, Int, argmin(self))
                }

                fn argmax(&self) -> usize {
                    dispatch!($int_type, Int, argmax(self))
                }
            }
        )*
    };
}

/// Macro for implementing ArgMinMax and NaNArgMinMax for floats
#[cfg(any(feature = "float", feature = "half"))]
macro_rules! impl_argminmax_float {
    // $float_type is the float data type of the array (e.g. f32)
    // you can pass multiple types (separated by commas) to this macro
    ($($float_type:ty),*) => {
        $(
            impl ArgMinMax for &[$float_type] {
                fn argminmax(&self) -> (usize, usize) {
                    dispatch!($float_type, FloatIgnoreNaN, argminmax(self))
                }

                fn argmin(&self) -> usize {
                    dispatch!($float_type, FloatIgnoreNaN, argmin(self))
                }

                fn argmax(&self) -> usize {
                    dispatch!($float_type, FloatIgnoreNaN, argmax(self))
                }
            }

            impl NaNArgMinMax for &[$float_type] {
                fn nanargminmax(&self) -> (usize, usize) {
                    dispatch!($float_type, FloatReturnNaN, argminmax(self))
                }

                fn nanargmin(&self) -> usize {
                    dispatch!($float_type, FloatReturnNaN, argmin(self))
                }

                fn nanargmax(&self) -> usize {
                    dispatch!($float_type, FloatReturnNaN, argmax(self))
                }
            }
        )*
    };
}

/// Macro for implementing ArgMinMax and ArgMinMaxMasked for integers that only have a
/// scalar implementation
macro_rules! impl_argminmax_int_scalar {
    // $int_type is the integer data type of the array (e.g. i128)
    // you can pass multiple types (separated by commas) to this macro
    ($($int_type:ty),*) => {
        $(
            impl ArgMinMax for &[$int_type] {
                fn argminmax(&self) -> (usize, usize) {
                    SCALAR::<Int>::argminmax(self)
                }

                fn argmin(&self) -> usize {
                    SCALAR::<Int>::argmin(self)
                }

                fn argmax(&self) -> usize {
                    SCALAR::<Int>::argmax(self)
                }
            }

            impl ArgMinMaxMasked for &[$int_type] {
                fn argminmax_masked(&self, validity: &[u8], offset: usize) -> Option<(usize, usize)> {
                    SCALAR::<Int>::argminmax_masked(self, validity, offset)
                }

                fn argmin_masked(&self, validity: &[u8], offset: usize) -> Option<usize> {
                    SCALAR::<Int>::argmin_masked(self, validity, offset)
                }

                fn argmax_masked(&self, validity: &[u8], offset: usize) -> Option<usize> {
                    SCALAR::<Int>::argmax_masked(self, validity, offset)
                }
            }
        )*
    };
}

// Implement ArgMinMax for (non-optional) integer rust primitive types
impl_argminmax_int!(i8, i16, i32, i64, u8, u16, u32, u64);
// 128-bit integers are scalar-only: a SIMD prototype only paid off for in-cache data
// on AVX512 (see #75)
impl_argminmax_int_scalar!(i128, u128);
// Implement for (optional) float rust primitive types
#[cfg(feature = "float")]
impl_argminmax_float!(f32, f64);

// Implement ArgMinMax for other data types
#[cfg(feature = "half")]
impl_argminmax_float!(f16);

/// Macro for implementing ArgMinMaxMasked for the passed data types with the given
/// DTypeStrategy (Int or FloatIgnoreNaN)
macro_rules! impl_argminmax_masked {
    ($dtype_strategy:ident, $($dtype:ty),*) => {
        $(
            impl ArgMinMaxMasked for &[$dtype] {
                fn argminmax_masked(&self, validity: &[u8], offset: usize) -> Option<(usize, usize)> {
                    dispatch!($dtype, $dtype_strategy, argminmax_masked(self, validity, offset))
                }

                fn argmin_masked(&self, validity: &[u8], offset: usize) -> Option<usize> {
                    dispatch!($dtype, $dtype_strategy, argmin_masked(self, validity, offset))
                }

                fn argmax_masked(&self, validity: &[u8], offset: usize) -> Option<usize> {
                    dispatch!($dtype, $dtype_strategy, argmax_masked(self, validity, offset))
                }
            }
        )*
    };
}

/// Macro for implementing NaNArgMinMaxMasked for the passed float data types
#[cfg(any(feature = "float", feature = "half"))]
macro_rules! impl_nanargminmax_masked {
    ($($float_type:ty),*) => {
        $(
            impl NaNArgMinMaxMasked for &[$float_type] {
                fn nanargminmax_masked(&self, validity: &[u8], offset: usize) -> Option<(usize, usize)> {
                    dispatch!($float_type, FloatReturnNaN, argminmax_masked(self, validity, offset))
                }

                fn nanargmin_masked(&self, validity: &[u8], offset: usize) -> Option<usize> {
                    dispatch!($float_type, FloatReturnNaN, argmin_masked(self, validity, offset))
                }

                fn nanargmax_masked(&self, validity: &[u8], offset: usize) -> Option<usize> {
                    dispatch!($float_type, FloatReturnNaN, argmax_masked(self, validity, offset))
                }
            }
        )*
    };
}

// Implement ArgMinMaxMasked (and NaNArgMinMaxMasked) for the same data types
impl_argminmax_masked!(Int, i8, i16, i32, i64, u8, u16, u32, u64);
#[cfg(feature = "float")]
impl_argminmax_masked!(FloatIgnoreNaN, f32, f64);
#[cfg(feature = "float")]
impl_nanargminmax_masked!(f32, f64);
#[cfg(feature = "half")]
impl_argminmax_masked!(FloatIgnoreNaN, f16);
#[cfg(feature = "half")]
impl_nanargminmax_masked!(f16);

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

    fn argmin(&self) -> usize {
        self.as_slice().argmin()
    }

    fn argmax(&self) -> usize {
        self.as_slice().argmax()
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

    fn nanargmin(&self) -> usize {
        self.as_slice().nanargmin()
    }

    fn nanargmax(&self) -> usize {
        self.as_slice().nanargmax()
    }
}

impl<T> ArgMinMaxMasked for Vec<T>
where
    for<'a> &'a [T]: ArgMinMaxMasked,
{
    fn argminmax_masked(&self, validity: &[u8], offset: usize) -> Option<(usize, usize)> {
        self.as_slice().argminmax_masked(validity, offset)
    }

    fn argmin_masked(&self, validity: &[u8], offset: usize) -> Option<usize> {
        self.as_slice().argmin_masked(validity, offset)
    }

    fn argmax_masked(&self, validity: &[u8], offset: usize) -> Option<usize> {
        self.as_slice().argmax_masked(validity, offset)
    }
}

#[cfg(any(feature = "float", feature = "half"))]
impl<T> NaNArgMinMaxMasked for Vec<T>
where
    for<'a> &'a [T]: NaNArgMinMaxMasked,
{
    fn nanargminmax_masked(&self, validity: &[u8], offset: usize) -> Option<(usize, usize)> {
        self.as_slice().nanargminmax_masked(validity, offset)
    }

    fn nanargmin_masked(&self, validity: &[u8], offset: usize) -> Option<usize> {
        self.as_slice().nanargmin_masked(validity, offset)
    }

    fn nanargmax_masked(&self, validity: &[u8], offset: usize) -> Option<usize> {
        self.as_slice().nanargmax_masked(validity, offset)
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

        fn argmin(&self) -> usize {
            self.as_slice().unwrap().argmin()
        }

        fn argmax(&self) -> usize {
            self.as_slice().unwrap().argmax()
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

        fn nanargmin(&self) -> usize {
            self.as_slice().unwrap().nanargmin()
        }

        fn nanargmax(&self) -> usize {
            self.as_slice().unwrap().nanargmax()
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
            self.values().as_ref().argminmax()
        }

        fn argmin(&self) -> usize {
            self.values().as_ref().argmin()
        }

        fn argmax(&self) -> usize {
            self.values().as_ref().argmax()
        }
    }

    #[cfg(any(feature = "float", feature = "half"))]
    impl<T> NaNArgMinMax for PrimitiveArray<T>
    where
        T: arrow::datatypes::ArrowNumericType,
        for<'a> &'a [T::Native]: NaNArgMinMax,
    {
        fn nanargminmax(&self) -> (usize, usize) {
            self.values().as_ref().nanargminmax()
        }

        fn nanargmin(&self) -> usize {
            self.values().as_ref().nanargmin()
        }

        fn nanargmax(&self) -> usize {
            self.values().as_ref().nanargmax()
        }
    }
}

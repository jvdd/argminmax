/// This module contains structs and traits that are used to configure the SIMD
/// implementation. The structs are used to store the register size and the trait is
/// used to get the lane size for a given datatype.
///
/// More info on SIMD:
/// https://github.com/rust-lang/portable-simd/blob/master/beginners-guide.md#target-features
///

/// SIMD instruction set trait - used to store the register size and get the lane size
/// for a given datatype
pub trait SIMDInstructionSet {
    /// The size of the register in bits
    const REGISTER_SIZE: usize;

    // Set the const lanesize for each datatype
    const LANE_SIZE_8: usize = Self::REGISTER_SIZE / (std::mem::size_of::<u8>() * 8);
    const LANE_SIZE_16: usize = Self::REGISTER_SIZE / (std::mem::size_of::<u16>() * 8);
    const LANE_SIZE_32: usize = Self::REGISTER_SIZE / (std::mem::size_of::<u32>() * 8);
    const LANE_SIZE_64: usize = Self::REGISTER_SIZE / (std::mem::size_of::<u64>() * 8);

    fn get_lane_size<DType>() -> usize {
        Self::REGISTER_SIZE / (std::mem::size_of::<DType>() * 8)
    }
}

// ----------------------------------- x86_64 / x86 ------------------------------------

/// SSE instruction set - this will be implemented for all:
/// - ints (see simd_i*.rs files) - Int DTypeStrategy
/// - uints (see simd_u*.rs files) - Int DTypeStrategy
/// - floats: returning NaNs (see simd_f*_return_nan.rs files) - FloatReturnNan DTypeStrategy
/// - floats: ignoring NaNs (see simd_f*_ignore_nan.rs files) - FloatIgnoreNaN DTypeStrategy
pub struct SSE<DTypeStrategy> {
    pub(crate) _dtype_strategy: DTypeStrategy,
}

impl<DTypeStrategy> SIMDInstructionSet for SSE<DTypeStrategy> {
    /// SSE register size is 128 bits
    /// https://en.wikipedia.org/wiki/Streaming_SIMD_Extensions#Registers
    const REGISTER_SIZE: usize = 128;
}

/// AVX2 instruction set - this will be implemented for all:
/// - ints (see simd_i*.rs files) - Int DTypeStrategy
/// - uints (see simd_u*.rs files) - Int DTypeStrategy
/// - floats: returning NaNs (see simd_f*_return_nan.rs files) - FloatReturnNan DTypeStrategy
/// - floats: ignoring NaNs (see simd_f*_ignore_nan.rs files) - FloatIgnoreNaN DTypeStrategy
///     ! important remark: AVX is enough for f32 and f64, but we need AVX2 for f16
///     -> f16 is currently not yet implemented (TODO)
pub struct AVX2<DTypeStrategy> {
    pub(crate) _dtype_strategy: DTypeStrategy,
}

impl<DTypeStrategy> SIMDInstructionSet for AVX2<DTypeStrategy> {
    /// AVX(2) register size is 256 bits
    /// AVX:  https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#Advanced_Vector_Extensions
    /// AVX2: https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#AVX2
    const REGISTER_SIZE: usize = 256;
}

/// AVX512 instruction set - this will be implemented for all:
/// - ints (see simd_i*.rs files) - Int DTypeStrategy
/// - uints (see simd_u*.rs files) - Int DTypeStrategy
/// - floats: returning NaNs (see simd_f*_return_nan.rs files) - FloatReturnNan DTypeStrategy
/// - floats: ignoring NaNs (see simd_f*_ignore_nan.rs files) - FloatIgnoreNaN DTypeStrategy
pub struct AVX512<DTypeStrategy> {
    pub(crate) _dtype_strategy: DTypeStrategy,
}

impl<DTypeStrategy> SIMDInstructionSet for AVX512<DTypeStrategy> {
    /// AVX512 register size is 512 bits
    /// https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#AVX-512
    const REGISTER_SIZE: usize = 512;
}

// ----------------------------------- aarch64 / arm -----------------------------------

/// NEON instruction set - this will be implemented for all:
/// - ints (see simd_i*.rs files) - Int DTypeStrategy
/// - uints (see simd_u*.rs files) - Int DTypeStrategy
/// - floats: returning NaNs (see simd_f*_return_nan.rs files) - FloatReturnNan DTypeStrategy
/// - floats: ignoring NaNs (see simd_f*_ignore_nan.rs files) - FloatIgnoreNaN DTypeStrategy
///
/// Note: there are no NEON instructions for 64-bit numbers, so for 64-bit numbers we
/// fall back to the scalar implementation.
pub struct NEON<DTypeStrategy> {
    pub(crate) _dtype_strategy: DTypeStrategy,
}

impl<DTypeStrategy> SIMDInstructionSet for NEON<DTypeStrategy> {
    /// NEON register size is 128 bits
    /// https://en.wikipedia.org/wiki/ARM_architecture#Advanced_SIMD_(Neon)
    const REGISTER_SIZE: usize = 128;
}

// ======================================= TESTS =======================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype_strategy::*;

    use rstest::rstest;
    use rstest_reuse::{self, *};

    #[cfg(feature = "half")]
    use half::f16;

    // The DTypeStrategy should not influence the lane size
    #[template]
    #[rstest]
    #[case::int(Int)]
    #[case::float_return_nan(FloatIgnoreNaN)]
    #[case::float_ignore_nan(FloatReturnNaN)]
    fn dtype_strategies<DTypeStrategy>(#[case] _dtype_strategy: DTypeStrategy) {}

    #[cfg(feature = "half")]
    #[apply(dtype_strategies)]
    fn test_lane_size_f16<DTypeStrategy>(#[case] _dtype_strategy: DTypeStrategy) {
        assert_eq!(SSE::<DTypeStrategy>::get_lane_size::<f16>(), 8);
        assert_eq!(AVX2::<DTypeStrategy>::get_lane_size::<f16>(), 16);
        assert_eq!(AVX512::<DTypeStrategy>::get_lane_size::<f16>(), 32);
        assert_eq!(NEON::<DTypeStrategy>::get_lane_size::<f16>(), 8);
    }

    #[apply(dtype_strategies)]
    fn test_lane_size_f32<DTypeStrategy>(#[case] _dtype_strategy: DTypeStrategy) {
        assert_eq!(SSE::<DTypeStrategy>::get_lane_size::<f32>(), 4);
        assert_eq!(AVX2::<DTypeStrategy>::get_lane_size::<f32>(), 8);
        assert_eq!(AVX512::<DTypeStrategy>::get_lane_size::<f32>(), 16);
        assert_eq!(NEON::<DTypeStrategy>::get_lane_size::<f32>(), 4);
    }

    #[apply(dtype_strategies)]
    fn test_lane_size_f64<DTypeStrategy>(#[case] _dtype_strategy: DTypeStrategy) {
        assert_eq!(SSE::<DTypeStrategy>::get_lane_size::<f64>(), 2);
        assert_eq!(AVX2::<DTypeStrategy>::get_lane_size::<f64>(), 4);
        assert_eq!(AVX512::<DTypeStrategy>::get_lane_size::<f64>(), 8);
        assert_eq!(NEON::<DTypeStrategy>::get_lane_size::<f64>(), 2);
    }

    #[apply(dtype_strategies)]
    fn test_lane_size_i8<DTypeStrategy>(#[case] _dtype_strategy: DTypeStrategy) {
        assert_eq!(SSE::<DTypeStrategy>::get_lane_size::<i8>(), 16);
        assert_eq!(AVX2::<DTypeStrategy>::get_lane_size::<i8>(), 32);
        assert_eq!(AVX512::<DTypeStrategy>::get_lane_size::<i8>(), 64);
        assert_eq!(NEON::<DTypeStrategy>::get_lane_size::<i8>(), 16);
    }

    #[apply(dtype_strategies)]
    fn test_lane_size_i16<DTypeStrategy>(#[case] _dtype_strategy: DTypeStrategy) {
        assert_eq!(SSE::<DTypeStrategy>::get_lane_size::<i16>(), 8);
        assert_eq!(AVX2::<DTypeStrategy>::get_lane_size::<i16>(), 16);
        assert_eq!(AVX512::<DTypeStrategy>::get_lane_size::<i16>(), 32);
        assert_eq!(NEON::<DTypeStrategy>::get_lane_size::<i16>(), 8);
    }

    #[apply(dtype_strategies)]
    fn test_lane_size_i32<DTypeStrategy>(#[case] _dtype_strategy: DTypeStrategy) {
        assert_eq!(SSE::<DTypeStrategy>::get_lane_size::<i32>(), 4);
        assert_eq!(AVX2::<DTypeStrategy>::get_lane_size::<i32>(), 8);
        assert_eq!(AVX512::<DTypeStrategy>::get_lane_size::<i32>(), 16);
        assert_eq!(NEON::<DTypeStrategy>::get_lane_size::<i32>(), 4);
    }

    #[apply(dtype_strategies)]
    fn test_lane_size_i64<DTypeStrategy>(#[case] _dtype_strategy: DTypeStrategy) {
        assert_eq!(SSE::<DTypeStrategy>::get_lane_size::<i64>(), 2);
        assert_eq!(AVX2::<DTypeStrategy>::get_lane_size::<i64>(), 4);
        assert_eq!(AVX512::<DTypeStrategy>::get_lane_size::<i64>(), 8);
        assert_eq!(NEON::<DTypeStrategy>::get_lane_size::<i64>(), 2);
    }

    #[apply(dtype_strategies)]
    fn test_lane_size_u8<DTypeStrategy>(#[case] _dtype_strategy: DTypeStrategy) {
        assert_eq!(SSE::<DTypeStrategy>::get_lane_size::<u8>(), 16);
        assert_eq!(AVX2::<DTypeStrategy>::get_lane_size::<u8>(), 32);
        assert_eq!(AVX512::<DTypeStrategy>::get_lane_size::<u8>(), 64);
        assert_eq!(NEON::<DTypeStrategy>::get_lane_size::<u8>(), 16);
    }

    #[apply(dtype_strategies)]
    fn test_lane_size_u16<DTypeStrategy>(#[case] _dtype_strategy: DTypeStrategy) {
        assert_eq!(SSE::<DTypeStrategy>::get_lane_size::<u16>(), 8);
        assert_eq!(AVX2::<DTypeStrategy>::get_lane_size::<u16>(), 16);
        assert_eq!(AVX512::<DTypeStrategy>::get_lane_size::<u16>(), 32);
        assert_eq!(NEON::<DTypeStrategy>::get_lane_size::<u16>(), 8);
    }

    #[apply(dtype_strategies)]
    fn test_lane_size_u32<DTypeStrategy>(#[case] _dtype_strategy: DTypeStrategy) {
        assert_eq!(SSE::<DTypeStrategy>::get_lane_size::<u32>(), 4);
        assert_eq!(AVX2::<DTypeStrategy>::get_lane_size::<u32>(), 8);
        assert_eq!(AVX512::<DTypeStrategy>::get_lane_size::<u32>(), 16);
        assert_eq!(NEON::<DTypeStrategy>::get_lane_size::<u32>(), 4);
    }

    #[apply(dtype_strategies)]
    fn test_lane_size_u64<DTypeStrategy>(#[case] _dtype_strategy: DTypeStrategy) {
        assert_eq!(SSE::<DTypeStrategy>::get_lane_size::<u64>(), 2);
        assert_eq!(AVX2::<DTypeStrategy>::get_lane_size::<u64>(), 4);
        assert_eq!(AVX512::<DTypeStrategy>::get_lane_size::<u64>(), 8);
        assert_eq!(NEON::<DTypeStrategy>::get_lane_size::<u64>(), 2);
    }
}

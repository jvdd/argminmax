// https://github.com/rust-lang/portable-simd/blob/master/beginners-guide.md#target-features

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
/// - ints (see, the simd_i*.rs files)
/// - uints (see, the simd_u*.rs files)
/// - floats: returning NaNs (see, the simd_f*_return_nan.rs files)
pub struct SSE;
/// SSE instruction set - this will be implemented for all:
/// - floats: ignoring NaNs (see, the `simd_f*_ignore_nan.rs` files)
pub struct SSEIgnoreNaN;

impl SIMDInstructionSet for SSE {
    /// SSE register size is 128 bits
    /// https://en.wikipedia.org/wiki/Streaming_SIMD_Extensions#Registers
    const REGISTER_SIZE: usize = 128;
}

/// AVX2 instruction set - this will be implemented for all:
/// - ints (see, the simd_i*.rs files)
/// - uints (see, the simd_u*.rs files)
/// - floats: returning NaNs (see, the simd_f*_return_nan.rs files)
pub struct AVX2;

/// AVX(2) instruction set - this will be implemented for all:
/// - floats: ignoring NaNs (see, the `simd_f*_ignore_nan.rs` files)
///
/// Important remark: AVX is enough for f32 and f64!
/// -> for f16 we need AVX2 - but this is currently not yet implemented (TODO)
///
/// Note: this struct does not implement the `SIMDInstructionSet` trait
pub struct AVX2IgnoreNaN;

impl SIMDInstructionSet for AVX2 {
    /// AVX(2) register size is 256 bits
    /// AVX:  https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#Advanced_Vector_Extensions
    /// AVX2: https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#AVX2
    const REGISTER_SIZE: usize = 256;
}

/// AVX512 instruction set - this will be implemented for all:
/// - ints (see, the simd_i*.rs files)
/// - uints (see, the simd_u*.rs files)
/// - floats: returning NaNs (see, the simd_f*_return_nan.rs files)
pub struct AVX512;

/// AVX512 instruction set - this will be implemented for all:
/// - floats: ignoring NaNs (see, the `simd_f*_ignore_nan.rs` files)
///
/// Note: this struct does not implement the `SIMDInstructionSet` trait
pub struct AVX512IgnoreNaN;

impl SIMDInstructionSet for AVX512 {
    /// AVX512 register size is 512 bits
    /// https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#AVX-512
    const REGISTER_SIZE: usize = 512;
}

// ----------------------------------- aarch64 / arm -----------------------------------

/// NEON instruction set - this will be implemented for all:
/// - ints (see, the simd_i*.rs files)
/// - uints (see, the simd_u*.rs files)
/// - floats: returning NaNs (see, the simd_f*_return_nan.rs files)
pub struct NEON;

/// NEON instruction set - this will be implemented for all:
/// - floats: ignoring NaNs (see, the `simd_f*_ignore_nan.rs` files)
///
/// Note: this struct does not implement the `SIMDInstructionSet` trait
pub struct NEONIgnoreNaN;

impl SIMDInstructionSet for NEON {
    /// NEON register size is 128 bits
    /// https://en.wikipedia.org/wiki/ARM_architecture#Advanced_SIMD_(Neon)
    const REGISTER_SIZE: usize = 128;
}

// --------------------------------------- Tests ---------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "half")]
    use half::f16;

    #[cfg(feature = "half")]
    #[test]
    fn test_lane_size_f16() {
        assert_eq!(AVX2::get_lane_size::<f16>(), 16);
        assert_eq!(AVX512::get_lane_size::<f16>(), 32);
        assert_eq!(NEON::get_lane_size::<f16>(), 8);
        assert_eq!(SSE::get_lane_size::<f16>(), 8);
    }

    #[test]
    fn test_lane_size_f32() {
        assert_eq!(AVX2::get_lane_size::<f32>(), 8);
        assert_eq!(AVX512::get_lane_size::<f32>(), 16);
        assert_eq!(NEON::get_lane_size::<f32>(), 4);
        assert_eq!(SSE::get_lane_size::<f32>(), 4);
    }

    #[test]
    fn test_lane_size_f64() {
        assert_eq!(AVX2::get_lane_size::<f64>(), 4);
        assert_eq!(AVX512::get_lane_size::<f64>(), 8);
        assert_eq!(NEON::get_lane_size::<f64>(), 2);
        assert_eq!(SSE::get_lane_size::<f64>(), 2);
    }

    #[test]
    fn test_lane_size_i8() {
        assert_eq!(AVX2::get_lane_size::<i8>(), 32);
        assert_eq!(AVX512::get_lane_size::<i8>(), 64);
        assert_eq!(NEON::get_lane_size::<i8>(), 16);
        assert_eq!(SSE::get_lane_size::<i8>(), 16);
    }

    #[test]
    fn test_lane_size_i16() {
        assert_eq!(AVX2::get_lane_size::<i16>(), 16);
        assert_eq!(AVX512::get_lane_size::<i16>(), 32);
        assert_eq!(NEON::get_lane_size::<i16>(), 8);
        assert_eq!(SSE::get_lane_size::<i16>(), 8);
    }

    #[test]
    fn test_lane_size_i32() {
        assert_eq!(AVX2::get_lane_size::<i32>(), 8);
        assert_eq!(AVX512::get_lane_size::<i32>(), 16);
        assert_eq!(NEON::get_lane_size::<i32>(), 4);
        assert_eq!(SSE::get_lane_size::<i32>(), 4);
    }

    #[test]
    fn test_lane_size_i64() {
        assert_eq!(AVX2::get_lane_size::<i64>(), 4);
        assert_eq!(AVX512::get_lane_size::<i64>(), 8);
        assert_eq!(NEON::get_lane_size::<i64>(), 2);
        assert_eq!(SSE::get_lane_size::<i64>(), 2);
    }

    #[test]
    fn test_lane_size_u8() {
        assert_eq!(AVX2::get_lane_size::<u8>(), 32);
        assert_eq!(AVX512::get_lane_size::<u8>(), 64);
        assert_eq!(NEON::get_lane_size::<u8>(), 16);
        assert_eq!(SSE::get_lane_size::<u8>(), 16);
    }

    #[test]
    fn test_lane_size_u16() {
        assert_eq!(AVX2::get_lane_size::<u16>(), 16);
        assert_eq!(AVX512::get_lane_size::<u16>(), 32);
        assert_eq!(NEON::get_lane_size::<u16>(), 8);
        assert_eq!(SSE::get_lane_size::<u16>(), 8);
    }

    #[test]
    fn test_lane_size_u32() {
        assert_eq!(AVX2::get_lane_size::<u32>(), 8);
        assert_eq!(AVX512::get_lane_size::<u32>(), 16);
        assert_eq!(NEON::get_lane_size::<u32>(), 4);
        assert_eq!(SSE::get_lane_size::<u32>(), 4);
    }

    #[test]
    fn test_lane_size_u64() {
        assert_eq!(AVX2::get_lane_size::<u64>(), 4);
        assert_eq!(AVX512::get_lane_size::<u64>(), 8);
        assert_eq!(NEON::get_lane_size::<u64>(), 2);
        assert_eq!(SSE::get_lane_size::<u64>(), 2);
    }
}

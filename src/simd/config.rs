// https://github.com/rust-lang/portable-simd/blob/master/beginners-guide.md#target-features

pub trait SIMDInstructionSet {
    const REGISTER_SIZE: usize;

    // Set the const lanesize for each datatype
    const LANE_SIZE_16: usize = Self::REGISTER_SIZE / (std::mem::size_of::<u16>() * 8);
    const LANE_SIZE_32: usize = Self::REGISTER_SIZE / (std::mem::size_of::<u32>() * 8);
    const LANE_SIZE_64: usize = Self::REGISTER_SIZE / (std::mem::size_of::<u64>() * 8);

    fn get_lane_size<DType>() -> usize {
        Self::REGISTER_SIZE / (std::mem::size_of::<DType>() * 8)
    }
}

// ----------------------------- x86_64 / x86 -----------------------------

pub struct SSE;

impl SIMDInstructionSet for SSE {
    const REGISTER_SIZE: usize = 128;
}

pub struct AVX2; // for f32 and f64 AVX is enough

impl SIMDInstructionSet for AVX2 {
    const REGISTER_SIZE: usize = 256;
}

pub struct AVX512;

impl SIMDInstructionSet for AVX512 {
    const REGISTER_SIZE: usize = 512;
}

// ----------------------------- aarch64 / arm -----------------------------

pub struct NEON;

impl SIMDInstructionSet for NEON {
    const REGISTER_SIZE: usize = 128;
}

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
}

// https://github.com/rust-lang/portable-simd/blob/master/beginners-guide.md#target-features


trait SIMDInstructionSet{
    const REGISTER_SIZE: usize;

    fn get_lane_size<DType>() -> usize {
        Self::REGISTER_SIZE / (std::mem::size_of::<DType>() * 8)
    }
}

struct AVX2;

impl SIMDInstructionSet for AVX2 {
    const REGISTER_SIZE: usize = 256;
}

struct AVX512;

impl SIMDInstructionSet for AVX512 {
    const REGISTER_SIZE: usize = 512;
}

struct NEON;

impl SIMDInstructionSet for NEON {
    const REGISTER_SIZE: usize = 128;
}

struct SSE;

impl SIMDInstructionSet for SSE {
    const REGISTER_SIZE: usize = 128;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lane_size() {
        assert_eq!(AVX2::get_lane_size::<f32>(), 8);
        assert_eq!(AVX512::get_lane_size::<f32>(), 16);
        assert_eq!(NEON::get_lane_size::<f32>(), 4);
        assert_eq!(SSE::get_lane_size::<f32>(), 4);
    }
}
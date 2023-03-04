use super::config::SIMDInstructionSet;
use super::generic::{impl_SIMDInit_Int, SIMDArgMinMax, SIMDInit, SIMDOps};
use crate::SCALAR;
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "arm")]
use std::arch::arm::*;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// The dtype-strategy for performing operations on i32 data: (default) Int
use super::super::dtype_strategy::Int;

const MAX_INDEX: usize = i32::MAX as usize;

// --------------------------------------- AVX2 ----------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx2 {
    use super::super::config::AVX2;
    use super::*;

    const LANE_SIZE: usize = AVX2::<Int>::LANE_SIZE_32;

    impl SIMDOps<i32, __m256i, __m256i, LANE_SIZE> for AVX2<Int> {
        const INITIAL_INDEX: __m256i =
            unsafe { std::mem::transmute([0i32, 1i32, 2i32, 3i32, 4i32, 5i32, 6i32, 7i32]) };
        const INDEX_INCREMENT: __m256i =
            unsafe { std::mem::transmute([LANE_SIZE as i32; LANE_SIZE]) };
        const MAX_INDEX: usize = MAX_INDEX;

        #[inline(always)]
        unsafe fn _reg_to_arr(reg: __m256i) -> [i32; LANE_SIZE] {
            std::mem::transmute::<__m256i, [i32; LANE_SIZE]>(reg)
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const i32) -> __m256i {
            _mm256_loadu_si256(data as *const __m256i)
        }

        #[inline(always)]
        unsafe fn _mm_add(a: __m256i, b: __m256i) -> __m256i {
            _mm256_add_epi32(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: __m256i, b: __m256i) -> __m256i {
            _mm256_cmpgt_epi32(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: __m256i, b: __m256i) -> __m256i {
            _mm256_cmpgt_epi32(b, a)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: __m256i, b: __m256i, mask: __m256i) -> __m256i {
            _mm256_blendv_epi8(a, b, mask)
        }
    }

    impl_SIMDInit_Int!(i32, __m256i, __m256i, LANE_SIZE, AVX2<Int>);

    impl SIMDArgMinMax<i32, __m256i, __m256i, LANE_SIZE, SCALAR<Int>> for AVX2<Int> {
        #[target_feature(enable = "avx2")]
        unsafe fn argminmax(data: &[i32]) -> (usize, usize) {
            Self::_argminmax(data)
        }
    }
}

// ---------------------------------------- SSE ----------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod sse {
    use super::super::config::SSE;
    use super::*;

    const LANE_SIZE: usize = SSE::<Int>::LANE_SIZE_32;

    impl SIMDOps<i32, __m128i, __m128i, LANE_SIZE> for SSE<Int> {
        const INITIAL_INDEX: __m128i = unsafe { std::mem::transmute([0i32, 1i32, 2i32, 3i32]) };
        const INDEX_INCREMENT: __m128i =
            unsafe { std::mem::transmute([LANE_SIZE as i32; LANE_SIZE]) };
        const MAX_INDEX: usize = MAX_INDEX;

        #[inline(always)]
        unsafe fn _reg_to_arr(reg: __m128i) -> [i32; LANE_SIZE] {
            std::mem::transmute::<__m128i, [i32; LANE_SIZE]>(reg)
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const i32) -> __m128i {
            _mm_loadu_si128(data as *const __m128i)
        }

        #[inline(always)]
        unsafe fn _mm_add(a: __m128i, b: __m128i) -> __m128i {
            _mm_add_epi32(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: __m128i, b: __m128i) -> __m128i {
            _mm_cmpgt_epi32(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: __m128i, b: __m128i) -> __m128i {
            _mm_cmplt_epi32(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: __m128i, b: __m128i, mask: __m128i) -> __m128i {
            _mm_blendv_epi8(a, b, mask)
        }
    }

    impl_SIMDInit_Int!(i32, __m128i, __m128i, LANE_SIZE, SSE<Int>);

    impl SIMDArgMinMax<i32, __m128i, __m128i, LANE_SIZE, SCALAR<Int>> for SSE<Int> {
        #[target_feature(enable = "sse4.1")]
        unsafe fn argminmax(data: &[i32]) -> (usize, usize) {
            Self::_argminmax(data)
        }
    }
}

// -------------------------------------- AVX512 ---------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx512 {
    use super::super::config::AVX512;
    use super::*;

    const LANE_SIZE: usize = AVX512::<Int>::LANE_SIZE_32;

    impl SIMDOps<i32, __m512i, u16, LANE_SIZE> for AVX512<Int> {
        const INITIAL_INDEX: __m512i = unsafe {
            std::mem::transmute([
                0i32, 1i32, 2i32, 3i32, 4i32, 5i32, 6i32, 7i32, 8i32, 9i32, 10i32, 11i32, 12i32,
                13i32, 14i32, 15i32,
            ])
        };
        const INDEX_INCREMENT: __m512i =
            unsafe { std::mem::transmute([LANE_SIZE as i32; LANE_SIZE]) };
        const MAX_INDEX: usize = MAX_INDEX;

        #[inline(always)]
        unsafe fn _reg_to_arr(reg: __m512i) -> [i32; LANE_SIZE] {
            std::mem::transmute::<__m512i, [i32; LANE_SIZE]>(reg)
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const i32) -> __m512i {
            _mm512_loadu_si512(data as *const i32)
        }

        #[inline(always)]
        unsafe fn _mm_add(a: __m512i, b: __m512i) -> __m512i {
            _mm512_add_epi32(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: __m512i, b: __m512i) -> u16 {
            _mm512_cmpgt_epi32_mask(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: __m512i, b: __m512i) -> u16 {
            _mm512_cmplt_epi32_mask(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: __m512i, b: __m512i, mask: u16) -> __m512i {
            _mm512_mask_blend_epi32(mask, a, b)
        }
    }

    impl_SIMDInit_Int!(i32, __m512i, u16, LANE_SIZE, AVX512<Int>);

    impl SIMDArgMinMax<i32, __m512i, u16, LANE_SIZE, SCALAR<Int>> for AVX512<Int> {
        #[target_feature(enable = "avx512f")]
        unsafe fn argminmax(data: &[i32]) -> (usize, usize) {
            Self::_argminmax(data)
        }
    }
}

// --------------------------------------- NEON ----------------------------------------

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
mod neon {
    use super::super::config::NEON;
    use super::*;

    const LANE_SIZE: usize = NEON::<Int>::LANE_SIZE_32;

    impl SIMDOps<i32, int32x4_t, uint32x4_t, LANE_SIZE> for NEON<Int> {
        const INITIAL_INDEX: int32x4_t = unsafe { std::mem::transmute([0i32, 1i32, 2i32, 3i32]) };
        const INDEX_INCREMENT: int32x4_t =
            unsafe { std::mem::transmute([LANE_SIZE as i32; LANE_SIZE]) };
        const MAX_INDEX: usize = MAX_INDEX;

        #[inline(always)]
        unsafe fn _reg_to_arr(reg: int32x4_t) -> [i32; LANE_SIZE] {
            std::mem::transmute::<int32x4_t, [i32; LANE_SIZE]>(reg)
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const i32) -> int32x4_t {
            vld1q_s32(data)
        }

        #[inline(always)]
        unsafe fn _mm_add(a: int32x4_t, b: int32x4_t) -> int32x4_t {
            vaddq_s32(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: int32x4_t, b: int32x4_t) -> uint32x4_t {
            vcgtq_s32(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: int32x4_t, b: int32x4_t) -> uint32x4_t {
            vcltq_s32(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: int32x4_t, b: int32x4_t, mask: uint32x4_t) -> int32x4_t {
            vbslq_s32(mask, b, a)
        }
    }

    impl_SIMDInit_Int!(i32, int32x4_t, uint32x4_t, LANE_SIZE, NEON<Int>);

    impl SIMDArgMinMax<i32, int32x4_t, uint32x4_t, LANE_SIZE, SCALAR<Int>> for NEON<Int> {
        #[target_feature(enable = "neon")]
        unsafe fn argminmax(data: &[i32]) -> (usize, usize) {
            Self::_argminmax(data)
        }
    }
}

// ======================================= TESTS =======================================

#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "arm",
    target_arch = "aarch64",
))]
#[cfg(test)]
mod tests {
    use rstest::rstest;
    use rstest_reuse::{self, *};

    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    use crate::simd::config::NEON;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    use crate::simd::config::{AVX2, AVX512, SSE};
    use crate::{Int, SIMDArgMinMax, ScalarArgMinMax, SCALAR};

    use super::super::test_utils::{
        test_first_index_identical_values_argminmax, test_long_array_argminmax,
        test_random_runs_argminmax,
    };

    use dev_utils::utils;

    fn get_array_i32(n: usize) -> Vec<i32> {
        utils::get_random_array(n, i32::MIN, i32::MAX)
    }

    // ------------ Template for x86 / x86_64 -------------

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[template]
    #[rstest]
    #[case::sse(SSE {_dtype_strategy: Int}, is_x86_feature_detected!("sse4.1"))]
    #[case::avx2(AVX2 {_dtype_strategy: Int}, is_x86_feature_detected!("avx2"))]
    #[case::avx512(AVX512 {_dtype_strategy: Int}, is_x86_feature_detected!("avx512f"))]
    fn simd_implementations<T, SIMDV, SIMDM, const LANE_SIZE: usize>(
        #[case] _simd: T,
        #[case] simd_available: bool,
    ) {
    }

    // ------------ Template for ARM / AArch64 ------------

    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    #[template]
    #[rstest]
    #[case::neon(NEON {_dtype_strategy: Int}, true)]
    fn simd_implementations<T, SIMDV, SIMDM, const LANE_SIZE: usize>(
        #[case] _simd: T,
        #[case] simd_available: bool,
    ) {
    }

    // ----------------- The actual tests -----------------

    #[apply(simd_implementations)]
    fn test_first_index_is_returned_when_identical_values_found<
        T,
        SIMDV,
        SIMDM,
        const LANE_SIZE: usize,
    >(
        #[case] _simd: T, // This is just to make sure the template is applied
        #[case] simd_available: bool,
    ) where
        T: SIMDArgMinMax<i32, SIMDV, SIMDM, LANE_SIZE, SCALAR<Int>>,
        SIMDV: Copy,
        SIMDM: Copy,
    {
        if !simd_available {
            return;
        }
        test_first_index_identical_values_argminmax(SCALAR::<Int>::argminmax, T::argminmax);
    }

    #[apply(simd_implementations)]
    fn test_return_same_result<T, SIMDV, SIMDM, const LANE_SIZE: usize>(
        #[case] _simd: T, // This is just to make sure the template is applied
        #[case] simd_available: bool,
    ) where
        T: SIMDArgMinMax<i32, SIMDV, SIMDM, LANE_SIZE, SCALAR<Int>>,
        SIMDV: Copy,
        SIMDM: Copy,
    {
        if !simd_available {
            return;
        }
        test_long_array_argminmax(get_array_i32, SCALAR::<Int>::argminmax, T::argminmax);
        test_random_runs_argminmax(get_array_i32, SCALAR::<Int>::argminmax, T::argminmax);
    }
}

use super::config::SIMDInstructionSet;
use super::generic::{SIMDArgMinMaxFloatIgnoreNaN, SIMDOps, SIMDSetOps};
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "arm")]
use std::arch::arm::*;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// https://stackoverflow.com/a/3793950
const MAX_INDEX: usize = 1 << f32::MANTISSA_DIGITS;

// ------------------------------------------ AVX2 ------------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx2 {
    use super::super::config::{AVX2FloatIgnoreNaN, AVX2};
    use super::*;

    const LANE_SIZE: usize = AVX2::LANE_SIZE_32;

    impl SIMDOps<f32, __m256, __m256, LANE_SIZE> for AVX2FloatIgnoreNaN {
        const INITIAL_INDEX: __m256 = unsafe {
            std::mem::transmute([
                0.0f32, 1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32, 7.0f32,
            ])
        };
        const INDEX_INCREMENT: __m256 =
            unsafe { std::mem::transmute([LANE_SIZE as f32; LANE_SIZE]) };
        const MAX_INDEX: usize = MAX_INDEX;

        #[inline(always)]
        unsafe fn _reg_to_arr(reg: __m256) -> [f32; LANE_SIZE] {
            std::mem::transmute::<__m256, [f32; LANE_SIZE]>(reg)
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const f32) -> __m256 {
            _mm256_loadu_ps(data as *const f32)
        }

        #[inline(always)]
        unsafe fn _mm_add(a: __m256, b: __m256) -> __m256 {
            _mm256_add_ps(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: __m256, b: __m256) -> __m256 {
            _mm256_cmp_ps(a, b, _CMP_GT_OQ)
        }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: __m256, b: __m256) -> __m256 {
            _mm256_cmp_ps(b, a, _CMP_GT_OQ)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: __m256, b: __m256, mask: __m256) -> __m256 {
            _mm256_blendv_ps(a, b, mask)
        }
    }

    impl SIMDSetOps<f32, __m256> for AVX2FloatIgnoreNaN {
        #[inline(always)]
        unsafe fn _mm_set1(a: f32) -> __m256 {
            _mm256_set1_ps(a)
        }
    }

    impl SIMDArgMinMaxFloatIgnoreNaN<f32, __m256, __m256, LANE_SIZE> for AVX2FloatIgnoreNaN {
        #[target_feature(enable = "avx")]
        unsafe fn argminmax(data: &[f32]) -> (usize, usize) {
            Self::_argminmax(data)
        }
    }

    // ------------------------------------ TESTS --------------------------------------

    #[cfg(test)]
    mod tests {
        use super::AVX2FloatIgnoreNaN as AVX2;
        use super::SIMDArgMinMaxFloatIgnoreNaN;
        use crate::scalar::generic::scalar_argminmax;

        extern crate dev_utils;
        use dev_utils::utils;

        fn get_array_f32(n: usize) -> Vec<f32> {
            utils::get_random_array(n, f32::MIN, f32::MAX)
        }

        #[test]
        fn test_both_versions_return_the_same_results() {
            if !is_x86_feature_detected!("avx") {
                return;
            }

            let data: &[f32] = &get_array_f32(1025);
            assert_eq!(data.len() % 8, 1);

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX2::argminmax(data) };
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }

        #[test]
        fn test_first_index_is_returned_when_identical_values_found() {
            if !is_x86_feature_detected!("avx") {
                return;
            }

            let data = [
                10.,
                std::f32::MAX,
                6.,
                std::f32::NEG_INFINITY,
                std::f32::NEG_INFINITY,
                std::f32::MAX,
                10_000.0,
            ];
            let data: Vec<f32> = data.iter().map(|x| *x).collect();
            let data: &[f32] = &data;

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            assert_eq!(argmin_index, 3);
            assert_eq!(argmax_index, 1);

            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX2::argminmax(data) };
            assert_eq!(argmin_simd_index, 3);
            assert_eq!(argmax_simd_index, 1);
        }

        #[test]
        fn test_no_overflow() {
            if !is_x86_feature_detected!("avx") {
                return;
            }

            let n: usize = 1 << 25;
            let data: &[f32] = &get_array_f32(n);

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX2::argminmax(data) };
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }

        #[test]
        fn test_many_random_runs() {
            if !is_x86_feature_detected!("avx") {
                return;
            }

            for _ in 0..10_000 {
                let data: &[f32] = &get_array_f32(32 * 8 + 1);
                let (argmin_index, argmax_index) = scalar_argminmax(data);
                let (argmin_simd_index, argmax_simd_index) = unsafe { AVX2::argminmax(data) };
                assert_eq!(argmin_index, argmin_simd_index);
                assert_eq!(argmax_index, argmax_simd_index);
            }
        }
    }
}

// ----------------------------------------- SSE -----------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod sse {
    use super::super::config::{SSEFloatIgnoreNaN, SSE};
    use super::*;

    const LANE_SIZE: usize = SSE::LANE_SIZE_32;

    impl SIMDOps<f32, __m128, __m128, LANE_SIZE> for SSEFloatIgnoreNaN {
        const INITIAL_INDEX: __m128 =
            unsafe { std::mem::transmute([0.0f32, 1.0f32, 2.0f32, 3.0f32]) };
        const INDEX_INCREMENT: __m128 =
            unsafe { std::mem::transmute([LANE_SIZE as f32; LANE_SIZE]) };
        const MAX_INDEX: usize = MAX_INDEX;

        #[inline(always)]
        unsafe fn _reg_to_arr(reg: __m128) -> [f32; LANE_SIZE] {
            std::mem::transmute::<__m128, [f32; LANE_SIZE]>(reg)
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const f32) -> __m128 {
            _mm_loadu_ps(data as *const f32)
        }

        #[inline(always)]
        unsafe fn _mm_add(a: __m128, b: __m128) -> __m128 {
            _mm_add_ps(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: __m128, b: __m128) -> __m128 {
            _mm_cmpgt_ps(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: __m128, b: __m128) -> __m128 {
            _mm_cmplt_ps(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: __m128, b: __m128, mask: __m128) -> __m128 {
            _mm_blendv_ps(a, b, mask)
        }
    }

    impl SIMDSetOps<f32, __m128> for SSEFloatIgnoreNaN {
        #[inline(always)]
        unsafe fn _mm_set1(a: f32) -> __m128 {
            _mm_set1_ps(a)
        }
    }

    impl SIMDArgMinMaxFloatIgnoreNaN<f32, __m128, __m128, LANE_SIZE> for SSEFloatIgnoreNaN {
        #[target_feature(enable = "sse4.1")]
        unsafe fn argminmax(data: &[f32]) -> (usize, usize) {
            Self::_argminmax(data)
        }
    }

    // ------------------------------------ TESTS --------------------------------------

    #[cfg(test)]
    mod tests {
        use super::SIMDArgMinMaxFloatIgnoreNaN;
        use super::SSEFloatIgnoreNaN as SSE;
        use crate::scalar::generic::scalar_argminmax;

        extern crate dev_utils;
        use dev_utils::utils;

        fn get_array_f32(n: usize) -> Vec<f32> {
            utils::get_random_array(n, f32::MIN, f32::MAX)
        }

        #[test]
        fn test_both_versions_return_the_same_results() {
            let data: &[f32] = &get_array_f32(1025);
            assert_eq!(data.len() % 4, 1);

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            let (argmin_simd_index, argmax_simd_index) = unsafe { SSE::argminmax(data) };
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }

        #[test]
        fn test_first_index_is_returned_when_identical_values_found() {
            let data = [
                10.,
                std::f32::MAX,
                6.,
                std::f32::NEG_INFINITY,
                std::f32::NEG_INFINITY,
                std::f32::MAX,
                10_000.0,
            ];
            let data: Vec<f32> = data.iter().map(|x| *x).collect();
            let data: &[f32] = &data;

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            assert_eq!(argmin_index, 3);
            assert_eq!(argmax_index, 1);

            let (argmin_simd_index, argmax_simd_index) = unsafe { SSE::argminmax(data) };
            assert_eq!(argmin_simd_index, 3);
            assert_eq!(argmax_simd_index, 1);
        }

        #[test]
        fn test_no_overflow() {
            let n: usize = 1 << 25;
            let data: &[f32] = &get_array_f32(n);

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            let (argmin_simd_index, argmax_simd_index) = unsafe { SSE::argminmax(data) };
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }

        #[test]
        fn test_many_random_runs() {
            for _ in 0..10_000 {
                let data: &[f32] = &get_array_f32(32 * 4 + 1);
                let (argmin_index, argmax_index) = scalar_argminmax(data);
                let (argmin_simd_index, argmax_simd_index) = unsafe { SSE::argminmax(data) };
                assert_eq!(argmin_index, argmin_simd_index);
                assert_eq!(argmax_index, argmax_simd_index);
            }
        }
    }
}

// --------------------------------------- AVX512 ----------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx512 {
    use super::super::config::{AVX512FloatIgnoreNaN, AVX512};
    use super::*;

    const LANE_SIZE: usize = AVX512::LANE_SIZE_32;

    impl SIMDOps<f32, __m512, u16, LANE_SIZE> for AVX512FloatIgnoreNaN {
        const INITIAL_INDEX: __m512 = unsafe {
            std::mem::transmute([
                0.0f32, 1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32, 7.0f32, 8.0f32, 9.0f32,
                10.0f32, 11.0f32, 12.0f32, 13.0f32, 14.0f32, 15.0f32,
            ])
        };
        const INDEX_INCREMENT: __m512 =
            unsafe { std::mem::transmute([LANE_SIZE as f32; LANE_SIZE]) };
        const MAX_INDEX: usize = MAX_INDEX;

        #[inline(always)]
        unsafe fn _reg_to_arr(reg: __m512) -> [f32; LANE_SIZE] {
            std::mem::transmute::<__m512, [f32; LANE_SIZE]>(reg)
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const f32) -> __m512 {
            _mm512_loadu_ps(data as *const f32)
        }

        #[inline(always)]
        unsafe fn _mm_add(a: __m512, b: __m512) -> __m512 {
            _mm512_add_ps(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: __m512, b: __m512) -> u16 {
            _mm512_cmp_ps_mask(a, b, _CMP_GT_OQ)
        }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: __m512, b: __m512) -> u16 {
            _mm512_cmp_ps_mask(a, b, _CMP_LT_OQ)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: __m512, b: __m512, mask: u16) -> __m512 {
            _mm512_mask_blend_ps(mask, a, b)
        }
    }

    impl SIMDSetOps<f32, __m512> for AVX512FloatIgnoreNaN {
        #[inline(always)]
        unsafe fn _mm_set1(a: f32) -> __m512 {
            _mm512_set1_ps(a)
        }
    }

    impl SIMDArgMinMaxFloatIgnoreNaN<f32, __m512, u16, LANE_SIZE> for AVX512FloatIgnoreNaN {
        #[target_feature(enable = "avx512f")]
        unsafe fn argminmax(data: &[f32]) -> (usize, usize) {
            Self::_argminmax(data)
        }
    }

    // ------------------------------------ TESTS --------------------------------------

    #[cfg(test)]
    mod tests {
        use super::AVX512FloatIgnoreNaN as AVX512;
        use super::SIMDArgMinMaxFloatIgnoreNaN;
        use crate::scalar::generic::scalar_argminmax;

        extern crate dev_utils;
        use dev_utils::utils;

        fn get_array_f32(n: usize) -> Vec<f32> {
            utils::get_random_array(n, f32::MIN, f32::MAX)
        }

        #[test]
        fn test_both_versions_return_the_same_results() {
            if !is_x86_feature_detected!("avx512f") {
                return;
            }

            let data: &[f32] = &get_array_f32(1025);
            assert_eq!(data.len() % 16, 1);

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX512::argminmax(data) };
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }

        #[test]
        fn test_first_index_is_returned_when_identical_values_found() {
            if !is_x86_feature_detected!("avx512f") {
                return;
            }

            let data = [
                10.,
                std::f32::MAX,
                6.,
                std::f32::NEG_INFINITY,
                std::f32::NEG_INFINITY,
                std::f32::MAX,
                10_000.0,
            ];
            let data: Vec<f32> = data.iter().map(|x| *x).collect();
            let data: &[f32] = &data;

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            assert_eq!(argmin_index, 3);
            assert_eq!(argmax_index, 1);

            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX512::argminmax(data) };
            assert_eq!(argmin_simd_index, 3);
            assert_eq!(argmax_simd_index, 1);
        }

        #[test]
        fn test_no_overflow() {
            if !is_x86_feature_detected!("avx512f") {
                return;
            }

            let n: usize = 1 << 25;
            let data: &[f32] = &get_array_f32(n);

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX512::argminmax(data) };
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }

        #[test]
        fn test_many_random_runs() {
            if !is_x86_feature_detected!("avx512f") {
                return;
            }

            for _ in 0..10_000 {
                let data: &[f32] = &get_array_f32(32 * 16 + 1);
                let (argmin_index, argmax_index) = scalar_argminmax(data);
                let (argmin_simd_index, argmax_simd_index) = unsafe { AVX512::argminmax(data) };
                assert_eq!(argmin_index, argmin_simd_index);
                assert_eq!(argmax_index, argmax_simd_index);
            }
        }
    }
}

// ---------------------------------------- NEON -----------------------------------------

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
mod neon {
    use super::super::config::{NEONFloatIgnoreNaN, NEON};
    use super::*;

    const LANE_SIZE: usize = NEON::LANE_SIZE_32;

    impl SIMDOps<f32, float32x4_t, uint32x4_t, LANE_SIZE> for NEONFloatIgnoreNaN {
        const INITIAL_INDEX: float32x4_t =
            unsafe { std::mem::transmute([0.0f32, 1.0f32, 2.0f32, 3.0f32]) };
        const INDEX_INCREMENT: float32x4_t =
            unsafe { std::mem::transmute([LANE_SIZE as f32; LANE_SIZE]) };
        const MAX_INDEX: usize = MAX_INDEX;

        #[inline(always)]
        unsafe fn _reg_to_arr(reg: float32x4_t) -> [f32; LANE_SIZE] {
            std::mem::transmute::<float32x4_t, [f32; LANE_SIZE]>(reg)
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const f32) -> float32x4_t {
            vld1q_f32(data as *const f32)
        }

        #[inline(always)]
        unsafe fn _mm_add(a: float32x4_t, b: float32x4_t) -> float32x4_t {
            vaddq_f32(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: float32x4_t, b: float32x4_t) -> uint32x4_t {
            vcgtq_f32(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: float32x4_t, b: float32x4_t) -> uint32x4_t {
            vcltq_f32(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: float32x4_t, b: float32x4_t, mask: uint32x4_t) -> float32x4_t {
            vbslq_f32(mask, b, a)
        }
    }

    impl SIMDSetOps<f32, float32x4_t> for NEONFloatIgnoreNaN {
        #[inline(always)]
        unsafe fn _mm_set1(a: f32) -> float32x4_t {
            vdupq_n_f32(a as f32)
        }
    }

    impl SIMDArgMinMaxFloatIgnoreNaN<f32, float32x4_t, uint32x4_t, LANE_SIZE> for NEONFloatIgnoreNaN {
        #[target_feature(enable = "neon")]
        unsafe fn argminmax(data: &[f32]) -> (usize, usize) {
            Self::_argminmax(data)
        }
    }

    // ------------------------------------ TESTS --------------------------------------

    #[cfg(test)]
    mod tests {
        use super::NEONFloatIgnoreNaN as NEON;
        use super::SIMDArgMinMaxFloatIgnoreNaN;
        use crate::scalar::generic::scalar_argminmax;

        extern crate dev_utils;
        use dev_utils::utils;

        fn get_array_f32(n: usize) -> Vec<f32> {
            utils::get_random_array(n, f32::MIN, f32::MAX)
        }

        #[test]
        fn test_both_versions_return_the_same_results() {
            let data: &[f32] = &get_array_f32(1025);
            assert_eq!(data.len() % 4, 1);

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            let (argmin_simd_index, argmax_simd_index) = unsafe { NEON::argminmax(data) };
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }

        #[test]
        fn test_first_index_is_returned_when_identical_values_found() {
            let data = [
                10.,
                std::f32::MAX,
                6.,
                std::f32::NEG_INFINITY,
                std::f32::NEG_INFINITY,
                std::f32::MAX,
                10_000.0,
            ];
            let data: Vec<f32> = data.iter().map(|x| *x).collect();
            let data: &[f32] = &data;

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            assert_eq!(argmin_index, 3);
            assert_eq!(argmax_index, 1);

            let (argmin_simd_index, argmax_simd_index) = unsafe { NEON::argminmax(data) };
            assert_eq!(argmin_simd_index, 3);
            assert_eq!(argmax_simd_index, 1);
        }

        #[test]
        fn test_no_overflow() {
            let n: usize = 1 << 25;
            let data: &[f32] = &get_array_f32(n);

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            let (argmin_simd_index, argmax_simd_index) = unsafe { NEON::argminmax(data) };
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }

        #[test]
        fn test_many_random_runs() {
            for _ in 0..10_000 {
                let data: &[f32] = &get_array_f32(32 * 4 + 1);
                let (argmin_index, argmax_index) = scalar_argminmax(data);
                let (argmin_simd_index, argmax_simd_index) = unsafe { NEON::argminmax(data) };
                assert_eq!(argmin_index, argmin_simd_index);
                assert_eq!(argmax_index, argmax_simd_index);
            }
        }
    }
}

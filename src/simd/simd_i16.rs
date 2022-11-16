use super::config::SIMDInstructionSet;
use super::generic::SIMD;
use ndarray::ArrayView1;
#[cfg(target_arch = "arm")]
use std::arch::arm::*;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// ------------------------------------------ AVX2 ------------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx2 {
    use super::super::config::AVX2;
    use super::*;

    const LANE_SIZE: usize = AVX2::LANE_SIZE_16;

    impl SIMD<i16, __m256i, __m256i, LANE_SIZE> for AVX2 {
        const INITIAL_INDEX: __m256i = unsafe {
            std::mem::transmute([
                0i16, 1i16, 2i16, 3i16, 4i16, 5i16, 6i16, 7i16, 8i16, 9i16, 10i16, 11i16, 12i16,
                13i16, 14i16, 15i16,
            ])
        };
        const MAX_INDEX: usize = i16::MAX as usize;

        #[inline(always)]
        unsafe fn _reg_to_arr(reg: __m256i) -> [i16; LANE_SIZE] {
            std::mem::transmute::<__m256i, [i16; LANE_SIZE]>(reg)
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const i16) -> __m256i {
            _mm256_loadu_si256(data as *const __m256i)
        }

        #[inline(always)]
        unsafe fn _mm_set1(a: usize) -> __m256i {
            _mm256_set1_epi16(a as i16)
        }

        #[inline(always)]
        unsafe fn _mm_add(a: __m256i, b: __m256i) -> __m256i {
            _mm256_add_epi16(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: __m256i, b: __m256i) -> __m256i {
            _mm256_cmpgt_epi16(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: __m256i, b: __m256i) -> __m256i {
            _mm256_cmpgt_epi16(b, a)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: __m256i, b: __m256i, mask: __m256i) -> __m256i {
            _mm256_blendv_epi8(a, b, mask)
        }

        // ------------------------------------ ARGMINMAX --------------------------------------

        #[target_feature(enable = "avx2")]
        unsafe fn argminmax(data: ArrayView1<i16>) -> (usize, usize) {
            Self::_argminmax(data)
        }
    }

    // ------------------------------------ TESTS --------------------------------------

    #[cfg(test)]
    mod tests {
        use super::{AVX2, SIMD};
        use crate::scalar::generic::scalar_argminmax;

        use ndarray::Array1;

        extern crate dev_utils;
        use dev_utils::utils;

        fn get_array_i16(n: usize) -> Array1<i16> {
            utils::get_random_array(n, i16::MIN, i16::MAX)
        }

        #[test]
        fn test_both_versions_return_the_same_results() {
            let data = get_array_i16(513);
            assert_eq!(data.len() % 16, 1);

            let (argmin_index, argmax_index) = scalar_argminmax(data.view());
            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX2::argminmax(data.view()) };
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }

        #[test]
        fn test_first_index_is_returned_when_identical_values_found() {
            let data = [
                10,
                std::i16::MIN,
                6,
                9,
                9,
                22,
                std::i16::MAX,
                4,
                std::i16::MAX,
            ];
            let data: Vec<i16> = data.iter().map(|x| *x).collect();
            let data = Array1::from(data);

            let (argmin_index, argmax_index) = scalar_argminmax(data.view());
            assert_eq!(argmin_index, 1);
            assert_eq!(argmax_index, 6);

            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX2::argminmax(data.view()) };
            assert_eq!(argmin_simd_index, 1);
            assert_eq!(argmax_simd_index, 6);
        }

        #[test]
        fn test_no_overflow() {
            let n: usize = 1 << 18;
            let data = get_array_i16(n);

            let (argmin_index, argmax_index) = scalar_argminmax(data.view());
            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX2::argminmax(data.view()) };
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }

        #[test]
        fn test_many_random_runs() {
            for _ in 0..10_000 {
                let data = get_array_i16(32 * 2 + 1);
                let (argmin_index, argmax_index) = scalar_argminmax(data.view());
                let (argmin_simd_index, argmax_simd_index) =
                    unsafe { AVX2::argminmax(data.view()) };
                assert_eq!(argmin_index, argmin_simd_index);
                assert_eq!(argmax_index, argmax_simd_index);
            }
        }
    }
}

// ----------------------------------------- SSE -----------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod sse {
    use super::super::config::SSE;
    use super::*;

    const LANE_SIZE: usize = SSE::LANE_SIZE_16;

    impl SIMD<i16, __m128i, __m128i, LANE_SIZE> for SSE {
        const INITIAL_INDEX: __m128i =
            unsafe { std::mem::transmute([0i16, 1i16, 2i16, 3i16, 4i16, 5i16, 6i16, 7i16]) };
        const MAX_INDEX: usize = i16::MAX as usize;

        #[inline(always)]
        unsafe fn _reg_to_arr(reg: __m128i) -> [i16; LANE_SIZE] {
            std::mem::transmute::<__m128i, [i16; LANE_SIZE]>(reg)
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const i16) -> __m128i {
            _mm_loadu_si128(data as *const __m128i)
        }

        #[inline(always)]
        unsafe fn _mm_set1(a: usize) -> __m128i {
            _mm_set1_epi16(a as i16)
        }

        #[inline(always)]
        unsafe fn _mm_add(a: __m128i, b: __m128i) -> __m128i {
            _mm_add_epi16(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: __m128i, b: __m128i) -> __m128i {
            _mm_cmpgt_epi16(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: __m128i, b: __m128i) -> __m128i {
            _mm_cmplt_epi16(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: __m128i, b: __m128i, mask: __m128i) -> __m128i {
            _mm_blendv_epi8(a, b, mask)
        }

        // ------------------------------------ ARGMINMAX --------------------------------------

        #[target_feature(enable = "sse4.1")]
        unsafe fn argminmax(data: ArrayView1<i16>) -> (usize, usize) {
            Self::_argminmax(data)
        }
    }

    // ------------------------------------ TESTS --------------------------------------

    #[cfg(test)]
    mod tests {
        use super::{SIMD, SSE};
        use crate::scalar::generic::scalar_argminmax;

        use ndarray::Array1;

        extern crate dev_utils;
        use dev_utils::utils;

        fn get_array_i16(n: usize) -> Array1<i16> {
            utils::get_random_array(n, i16::MIN, i16::MAX)
        }

        #[test]
        fn test_both_versions_return_the_same_results() {
            let data = get_array_i16(513);
            assert_eq!(data.len() % 8, 1);

            let (argmin_index, argmax_index) = scalar_argminmax(data.view());
            let (argmin_simd_index, argmax_simd_index) = unsafe { SSE::argminmax(data.view()) };
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }

        #[test]
        fn test_first_index_is_returned_when_identical_values_found() {
            let data = [
                10,
                std::i16::MIN,
                6,
                9,
                9,
                22,
                std::i16::MAX,
                4,
                std::i16::MAX,
            ];
            let data: Vec<i16> = data.iter().map(|x| *x).collect();
            let data = Array1::from(data);

            let (argmin_index, argmax_index) = scalar_argminmax(data.view());
            assert_eq!(argmin_index, 1);
            assert_eq!(argmax_index, 6);

            let (argmin_simd_index, argmax_simd_index) = unsafe { SSE::argminmax(data.view()) };
            assert_eq!(argmin_simd_index, 1);
            assert_eq!(argmax_simd_index, 6);
        }

        #[test]
        fn test_no_overflow() {
            let n: usize = 1 << 18;
            let data = get_array_i16(n);

            let (argmin_index, argmax_index) = scalar_argminmax(data.view());
            let (argmin_simd_index, argmax_simd_index) = unsafe { SSE::argminmax(data.view()) };
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }

        #[test]
        fn test_many_random_runs() {
            for _ in 0..10_000 {
                let data = get_array_i16(8 * 2 + 1);
                let (argmin_index, argmax_index) = scalar_argminmax(data.view());
                let (argmin_simd_index, argmax_simd_index) = unsafe { SSE::argminmax(data.view()) };
                assert_eq!(argmin_index, argmin_simd_index);
                assert_eq!(argmax_index, argmax_simd_index);
            }
        }
    }
}

// --------------------------------------- AVX512 ----------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx512 {
    use super::super::config::AVX512;
    use super::*;

    const LANE_SIZE: usize = AVX512::LANE_SIZE_16;

    impl SIMD<i16, __m512i, u32, LANE_SIZE> for AVX512 {
        const INITIAL_INDEX: __m512i = unsafe {
            std::mem::transmute([
                0i16, 1i16, 2i16, 3i16, 4i16, 5i16, 6i16, 7i16, 8i16, 9i16, 10i16, 11i16, 12i16,
                13i16, 14i16, 15i16, 16i16, 17i16, 18i16, 19i16, 20i16, 21i16, 22i16, 23i16, 24i16,
                25i16, 26i16, 27i16, 28i16, 29i16, 30i16, 31i16,
            ])
        };
        const MAX_INDEX: usize = i16::MAX as usize;

        #[inline(always)]
        unsafe fn _reg_to_arr(reg: __m512i) -> [i16; LANE_SIZE] {
            std::mem::transmute::<__m512i, [i16; LANE_SIZE]>(reg)
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const i16) -> __m512i {
            _mm512_loadu_epi16(data as *const i16)
        }

        #[inline(always)]
        unsafe fn _mm_set1(a: usize) -> __m512i {
            _mm512_set1_epi16(a as i16)
        }

        #[inline(always)]
        unsafe fn _mm_add(a: __m512i, b: __m512i) -> __m512i {
            _mm512_add_epi16(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: __m512i, b: __m512i) -> u32 {
            _mm512_cmpgt_epi16_mask(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: __m512i, b: __m512i) -> u32 {
            _mm512_cmplt_epi16_mask(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: __m512i, b: __m512i, mask: u32) -> __m512i {
            _mm512_mask_blend_epi16(mask, a, b)
        }

        // ------------------------------------ ARGMINMAX --------------------------------------

        #[target_feature(enable = "avx512bw")]
        unsafe fn argminmax(data: ArrayView1<i16>) -> (usize, usize) {
            Self::_argminmax(data)
        }
    }

    // ------------------------------------ TESTS --------------------------------------

    #[cfg(test)]
    mod tests {
        use super::{AVX512, SIMD};
        use crate::scalar::generic::scalar_argminmax;

        use ndarray::Array1;

        extern crate dev_utils;
        use dev_utils::utils;

        fn get_array_i16(n: usize) -> Array1<i16> {
            utils::get_random_array(n, i16::MIN, i16::MAX)
        }

        #[test]
        fn test_both_versions_return_the_same_results() {
            if !is_x86_feature_detected!("avx512bw") {
                return;
            }

            let data = get_array_i16(513);
            assert_eq!(data.len() % 8, 1);

            let (argmin_index, argmax_index) = scalar_argminmax(data.view());
            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX512::argminmax(data.view()) };
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }

        #[test]
        fn test_first_index_is_returned_when_identical_values_found() {
            if !is_x86_feature_detected!("avx512bw") {
                return;
            }

            let data = [
                10,
                std::i16::MIN,
                6,
                9,
                9,
                22,
                std::i16::MAX,
                4,
                std::i16::MAX,
            ];
            let data: Vec<i16> = data.iter().map(|x| *x).collect();
            let data = Array1::from(data);

            let (argmin_index, argmax_index) = scalar_argminmax(data.view());
            assert_eq!(argmin_index, 1);
            assert_eq!(argmax_index, 6);

            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX512::argminmax(data.view()) };
            assert_eq!(argmin_simd_index, 1);
            assert_eq!(argmax_simd_index, 6);
        }

        #[test]
        fn test_no_overflow() {
            if !is_x86_feature_detected!("avx512bw") {
                return;
            }

            let n: usize = 1 << 18;
            let data = get_array_i16(n);

            let (argmin_index, argmax_index) = scalar_argminmax(data.view());
            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX512::argminmax(data.view()) };
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }

        #[test]
        fn test_many_random_runs() {
            if !is_x86_feature_detected!("avx512bw") {
                return;
            }

            for _ in 0..10_000 {
                let data = get_array_i16(1025);
                let (argmin_index, argmax_index) = scalar_argminmax(data.view());
                let (argmin_simd_index, argmax_simd_index) =
                    unsafe { AVX512::argminmax(data.view()) };
                assert_eq!(argmin_index, argmin_simd_index);
                assert_eq!(argmax_index, argmax_simd_index);
            }
        }
    }
}

// ---------------------------------------- NEON -----------------------------------------

#[cfg(target_arch = "arm")]
mod neon {
    use super::super::config::NEON;
    use super::*;

    const LANE_SIZE: usize = NEON::LANE_SIZE_16;

    impl SIMD<i16, int16x8_t, uint16x8_t, LANE_SIZE> for NEON {
        const INITIAL_INDEX: int16x8_t =
            unsafe { std::mem::transmute([0i16, 1i16, 2i16, 3i16, 4i16, 5i16, 6i16, 7i16]) };
        const MAX_INDEX: usize = i16::MAX as usize;

        #[inline(always)]
        unsafe fn _reg_to_arr(reg: int16x8_t) -> [i16; LANE_SIZE] {
            std::mem::transmute::<int16x8_t, [i16; LANE_SIZE]>(reg)
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const i16) -> int16x8_t {
            vld1q_s16(data as *const i16)
        }

        #[inline(always)]
        unsafe fn _mm_set1(a: usize) -> int16x8_t {
            vdupq_n_s16(a as i16)
        }

        #[inline(always)]
        unsafe fn _mm_add(a: int16x8_t, b: int16x8_t) -> int16x8_t {
            vaddq_s16(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: int16x8_t, b: int16x8_t) -> uint16x8_t {
            vcgtq_s16(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: int16x8_t, b: int16x8_t) -> uint16x8_t {
            vcltq_s16(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: int16x8_t, b: int16x8_t, mask: uint16x8_t) -> int16x8_t {
            vbslq_s16(mask, b, a)
        }

        // ------------------------------------ ARGMINMAX --------------------------------------

        #[target_feature(enable = "neon")]
        unsafe fn argminmax(data: ArrayView1<i16>) -> (usize, usize) {
            Self::_argminmax(data)
        }
    }

    // ------------------------------------ TESTS --------------------------------------

    #[cfg(test)]
    mod tests {
        use super::{NEON, SIMD};
        use crate::scalar::generic::scalar_argminmax;

        use ndarray::Array1;

        extern crate dev_utils;
        use dev_utils::utils;

        fn get_array_i16(n: usize) -> Array1<i16> {
            utils::get_random_array(n, i16::MIN, i16::MAX)
        }

        #[test]
        fn test_both_versions_return_the_same_results() {
            let data = get_array_i16(513);
            assert_eq!(data.len() % 8, 1);

            let (argmin_index, argmax_index) = scalar_argminmax(data.view());
            let (argmin_simd_index, argmax_simd_index) = unsafe { NEON::argminmax(data.view()) };
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }

        #[test]
        fn test_first_index_is_returned_when_identical_values_found() {
            let data = [
                10,
                std::i16::MIN,
                6,
                9,
                9,
                22,
                std::i16::MAX,
                4,
                std::i16::MAX,
            ];
            let data: Vec<i16> = data.iter().map(|x| *x).collect();
            let data = Array1::from(data);

            let (argmin_index, argmax_index) = scalar_argminmax(data.view());
            assert_eq!(argmin_index, 1);
            assert_eq!(argmax_index, 6);

            let (argmin_simd_index, argmax_simd_index) = unsafe { NEON::argminmax(data.view()) };
            assert_eq!(argmin_simd_index, 1);
            assert_eq!(argmax_simd_index, 6);
        }

        #[test]
        fn test_no_overflow() {
            let n: usize = 1 << 18;
            let data = get_array_i16(n);

            let (argmin_index, argmax_index) = scalar_argminmax(data.view());
            let (argmin_simd_index, argmax_simd_index) = unsafe { NEON::argminmax(data.view()) };
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }

        #[test]
        fn test_many_random_runs() {
            for _ in 0..10_000 {
                let data = get_array_i16(16 * 8 + 1);
                let (argmin_index, argmax_index) = scalar_argminmax(data.view());
                let (argmin_simd_index, argmax_simd_index) =
                    unsafe { NEON::argminmax(data.view()) };
                assert_eq!(argmin_index, argmin_simd_index);
                assert_eq!(argmax_index, argmax_simd_index);
            }
        }
    }
}

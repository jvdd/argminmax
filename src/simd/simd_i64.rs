#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use super::config::SIMDInstructionSet;
use super::generic::SIMD;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

const MAX_INDEX: usize = i64::MAX as usize;

const MIN_VALUE: i64 = i64::MIN;
const MAX_VALUE: i64 = i64::MAX;

// ------------------------------------------ AVX2 ------------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx2 {
    use super::super::config::AVX2;
    use super::*;

    const LANE_SIZE: usize = AVX2::LANE_SIZE_64;

    impl SIMD<i64, __m256i, __m256i, LANE_SIZE> for AVX2 {
        const INITIAL_INDEX: __m256i = unsafe { std::mem::transmute([0i64, 1i64, 2i64, 3i64]) };
        const INDEX_INCREMENT: __m256i =
            unsafe { std::mem::transmute([LANE_SIZE as i64; LANE_SIZE]) };
        const MAX_INDEX: usize = MAX_INDEX;

        const MIN_VALUE: i64 = MIN_VALUE;
        const MAX_VALUE: i64 = MAX_VALUE;

        #[inline(always)]
        unsafe fn _reg_to_arr(reg: __m256i) -> [i64; LANE_SIZE] {
            std::mem::transmute::<__m256i, [i64; LANE_SIZE]>(reg)
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const i64) -> __m256i {
            _mm256_loadu_si256(data as *const __m256i)
        }

        #[inline(always)]
        unsafe fn _mm_set1(a: i64) -> __m256i {
            _mm256_set1_epi64x(a)
        }

        #[inline(always)]
        unsafe fn _mm_add(a: __m256i, b: __m256i) -> __m256i {
            _mm256_add_epi64(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: __m256i, b: __m256i) -> __m256i {
            _mm256_cmpgt_epi64(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: __m256i, b: __m256i) -> __m256i {
            _mm256_cmpgt_epi64(b, a)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: __m256i, b: __m256i, mask: __m256i) -> __m256i {
            _mm256_blendv_epi8(a, b, mask)
        }

        // ------------------------------------ ARGMINMAX --------------------------------------

        #[target_feature(enable = "avx2")]
        unsafe fn argminmax(data: &[i64]) -> (usize, usize) {
            Self::_argminmax(data)
        }
    }

    // ------------------------------------ TESTS --------------------------------------

    #[cfg(test)]
    mod tests {
        use super::{AVX2, SIMD};
        use crate::scalar::generic::scalar_argminmax;

        extern crate dev_utils;
        use dev_utils::utils;

        fn get_array_i64(n: usize) -> Vec<i64> {
            utils::get_random_array(n, i64::MIN, i64::MAX)
        }

        #[test]
        fn test_both_versions_return_the_same_results() {
            if !is_x86_feature_detected!("avx2") {
                return;
            }

            let data: &[i64] = &get_array_i64(1025);
            assert_eq!(data.len() % 4, 1);

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX2::argminmax(data) };
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }

        #[test]
        fn test_first_index_is_returned_when_identical_values_found() {
            if !is_x86_feature_detected!("avx2") {
                return;
            }

            let data = [
                std::i64::MIN,
                std::i64::MIN,
                4,
                6,
                9,
                std::i64::MAX,
                22,
                std::i64::MAX,
            ];
            let data: Vec<i64> = data.iter().map(|x| *x).collect();
            let data: &[i64] = &data;

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            assert_eq!(argmin_index, 0);
            assert_eq!(argmax_index, 5);

            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX2::argminmax(data) };
            assert_eq!(argmin_simd_index, 0);
            assert_eq!(argmax_simd_index, 5);
        }

        #[test]
        fn test_many_random_runs() {
            if !is_x86_feature_detected!("avx2") {
                return;
            }

            for _ in 0..10_000 {
                let data: &[i64] = &get_array_i64(32 * 8 + 1);
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
    use super::super::config::SSE;
    use super::*;

    const LANE_SIZE: usize = SSE::LANE_SIZE_64;

    impl SIMD<i64, __m128i, __m128i, LANE_SIZE> for SSE {
        const INITIAL_INDEX: __m128i = unsafe { std::mem::transmute([0i64, 1i64]) };
        const INDEX_INCREMENT: __m128i =
            unsafe { std::mem::transmute([LANE_SIZE as i64; LANE_SIZE]) };
        const MAX_INDEX: usize = MAX_INDEX;

        const MIN_VALUE: i64 = MIN_VALUE;
        const MAX_VALUE: i64 = MAX_VALUE;

        #[inline(always)]
        unsafe fn _reg_to_arr(reg: __m128i) -> [i64; LANE_SIZE] {
            std::mem::transmute::<__m128i, [i64; LANE_SIZE]>(reg)
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const i64) -> __m128i {
            _mm_loadu_si128(data as *const __m128i)
        }

        #[inline(always)]
        unsafe fn _mm_set1(a: i64) -> __m128i {
            _mm_set1_epi64x(a)
        }

        #[inline(always)]
        unsafe fn _mm_add(a: __m128i, b: __m128i) -> __m128i {
            _mm_add_epi64(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: __m128i, b: __m128i) -> __m128i {
            _mm_cmpgt_epi64(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: __m128i, b: __m128i) -> __m128i {
            _mm_cmpgt_epi64(b, a)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: __m128i, b: __m128i, mask: __m128i) -> __m128i {
            _mm_blendv_epi8(a, b, mask)
        }

        // ------------------------------------ ARGMINMAX --------------------------------------

        #[target_feature(enable = "sse4.2")]
        unsafe fn argminmax(data: &[i64]) -> (usize, usize) {
            Self::_argminmax(data)
        }
    }

    // ------------------------------------ TESTS --------------------------------------

    #[cfg(test)]
    mod tests {
        use super::{SIMD, SSE};
        use crate::scalar::generic::scalar_argminmax;

        extern crate dev_utils;
        use dev_utils::utils;

        fn get_array_i64(n: usize) -> Vec<i64> {
            utils::get_random_array(n, i64::MIN, i64::MAX)
        }

        #[test]
        fn test_both_versions_return_the_same_results() {
            let data: &[i64] = &get_array_i64(1025);
            assert_eq!(data.len() % 2, 1);

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            let (argmin_simd_index, argmax_simd_index) = unsafe { SSE::argminmax(data) };
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }

        #[test]
        fn test_first_index_is_returned_when_identical_values_found() {
            let data = [
                std::i64::MIN,
                std::i64::MIN,
                4,
                6,
                9,
                std::i64::MAX,
                22,
                std::i64::MAX,
            ];
            let data: Vec<i64> = data.iter().map(|x| *x).collect();
            let data: &[i64] = &data;

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            assert_eq!(argmin_index, 0);
            assert_eq!(argmax_index, 5);

            let (argmin_simd_index, argmax_simd_index) = unsafe { SSE::argminmax(data) };
            assert_eq!(argmin_simd_index, 0);
            assert_eq!(argmax_simd_index, 5);
        }

        #[test]
        fn test_many_random_runs() {
            for _ in 0..10_000 {
                let data: &[i64] = &get_array_i64(16 * 2 + 1);
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
    use super::super::config::AVX512;
    use super::*;

    const LANE_SIZE: usize = AVX512::LANE_SIZE_64;

    impl SIMD<i64, __m512i, u8, LANE_SIZE> for AVX512 {
        const INITIAL_INDEX: __m512i =
            unsafe { std::mem::transmute([0i64, 1i64, 2i64, 3i64, 4i64, 5i64, 6i64, 7i64]) };
        const INDEX_INCREMENT: __m512i =
            unsafe { std::mem::transmute([LANE_SIZE as i64; LANE_SIZE]) };
        const MAX_INDEX: usize = MAX_INDEX;

        const MIN_VALUE: i64 = MIN_VALUE;
        const MAX_VALUE: i64 = MAX_VALUE;

        #[inline(always)]
        unsafe fn _reg_to_arr(reg: __m512i) -> [i64; LANE_SIZE] {
            std::mem::transmute::<__m512i, [i64; LANE_SIZE]>(reg)
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const i64) -> __m512i {
            _mm512_loadu_epi64(data as *const i64)
        }

        #[inline(always)]
        unsafe fn _mm_set1(a: i64) -> __m512i {
            _mm512_set1_epi64(a)
        }

        #[inline(always)]
        unsafe fn _mm_add(a: __m512i, b: __m512i) -> __m512i {
            _mm512_add_epi64(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: __m512i, b: __m512i) -> u8 {
            _mm512_cmpgt_epi64_mask(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: __m512i, b: __m512i) -> u8 {
            _mm512_cmpgt_epi64_mask(b, a)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: __m512i, b: __m512i, mask: u8) -> __m512i {
            _mm512_mask_blend_epi64(mask, a, b)
        }

        // ------------------------------------ ARGMINMAX --------------------------------------

        #[target_feature(enable = "avx512f")]
        unsafe fn argminmax(data: &[i64]) -> (usize, usize) {
            Self::_argminmax(data)
        }
    }

    // ------------------------------------ TESTS --------------------------------------

    #[cfg(test)]
    mod tests {
        use super::{AVX512, SIMD};
        use crate::scalar::generic::scalar_argminmax;

        extern crate dev_utils;
        use dev_utils::utils;

        fn get_array_i64(n: usize) -> Vec<i64> {
            utils::get_random_array(n, i64::MIN, i64::MAX)
        }

        #[test]
        fn test_both_versions_return_the_same_results() {
            if !is_x86_feature_detected!("avx512f") {
                return;
            }

            let data: &[i64] = &get_array_i64(1025);
            assert_eq!(data.len() % 4, 1);

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
                std::i64::MIN,
                std::i64::MIN,
                4,
                6,
                9,
                std::i64::MAX,
                22,
                std::i64::MAX,
            ];
            let data: Vec<i64> = data.iter().map(|x| *x).collect();
            let data: &[i64] = &data;

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            assert_eq!(argmin_index, 0);
            assert_eq!(argmax_index, 5);

            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX512::argminmax(data) };
            assert_eq!(argmin_simd_index, 0);
            assert_eq!(argmax_simd_index, 5);
        }

        #[test]
        fn test_many_random_runs() {
            if !is_x86_feature_detected!("avx512f") {
                return;
            }

            for _ in 0..10_000 {
                let data: &[i64] = &get_array_i64(16 * 2 + 1);
                let (argmin_index, argmax_index) = scalar_argminmax(data);
                let (argmin_simd_index, argmax_simd_index) = unsafe { AVX512::argminmax(data) };
                assert_eq!(argmin_index, argmin_simd_index);
                assert_eq!(argmax_index, argmax_simd_index);
            }
        }
    }
}

// ---------------------------------------- NEON -----------------------------------------

// There are no NEON intrinsics for f64, so we need to use the scalar version.
//   although NEON intrinsics exist for i64 and u64, we cannot use them as
//   they there is no 64-bit variant (of any data type) for the following three
//   intrinsics: vadd_, vcgt_, vclt_

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
mod neon {
    use super::super::config::NEON;
    use super::super::generic::unimplement_simd;
    use super::*;

    // We need to (un)implement the SIMD trait for the NEON struct as otherwise the
    // compiler will complain that the trait is not implemented for the struct -
    // even though we are not using the trait for the NEON struct when dealing with
    // > 64 bit data types.
    unimplement_simd!(i64, usize, NEON);
}

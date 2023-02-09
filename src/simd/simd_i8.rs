use super::config::SIMDInstructionSet;
use super::generic::{SIMDArgMinMax, SIMDOps};
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "arm")]
use std::arch::arm::*;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

const MAX_INDEX: usize = i8::MAX as usize;

// ------------------------------------------ AVX2 ------------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx2 {
    use super::super::config::AVX2;
    use super::*;

    const LANE_SIZE: usize = AVX2::LANE_SIZE_8;

    impl SIMDOps<i8, __m256i, __m256i, LANE_SIZE> for AVX2 {
        const INITIAL_INDEX: __m256i = unsafe {
            std::mem::transmute([
                0i8, 1i8, 2i8, 3i8, 4i8, 5i8, 6i8, 7i8, 8i8, 9i8, 10i8, 11i8, 12i8, 13i8, 14i8,
                15i8, 16i8, 17i8, 18i8, 19i8, 20i8, 21i8, 22i8, 23i8, 24i8, 25i8, 26i8, 27i8, 28i8,
                29i8, 30i8, 31i8,
            ])
        };
        const INDEX_INCREMENT: __m256i =
            unsafe { std::mem::transmute([LANE_SIZE as i8; LANE_SIZE]) };
        const MAX_INDEX: usize = MAX_INDEX;

        #[inline(always)]
        unsafe fn _reg_to_arr(reg: __m256i) -> [i8; LANE_SIZE] {
            std::mem::transmute::<__m256i, [i8; LANE_SIZE]>(reg)
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const i8) -> __m256i {
            _mm256_loadu_si256(data as *const __m256i)
        }

        #[inline(always)]
        unsafe fn _mm_add(a: __m256i, b: __m256i) -> __m256i {
            _mm256_add_epi8(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: __m256i, b: __m256i) -> __m256i {
            _mm256_cmpgt_epi8(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: __m256i, b: __m256i) -> __m256i {
            _mm256_cmpgt_epi8(b, a)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: __m256i, b: __m256i, mask: __m256i) -> __m256i {
            _mm256_blendv_epi8(a, b, mask)
        }

        #[inline(always)]
        unsafe fn _horiz_min(index: __m256i, value: __m256i) -> (usize, i8) {
            // 0. Find the minimum value
            let mut vmin: __m256i = value;
            vmin = _mm256_min_epi8(vmin, _mm256_permute2x128_si256(vmin, vmin, 1));
            vmin = _mm256_min_epi8(vmin, _mm256_alignr_epi8(vmin, vmin, 8));
            vmin = _mm256_min_epi8(vmin, _mm256_alignr_epi8(vmin, vmin, 4));
            vmin = _mm256_min_epi8(vmin, _mm256_alignr_epi8(vmin, vmin, 2));
            vmin = _mm256_min_epi8(vmin, _mm256_alignr_epi8(vmin, vmin, 1));
            let min_value: i8 = _mm256_extract_epi8(vmin, 0) as i8;

            // Extract the index of the minimum value
            // 1. Create a mask with the index of the minimum value
            let mask = _mm256_cmpeq_epi8(value, vmin);
            // 2. Blend the mask with the index
            let search_index = _mm256_blendv_epi8(
                _mm256_set1_epi8(i8::MAX), // if mask is 0, use i8::MAX
                index,                     // if mask is 1, use index
                mask,
            );
            // 3. Find the minimum index
            let mut imin: __m256i = search_index;
            imin = _mm256_min_epi8(imin, _mm256_permute2x128_si256(imin, imin, 1));
            imin = _mm256_min_epi8(imin, _mm256_alignr_epi8(imin, imin, 8));
            imin = _mm256_min_epi8(imin, _mm256_alignr_epi8(imin, imin, 4));
            imin = _mm256_min_epi8(imin, _mm256_alignr_epi8(imin, imin, 2));
            imin = _mm256_min_epi8(imin, _mm256_alignr_epi8(imin, imin, 1));
            let min_index: usize = _mm256_extract_epi8(imin, 0) as usize;

            (min_index, min_value)
        }

        #[inline(always)]
        unsafe fn _horiz_max(index: __m256i, value: __m256i) -> (usize, i8) {
            // 0. Find the maximum value
            let mut vmax: __m256i = value;
            vmax = _mm256_max_epi8(vmax, _mm256_permute2x128_si256(vmax, vmax, 1));
            vmax = _mm256_max_epi8(vmax, _mm256_alignr_epi8(vmax, vmax, 8));
            vmax = _mm256_max_epi8(vmax, _mm256_alignr_epi8(vmax, vmax, 4));
            vmax = _mm256_max_epi8(vmax, _mm256_alignr_epi8(vmax, vmax, 2));
            vmax = _mm256_max_epi8(vmax, _mm256_alignr_epi8(vmax, vmax, 1));
            let max_value: i8 = _mm256_extract_epi8(vmax, 0) as i8;

            // Extract the index of the maximum value
            // 1. Create a mask with the index of the maximum value
            let mask = _mm256_cmpeq_epi8(value, vmax);
            // 2. Blend the mask with the index
            let search_index = _mm256_blendv_epi8(
                _mm256_set1_epi8(i8::MAX), // if mask is 0, use i8::MAX
                index,                     // if mask is 1, use index
                mask,
            );
            // 3. Find the maximum index
            let mut imin: __m256i = search_index;
            imin = _mm256_min_epi8(imin, _mm256_permute2x128_si256(imin, imin, 1));
            imin = _mm256_min_epi8(imin, _mm256_alignr_epi8(imin, imin, 8));
            imin = _mm256_min_epi8(imin, _mm256_alignr_epi8(imin, imin, 4));
            imin = _mm256_min_epi8(imin, _mm256_alignr_epi8(imin, imin, 2));
            imin = _mm256_min_epi8(imin, _mm256_alignr_epi8(imin, imin, 1));
            let max_index: usize = _mm256_extract_epi8(imin, 0) as usize;

            (max_index, max_value)
        }
    }

    impl SIMDArgMinMax<i8, __m256i, __m256i, LANE_SIZE> for AVX2 {
        #[target_feature(enable = "avx2")]
        unsafe fn argminmax(data: &[i8]) -> (usize, usize) {
            Self::_argminmax(data)
        }
    }

    // ------------------------------------ TESTS --------------------------------------

    #[cfg(test)]
    mod tests {
        use super::{SIMDArgMinMax, AVX2};
        use crate::scalar::generic::scalar_argminmax;

        extern crate dev_utils;
        use dev_utils::utils;

        fn get_array_i8(n: usize) -> Vec<i8> {
            utils::get_random_array(n, i8::MIN, i8::MAX)
        }

        #[test]
        fn test_both_versions_return_the_same_results() {
            if !is_x86_feature_detected!("avx2") {
                return;
            }

            let data: &[i8] = &get_array_i8(513);
            assert_eq!(data.len() % 16, 1);

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

            let data = [10, std::i8::MIN, 6, 9, 9, 22, std::i8::MAX, 4, std::i8::MAX];
            let data: Vec<i8> = data.iter().map(|x| *x).collect();
            let data: &[i8] = &data;

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            assert_eq!(argmin_index, 1);
            assert_eq!(argmax_index, 6);

            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX2::argminmax(data) };
            assert_eq!(argmin_simd_index, 1);
            assert_eq!(argmax_simd_index, 6);
        }

        #[test]
        fn test_no_overflow() {
            if !is_x86_feature_detected!("avx2") {
                return;
            }

            let n: usize = 1 << 10;
            let data: &[i8] = &get_array_i8(n);

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX2::argminmax(data) };
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }

        #[test]
        fn test_many_random_runs() {
            if !is_x86_feature_detected!("avx2") {
                return;
            }

            for _ in 0..10_000 {
                let data: &[i8] = &get_array_i8(8 * 32 + 1);
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

    const LANE_SIZE: usize = SSE::LANE_SIZE_8;

    impl SIMDOps<i8, __m128i, __m128i, LANE_SIZE> for SSE {
        const INITIAL_INDEX: __m128i = unsafe {
            std::mem::transmute([
                0i8, 1i8, 2i8, 3i8, 4i8, 5i8, 6i8, 7i8, 8i8, 9i8, 10i8, 11i8, 12i8, 13i8, 14i8,
                15i8,
            ])
        };
        const INDEX_INCREMENT: __m128i =
            unsafe { std::mem::transmute([LANE_SIZE as i8; LANE_SIZE]) };
        const MAX_INDEX: usize = MAX_INDEX;

        #[inline(always)]
        unsafe fn _reg_to_arr(reg: __m128i) -> [i8; LANE_SIZE] {
            std::mem::transmute::<__m128i, [i8; LANE_SIZE]>(reg)
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const i8) -> __m128i {
            _mm_loadu_si128(data as *const __m128i)
        }

        #[inline(always)]
        unsafe fn _mm_add(a: __m128i, b: __m128i) -> __m128i {
            _mm_add_epi8(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: __m128i, b: __m128i) -> __m128i {
            _mm_cmpgt_epi8(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: __m128i, b: __m128i) -> __m128i {
            _mm_cmplt_epi8(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: __m128i, b: __m128i, mask: __m128i) -> __m128i {
            _mm_blendv_epi8(a, b, mask)
        }

        #[inline(always)]
        unsafe fn _horiz_min(index: __m128i, value: __m128i) -> (usize, i8) {
            // 0. Find the minimum value
            let mut vmin: __m128i = value;
            vmin = _mm_min_epi8(vmin, _mm_alignr_epi8(vmin, vmin, 8));
            vmin = _mm_min_epi8(vmin, _mm_alignr_epi8(vmin, vmin, 4));
            vmin = _mm_min_epi8(vmin, _mm_alignr_epi8(vmin, vmin, 2));
            vmin = _mm_min_epi8(vmin, _mm_alignr_epi8(vmin, vmin, 1));
            let min_value: i8 = _mm_extract_epi8(vmin, 0) as i8;

            // Extract the index of the minimum value
            // 1. Create a mask with the index of the minimum value
            let mask = _mm_cmpeq_epi8(value, vmin);
            // 2. Blend the mask with the index
            let search_index = _mm_blendv_epi8(
                _mm_set1_epi8(i8::MAX), // if mask is 0, use i8::MAX
                index,                  // if mask is 1, use index
                mask,
            );
            // 3. Find the minimum index
            let mut imin: __m128i = search_index;
            imin = _mm_min_epi8(imin, _mm_alignr_epi8(imin, imin, 8));
            imin = _mm_min_epi8(imin, _mm_alignr_epi8(imin, imin, 4));
            imin = _mm_min_epi8(imin, _mm_alignr_epi8(imin, imin, 2));
            imin = _mm_min_epi8(imin, _mm_alignr_epi8(imin, imin, 1));
            let min_index: usize = _mm_extract_epi8(imin, 0) as usize;

            (min_index, min_value)
        }

        #[inline(always)]
        unsafe fn _horiz_max(index: __m128i, value: __m128i) -> (usize, i8) {
            // 0. Find the maximum value
            let mut vmax: __m128i = value;
            vmax = _mm_max_epi8(vmax, _mm_alignr_epi8(vmax, vmax, 8));
            vmax = _mm_max_epi8(vmax, _mm_alignr_epi8(vmax, vmax, 4));
            vmax = _mm_max_epi8(vmax, _mm_alignr_epi8(vmax, vmax, 2));
            vmax = _mm_max_epi8(vmax, _mm_alignr_epi8(vmax, vmax, 1));
            let max_value: i8 = _mm_extract_epi8(vmax, 0) as i8;

            // Extract the index of the maximum value
            // 1. Create a mask with the index of the maximum value
            let mask = _mm_cmpeq_epi8(value, vmax);
            // 2. Blend the mask with the index
            let search_index = _mm_blendv_epi8(
                _mm_set1_epi8(i8::MAX), // if mask is 0, use i8::MAX
                index,                  // if mask is 1, use index
                mask,
            );
            // 3. Find the minimum index
            let mut imin: __m128i = search_index;
            imin = _mm_min_epi8(imin, _mm_alignr_epi8(imin, imin, 8));
            imin = _mm_min_epi8(imin, _mm_alignr_epi8(imin, imin, 4));
            imin = _mm_min_epi8(imin, _mm_alignr_epi8(imin, imin, 2));
            imin = _mm_min_epi8(imin, _mm_alignr_epi8(imin, imin, 1));
            let max_index: usize = _mm_extract_epi8(imin, 0) as usize;

            (max_index, max_value)
        }
    }

    impl SIMDArgMinMax<i8, __m128i, __m128i, LANE_SIZE> for SSE {
        #[target_feature(enable = "sse4.1")]
        unsafe fn argminmax(data: &[i8]) -> (usize, usize) {
            Self::_argminmax(data)
        }
    }

    // ------------------------------------ TESTS --------------------------------------

    #[cfg(test)]
    mod tests {
        use super::{SIMDArgMinMax, SSE};
        use crate::scalar::generic::scalar_argminmax;

        extern crate dev_utils;
        use dev_utils::utils;

        fn get_array_i8(n: usize) -> Vec<i8> {
            utils::get_random_array(n, i8::MIN, i8::MAX)
        }

        #[test]
        fn test_both_versions_return_the_same_results() {
            let data: &[i8] = &get_array_i8(513);
            assert_eq!(data.len() % 8, 1);

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            let (argmin_simd_index, argmax_simd_index) = unsafe { SSE::argminmax(data) };
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }

        #[test]
        fn test_first_index_is_returned_when_identical_values_found() {
            let data = [10, std::i8::MIN, 6, 9, 9, 22, std::i8::MAX, 4, std::i8::MAX];
            let data: Vec<i8> = data.iter().map(|x| *x).collect();
            let data: &[i8] = &data;

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            assert_eq!(argmin_index, 1);
            assert_eq!(argmax_index, 6);

            let (argmin_simd_index, argmax_simd_index) = unsafe { SSE::argminmax(data) };
            assert_eq!(argmin_simd_index, 1);
            assert_eq!(argmax_simd_index, 6);
        }

        #[test]
        fn test_no_overflow() {
            let n: usize = 1 << 10;
            let data: &[i8] = &get_array_i8(n);

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            let (argmin_simd_index, argmax_simd_index) = unsafe { SSE::argminmax(data) };
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }

        #[test]
        fn test_many_random_runs() {
            for _ in 0..10_000 {
                let data: &[i8] = &get_array_i8(8 * 32 + 1);
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

    const LANE_SIZE: usize = AVX512::LANE_SIZE_8;

    impl SIMDOps<i8, __m512i, u64, LANE_SIZE> for AVX512 {
        const INITIAL_INDEX: __m512i = unsafe {
            std::mem::transmute([
                0i8, 1i8, 2i8, 3i8, 4i8, 5i8, 6i8, 7i8, 8i8, 9i8, 10i8, 11i8, 12i8, 13i8, 14i8,
                15i8, 16i8, 17i8, 18i8, 19i8, 20i8, 21i8, 22i8, 23i8, 24i8, 25i8, 26i8, 27i8, 28i8,
                29i8, 30i8, 31i8, 32i8, 33i8, 34i8, 35i8, 36i8, 37i8, 38i8, 39i8, 40i8, 41i8, 42i8,
                43i8, 44i8, 45i8, 46i8, 47i8, 48i8, 49i8, 50i8, 51i8, 52i8, 53i8, 54i8, 55i8, 56i8,
                57i8, 58i8, 59i8, 60i8, 61i8, 62i8, 63i8,
            ])
        };
        const INDEX_INCREMENT: __m512i =
            unsafe { std::mem::transmute([LANE_SIZE as i8; LANE_SIZE]) };
        const MAX_INDEX: usize = MAX_INDEX;

        #[inline(always)]
        unsafe fn _reg_to_arr(reg: __m512i) -> [i8; LANE_SIZE] {
            std::mem::transmute::<__m512i, [i8; LANE_SIZE]>(reg)
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const i8) -> __m512i {
            _mm512_loadu_epi8(data as *const i8)
        }

        #[inline(always)]
        unsafe fn _mm_add(a: __m512i, b: __m512i) -> __m512i {
            _mm512_add_epi8(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: __m512i, b: __m512i) -> u64 {
            _mm512_cmpgt_epi8_mask(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: __m512i, b: __m512i) -> u64 {
            _mm512_cmplt_epi8_mask(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: __m512i, b: __m512i, mask: u64) -> __m512i {
            _mm512_mask_blend_epi8(mask, a, b)
        }

        #[inline(always)]
        unsafe fn _horiz_min(index: __m512i, value: __m512i) -> (usize, i8) {
            // 0. Find the minimum value
            let mut vmin: __m512i = value;
            vmin = _mm512_min_epi8(vmin, _mm512_alignr_epi32(vmin, vmin, 8));
            vmin = _mm512_min_epi8(vmin, _mm512_alignr_epi32(vmin, vmin, 4));
            vmin = _mm512_min_epi8(vmin, _mm512_alignr_epi8(vmin, vmin, 8));
            vmin = _mm512_min_epi8(vmin, _mm512_alignr_epi8(vmin, vmin, 4));
            vmin = _mm512_min_epi8(vmin, _mm512_alignr_epi8(vmin, vmin, 2));
            vmin = _mm512_min_epi8(vmin, _mm512_alignr_epi8(vmin, vmin, 1));
            let min_value: i8 = _mm_extract_epi8(_mm512_castsi512_si128(vmin), 0) as i8;

            // Extract the index of the minimum value
            // 1. Create a mask with the index of the minimum value
            let mask = _mm512_cmpeq_epi8_mask(value, vmin);
            // 2. Blend the mask with the index
            let search_index = _mm512_mask_blend_epi8(
                mask,
                _mm512_set1_epi8(i8::MAX), // if mask is 0, use i8::MAX
                index,                     // if mask is 1, use index
            );
            // 3. Find the minimum index
            let mut imin: __m512i = search_index;
            imin = _mm512_min_epi8(imin, _mm512_alignr_epi32(imin, imin, 8));
            imin = _mm512_min_epi8(imin, _mm512_alignr_epi32(imin, imin, 4));
            imin = _mm512_min_epi8(imin, _mm512_alignr_epi8(imin, imin, 8));
            imin = _mm512_min_epi8(imin, _mm512_alignr_epi8(imin, imin, 4));
            imin = _mm512_min_epi8(imin, _mm512_alignr_epi8(imin, imin, 2));
            imin = _mm512_min_epi8(imin, _mm512_alignr_epi8(imin, imin, 1));
            let min_index: usize = _mm_extract_epi8(_mm512_castsi512_si128(imin), 0) as usize;

            (min_index, min_value)
        }

        #[inline(always)]
        unsafe fn _horiz_max(index: __m512i, value: __m512i) -> (usize, i8) {
            // 0. Find the maximum value
            let mut vmax: __m512i = value;
            vmax = _mm512_max_epi8(vmax, _mm512_alignr_epi32(vmax, vmax, 8));
            vmax = _mm512_max_epi8(vmax, _mm512_alignr_epi32(vmax, vmax, 4));
            vmax = _mm512_max_epi8(vmax, _mm512_alignr_epi8(vmax, vmax, 8));
            vmax = _mm512_max_epi8(vmax, _mm512_alignr_epi8(vmax, vmax, 4));
            vmax = _mm512_max_epi8(vmax, _mm512_alignr_epi8(vmax, vmax, 2));
            vmax = _mm512_max_epi8(vmax, _mm512_alignr_epi8(vmax, vmax, 1));
            let max_value: i8 = _mm_extract_epi8(_mm512_castsi512_si128(vmax), 0) as i8;

            // Extract the index of the maximum value
            // 1. Create a mask with the index of the maximum value
            let mask = _mm512_cmpeq_epi8_mask(value, vmax);
            // 2. Blend the mask with the index
            let search_index = _mm512_mask_blend_epi8(
                mask,
                _mm512_set1_epi8(i8::MAX), // if mask is 0, use i8::MAX
                index,                     // if mask is 1, use index
            );
            // 3. Find the maximum index
            let mut imin: __m512i = search_index;
            imin = _mm512_min_epi8(imin, _mm512_alignr_epi32(imin, imin, 8));
            imin = _mm512_min_epi8(imin, _mm512_alignr_epi32(imin, imin, 4));
            imin = _mm512_min_epi8(imin, _mm512_alignr_epi8(imin, imin, 8));
            imin = _mm512_min_epi8(imin, _mm512_alignr_epi8(imin, imin, 4));
            imin = _mm512_min_epi8(imin, _mm512_alignr_epi8(imin, imin, 2));
            imin = _mm512_min_epi8(imin, _mm512_alignr_epi8(imin, imin, 1));
            let max_index: usize = _mm_extract_epi8(_mm512_castsi512_si128(imin), 0) as usize;

            (max_index, max_value)
        }
    }

    impl SIMDArgMinMax<i8, __m512i, u64, LANE_SIZE> for AVX512 {
        #[target_feature(enable = "avx512bw")]
        unsafe fn argminmax(data: &[i8]) -> (usize, usize) {
            Self::_argminmax(data)
        }
    }

    // ------------------------------------ TESTS --------------------------------------

    #[cfg(test)]
    mod tests {
        use super::{SIMDArgMinMax, AVX512};
        use crate::scalar::generic::scalar_argminmax;

        extern crate dev_utils;
        use dev_utils::utils;

        fn get_array_i8(n: usize) -> Vec<i8> {
            utils::get_random_array(n, i8::MIN, i8::MAX)
        }

        #[test]
        fn test_both_versions_return_the_same_results() {
            if !is_x86_feature_detected!("avx512bw") {
                return;
            }

            let data: &[i8] = &get_array_i8(513);
            assert_eq!(data.len() % 8, 1);

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX512::argminmax(data) };
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }

        #[test]
        fn test_first_index_is_returned_when_identical_values_found() {
            if !is_x86_feature_detected!("avx512bw") {
                return;
            }

            let data = [10, std::i8::MIN, 6, 9, 9, 22, std::i8::MAX, 4, std::i8::MAX];
            let data: Vec<i8> = data.iter().map(|x| *x).collect();
            let data: &[i8] = &data;

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            assert_eq!(argmin_index, 1);
            assert_eq!(argmax_index, 6);

            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX512::argminmax(data) };
            assert_eq!(argmin_simd_index, 1);
            assert_eq!(argmax_simd_index, 6);
        }

        #[test]
        fn test_no_overflow() {
            if !is_x86_feature_detected!("avx512bw") {
                return;
            }

            let n: usize = 1 << 10;
            let data: &[i8] = &get_array_i8(n);

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX512::argminmax(data) };
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }

        #[test]
        fn test_many_random_runs() {
            if !is_x86_feature_detected!("avx512bw") {
                return;
            }

            for _ in 0..10_000 {
                let data: &[i8] = &get_array_i8(1025);
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
    use super::super::config::NEON;
    use super::*;

    const LANE_SIZE: usize = NEON::LANE_SIZE_8;

    impl SIMD<i8, int8x16_t, uint8x16_t, LANE_SIZE> for NEON {
        const INITIAL_INDEX: int8x16_t = unsafe {
            std::mem::transmute([
                0i8, 1i8, 2i8, 3i8, 4i8, 5i8, 6i8, 7i8, 8i8, 9i8, 10i8, 11i8, 12i8, 13i8, 14i8,
                15i8,
            ])
        };
        const MAX_INDEX: usize = i8::MAX as usize;

        #[inline(always)]
        unsafe fn _reg_to_arr(reg: int8x16_t) -> [i8; LANE_SIZE] {
            std::mem::transmute::<int8x16_t, [i8; LANE_SIZE]>(reg)
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const i8) -> int8x16_t {
            // TODO: requires v7
            vld1q_s8(data as *const i8)
        }

        #[inline(always)]
        unsafe fn _mm_set1(a: usize) -> int8x16_t {
            vdupq_n_s8(a as i8)
        }

        #[inline(always)]
        unsafe fn _mm_add(a: int8x16_t, b: int8x16_t) -> int8x16_t {
            vaddq_s8(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: int8x16_t, b: int8x16_t) -> uint8x16_t {
            vcgtq_s8(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: int8x16_t, b: int8x16_t) -> uint8x16_t {
            vcltq_s8(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: int8x16_t, b: int8x16_t, mask: uint8x16_t) -> int8x16_t {
            vbslq_s8(mask, b, a)
        }

        // ------------------------------------ ARGMINMAX --------------------------------------

        #[target_feature(enable = "neon")]
        unsafe fn argminmax(data: &[i8]) -> (usize, usize) {
            Self::_argminmax(data)
        }

        #[inline(always)]
        unsafe fn _horiz_min(index: int8x16_t, value: int8x16_t) -> (usize, i8) {
            // 0. Find the minimum value
            let mut vmin: int8x16_t = value;
            vmin = vminq_s8(vmin, vextq_s8(vmin, vmin, 8));
            vmin = vminq_s8(vmin, vextq_s8(vmin, vmin, 4));
            vmin = vminq_s8(vmin, vextq_s8(vmin, vmin, 2));
            vmin = vminq_s8(vmin, vextq_s8(vmin, vmin, 1));
            let min_value: i8 = vgetq_lane_s8(vmin, 0);

            // Extract the index of the minimum value
            // 1. Create a mask with the index of the minimum value
            let mask = vceqq_s8(value, vmin);
            // 2. Blend the mask with the index
            let search_index = vbslq_s8(
                mask,
                index,               // if mask is 1, use index
                vdupq_n_s8(i8::MAX), // if mask is 0, use i8::MAX
            );
            // 3. Find the minimum index
            let mut imin: int8x16_t = search_index;
            imin = vminq_s8(imin, vextq_s8(imin, imin, 8));
            imin = vminq_s8(imin, vextq_s8(imin, imin, 4));
            imin = vminq_s8(imin, vextq_s8(imin, imin, 2));
            imin = vminq_s8(imin, vextq_s8(imin, imin, 1));
            let min_index: usize = vgetq_lane_s8(imin, 0) as usize;

            (min_index, min_value)
        }

        #[inline(always)]
        unsafe fn _horiz_max(index: int8x16_t, value: int8x16_t) -> (usize, i8) {
            // 0. Find the maximum value
            let mut vmax: int8x16_t = value;
            vmax = vmaxq_s8(vmax, vextq_s8(vmax, vmax, 8));
            vmax = vmaxq_s8(vmax, vextq_s8(vmax, vmax, 4));
            vmax = vmaxq_s8(vmax, vextq_s8(vmax, vmax, 2));
            vmax = vmaxq_s8(vmax, vextq_s8(vmax, vmax, 1));
            let max_value: i8 = vgetq_lane_s8(vmax, 0);

            // Extract the index of the maximum value
            // 1. Create a mask with the index of the maximum value
            let mask = vceqq_s8(value, vmax);
            // 2. Blend the mask with the index
            let search_index = vbslq_s8(
                mask,
                index,               // if mask is 1, use index
                vdupq_n_s8(i8::MAX), // if mask is 0, use i8::MAX
            );
            // 3. Find the maximum index
            let mut imin: int8x16_t = search_index;
            imin = vminq_s8(imin, vextq_s8(imin, imin, 8));
            imin = vminq_s8(imin, vextq_s8(imin, imin, 4));
            imin = vminq_s8(imin, vextq_s8(imin, imin, 2));
            imin = vminq_s8(imin, vextq_s8(imin, imin, 1));
            let max_index: usize = vgetq_lane_s8(imin, 0) as usize;

            (max_index, max_value)
        }
    }

    // ------------------------------------ TESTS --------------------------------------

    #[cfg(test)]
    mod tests {
        use super::{NEON, SIMD};
        use crate::scalar::generic::scalar_argminmax;

        extern crate dev_utils;
        use dev_utils::utils;

        fn get_array_i8(n: usize) -> Vec<i8> {
            utils::get_random_array(n, i8::MIN, i8::MAX)
        }

        #[test]
        fn test_both_versions_return_the_same_results() {
            let data: &[i8] = &get_array_i8(513);
            assert_eq!(data.len() % 8, 1);

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            let (argmin_simd_index, argmax_simd_index) = unsafe { NEON::argminmax(data) };
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }

        #[test]
        fn test_first_index_is_returned_when_identical_values_found() {
            let data = [10, std::i8::MIN, 6, 9, 9, 22, std::i8::MAX, 4, std::i8::MAX];
            let data: Vec<i8> = data.iter().map(|x| *x).collect();
            let data: &[i8] = &data;

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            assert_eq!(argmin_index, 1);
            assert_eq!(argmax_index, 6);

            let (argmin_simd_index, argmax_simd_index) = unsafe { NEON::argminmax(data) };
            assert_eq!(argmin_simd_index, 1);
            assert_eq!(argmax_simd_index, 6);
        }

        #[test]
        fn test_no_overflow() {
            let n: usize = 1 << 10;
            let data: &[i8] = &get_array_i8(n);

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            let (argmin_simd_index, argmax_simd_index) = unsafe { NEON::argminmax(data) };
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }

        #[test]
        fn test_many_random_runs() {
            for _ in 0..10_000 {
                let data: &[i8] = &get_array_i8(16 * 8 + 1);
                let (argmin_index, argmax_index) = scalar_argminmax(data);
                let (argmin_simd_index, argmax_simd_index) = unsafe { NEON::argminmax(data) };
                assert_eq!(argmin_index, argmin_simd_index);
                assert_eq!(argmax_index, argmax_simd_index);
            }
        }
    }
}

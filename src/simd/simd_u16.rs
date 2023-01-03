use super::config::SIMDInstructionSet;
use super::generic::SIMD;
use ndarray::ArrayView1;
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "arm")]
use std::arch::arm::*;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const XOR_VALUE: i16 = 0x7FFF;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
fn _i16decrord_to_u16(decrord_i16: i16) -> u16 {
    // let v = ord_i16 ^ 0x7FFF;
    unsafe { std::mem::transmute::<i16, u16>(decrord_i16 ^ XOR_VALUE) }
}

// ------------------------------------------ AVX2 ------------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx2 {
    use super::super::config::AVX2;
    use super::*;

    const LANE_SIZE: usize = AVX2::LANE_SIZE_16;
    const XOR_MASK: __m256i = unsafe { std::mem::transmute([XOR_VALUE; LANE_SIZE]) };

    #[inline(always)]
    unsafe fn _u16_to_i16decrord(u16: __m256i) -> __m256i {
        // on a scalar: v^ 0x7FFF
        // transforms to monotonically **decreasing** order
        _mm256_xor_si256(u16, XOR_MASK) // Only 1 operation
    }

    #[inline(always)]
    unsafe fn _reg_to_i16_arr(reg: __m256i) -> [i16; LANE_SIZE] {
        std::mem::transmute::<__m256i, [i16; LANE_SIZE]>(reg)
    }

    impl SIMD<u16, __m256i, __m256i, LANE_SIZE> for AVX2 {
        const INITIAL_INDEX: __m256i = unsafe {
            std::mem::transmute([
                0i16, 1i16, 2i16, 3i16, 4i16, 5i16, 6i16, 7i16, 8i16, 9i16, 10i16, 11i16, 12i16,
                13i16, 14i16, 15i16,
            ])
        };
        const MAX_INDEX: usize = i16::MAX as usize;

        #[inline(always)]
        unsafe fn _reg_to_arr(_: __m256i) -> [u16; LANE_SIZE] {
            // Not used because we work with i16ord and override _get_min_index_value and _get_max_index_value
            unimplemented!()
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const u16) -> __m256i {
            _u16_to_i16decrord(_mm256_loadu_si256(data as *const __m256i))
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
        unsafe fn argminmax(data: ArrayView1<u16>) -> (usize, usize) {
            Self::_argminmax(data)
        }

        #[inline(always)]
        unsafe fn _horiz_min(index: __m256i, value: __m256i) -> (usize, u16) {
            // 0. Find the minimum value
            let mut vmin: __m256i = value;
            vmin = _mm256_min_epi16(vmin, _mm256_permute2x128_si256(vmin, vmin, 1));
            vmin = _mm256_min_epi16(vmin, _mm256_alignr_epi8(vmin, vmin, 8));
            vmin = _mm256_min_epi16(vmin, _mm256_alignr_epi8(vmin, vmin, 4));
            vmin = _mm256_min_epi16(vmin, _mm256_alignr_epi8(vmin, vmin, 2));
            let min_value: i16 = _mm256_extract_epi16(vmin, 0) as i16;

            // Extract the index of the minimum value
            // 1. Create a mask with the index of the minimum value
            let mask = _mm256_cmpeq_epi16(value, vmin);
            // 2. Blend the mask with the index
            let search_index = _mm256_blendv_epi8(
                _mm256_set1_epi16(i16::MAX), // if mask is 0, use i16::MAX
                index,                       // if mask is 1, use index
                mask,
            );
            // 3. Find the minimum index
            let mut imin: __m256i = search_index;
            imin = _mm256_min_epi16(imin, _mm256_permute2x128_si256(imin, imin, 1));
            imin = _mm256_min_epi16(imin, _mm256_alignr_epi8(imin, imin, 8));
            imin = _mm256_min_epi16(imin, _mm256_alignr_epi8(imin, imin, 4));
            imin = _mm256_min_epi16(imin, _mm256_alignr_epi8(imin, imin, 2));
            let min_index: usize = _mm256_extract_epi16(imin, 0) as usize;

            (min_index, _i16decrord_to_u16(min_value))
        }

        #[inline(always)]
        unsafe fn _horiz_max(index: __m256i, value: __m256i) -> (usize, u16) {
            // 0. Find the maximum value
            let mut vmax: __m256i = value;
            vmax = _mm256_max_epi16(vmax, _mm256_permute2x128_si256(vmax, vmax, 1));
            vmax = _mm256_max_epi16(vmax, _mm256_alignr_epi8(vmax, vmax, 8));
            vmax = _mm256_max_epi16(vmax, _mm256_alignr_epi8(vmax, vmax, 4));
            vmax = _mm256_max_epi16(vmax, _mm256_alignr_epi8(vmax, vmax, 2));
            let max_value: i16 = _mm256_extract_epi16(vmax, 0) as i16;

            // Extract the index of the maximum value
            // 1. Create a mask with the index of the maximum value
            let mask = _mm256_cmpeq_epi16(value, vmax);
            // 2. Blend the mask with the index
            let search_index = _mm256_blendv_epi8(
                _mm256_set1_epi16(i16::MAX), // if mask is 0, use i16::MAX
                index,                       // if mask is 1, use index
                mask,
            );
            // 3. Find the maximum index
            let mut imin: __m256i = search_index;
            imin = _mm256_min_epi16(imin, _mm256_permute2x128_si256(imin, imin, 1));
            imin = _mm256_min_epi16(imin, _mm256_alignr_epi8(imin, imin, 8));
            imin = _mm256_min_epi16(imin, _mm256_alignr_epi8(imin, imin, 4));
            imin = _mm256_min_epi16(imin, _mm256_alignr_epi8(imin, imin, 2));
            let max_index: usize = _mm256_extract_epi16(imin, 0) as usize;

            (max_index, _i16decrord_to_u16(max_value))
        }

        #[inline(always)]
        unsafe fn _get_min_max_index_value(
            index_low: __m256i,
            values_low: __m256i,
            index_high: __m256i,
            values_high: __m256i,
        ) -> (usize, u16, usize, u16) {
            let (min_index, min_value) = Self::_horiz_min(index_low, values_low);
            let (max_index, max_value) = Self::_horiz_max(index_high, values_high);
            // Swap min and max here because we worked with i16ord in decreasing order (max => actual min, and vice versa)
            (max_index, max_value, min_index, min_value)
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

        fn get_array_u16(n: usize) -> Array1<u16> {
            utils::get_random_array(n, u16::MIN, u16::MAX)
        }

        #[test]
        fn test_both_versions_return_the_same_results() {
            if !is_x86_feature_detected!("avx2") {
                return;
            }

            let data = get_array_u16(513);
            assert_eq!(data.len() % 16, 1);

            let (argmin_index, argmax_index) = scalar_argminmax(data.view());
            let (simd_argmin_index, simd_argmax_index) = unsafe { AVX2::argminmax(data.view()) };
            assert_eq!(argmin_index, simd_argmin_index);
            assert_eq!(argmax_index, simd_argmax_index);
        }

        #[test]
        fn test_first_index_is_returned_when_identical_values_found() {
            if !is_x86_feature_detected!("avx2") {
                return;
            }

            let data = [
                10,
                std::u16::MIN,
                6,
                9,
                9,
                22,
                std::u16::MAX,
                4,
                std::u16::MAX,
            ];
            let data: Vec<u16> = data.iter().map(|x| *x).collect();
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
            if !is_x86_feature_detected!("avx2") {
                return;
            }

            let n: usize = 1 << 18;
            let data = get_array_u16(n);

            let (argmin_index, argmax_index) = scalar_argminmax(data.view());
            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX2::argminmax(data.view()) };
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }

        #[test]
        fn test_many_random_runs() {
            if !is_x86_feature_detected!("avx2") {
                return;
            }

            for _ in 0..10_000 {
                let data = get_array_u16(32 * 2 + 1);
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
    const XOR_MASK: __m128i = unsafe { std::mem::transmute([XOR_VALUE; LANE_SIZE]) };

    #[inline(always)]
    unsafe fn _u16_to_i16decrord(u16: __m128i) -> __m128i {
        // on a scalar: v^ 0x7F
        // transforms to monotonically **decreasing** order
        _mm_xor_si128(u16, XOR_MASK)
    }

    #[inline(always)]
    unsafe fn _reg_to_i16_arr(reg: __m128i) -> [i16; LANE_SIZE] {
        std::mem::transmute::<__m128i, [i16; LANE_SIZE]>(reg)
    }

    impl SIMD<u16, __m128i, __m128i, LANE_SIZE> for SSE {
        const INITIAL_INDEX: __m128i =
            unsafe { std::mem::transmute([0i16, 1i16, 2i16, 3i16, 4i16, 5i16, 6i16, 7i16]) };
        const MAX_INDEX: usize = i16::MAX as usize;

        #[inline(always)]
        unsafe fn _reg_to_arr(_: __m128i) -> [u16; LANE_SIZE] {
            // Not used because we work with i16ord and override _get_min_index_value and _get_max_index_value
            unimplemented!()
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const u16) -> __m128i {
            _u16_to_i16decrord(_mm_loadu_si128(data as *const __m128i))
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
        unsafe fn argminmax(data: ArrayView1<u16>) -> (usize, usize) {
            Self::_argminmax(data)
        }

        #[inline(always)]
        unsafe fn _horiz_min(index: __m128i, value: __m128i) -> (usize, u16) {
            // 0. Find the minimum value
            let mut vmin: __m128i = value;
            vmin = _mm_min_epi16(vmin, _mm_alignr_epi8(vmin, vmin, 8));
            vmin = _mm_min_epi16(vmin, _mm_alignr_epi8(vmin, vmin, 4));
            vmin = _mm_min_epi16(vmin, _mm_alignr_epi8(vmin, vmin, 2));
            let min_value: i16 = _mm_extract_epi16(vmin, 0) as i16;

            // Extract the index of the minimum value
            // 1. Create a mask with the index of the minimum value
            let mask = _mm_cmpeq_epi16(value, vmin);
            // 2. Blend the mask with the index
            let search_index = _mm_blendv_epi8(
                _mm_set1_epi16(i16::MAX), // if mask is 0, use i16::MAX
                index,                    // if mask is 1, use index
                mask,
            );
            // 3. Find the minimum index
            let mut imin: __m128i = search_index;
            imin = _mm_min_epi16(imin, _mm_alignr_epi8(imin, imin, 8));
            imin = _mm_min_epi16(imin, _mm_alignr_epi8(imin, imin, 4));
            imin = _mm_min_epi16(imin, _mm_alignr_epi8(imin, imin, 2));
            let min_index: usize = _mm_extract_epi16(imin, 0) as usize;

            (min_index, _i16decrord_to_u16(min_value))
        }

        #[inline(always)]
        unsafe fn _horiz_max(index: __m128i, value: __m128i) -> (usize, u16) {
            // 0. Find the maximum value
            let mut vmax: __m128i = value;
            vmax = _mm_max_epi16(vmax, _mm_alignr_epi8(vmax, vmax, 8));
            vmax = _mm_max_epi16(vmax, _mm_alignr_epi8(vmax, vmax, 4));
            vmax = _mm_max_epi16(vmax, _mm_alignr_epi8(vmax, vmax, 2));
            let max_value: i16 = _mm_extract_epi16(vmax, 0) as i16;

            // Extract the index of the maximum value
            // 1. Create a mask with the index of the maximum value
            let mask = _mm_cmpeq_epi16(value, vmax);
            // 2. Blend the mask with the index
            let search_index = _mm_blendv_epi8(
                _mm_set1_epi16(i16::MAX), // if mask is 0, use i8::MAX
                index,                    // if mask is 1, use index
                mask,
            );
            // 3. Find the maximum index
            let mut imin: __m128i = search_index;
            imin = _mm_min_epi16(imin, _mm_alignr_epi8(imin, imin, 8));
            imin = _mm_min_epi16(imin, _mm_alignr_epi8(imin, imin, 4));
            imin = _mm_min_epi16(imin, _mm_alignr_epi8(imin, imin, 2));
            let max_index: usize = _mm_extract_epi16(imin, 0) as usize;

            (max_index, _i16decrord_to_u16(max_value))
        }

        #[inline(always)]
        unsafe fn _get_min_max_index_value(
            index_low: __m128i,
            values_low: __m128i,
            index_high: __m128i,
            values_high: __m128i,
        ) -> (usize, u16, usize, u16) {
            let (min_index, min_value) = Self::_horiz_min(index_low, values_low);
            let (max_index, max_value) = Self::_horiz_max(index_high, values_high);
            // Swap min and max here because we worked with i16ord in decreasing order (max => actual min, and vice versa)
            (max_index, max_value, min_index, min_value)
        }
    }

    // ----------------------------------------- TESTS -----------------------------------------

    #[cfg(test)]
    mod tests {
        use super::{SIMD, SSE};
        use crate::scalar::generic::scalar_argminmax;

        use ndarray::Array1;

        extern crate dev_utils;
        use dev_utils::utils;

        fn get_array_u16(n: usize) -> Array1<u16> {
            utils::get_random_array(n, u16::MIN, u16::MAX)
        }

        #[test]
        fn test_both_versions_return_the_same_results() {
            let data = get_array_u16(513);
            assert_eq!(data.len() % 16, 1);

            let (argmin_index, argmax_index) = scalar_argminmax(data.view());
            let (simd_argmin_index, simd_argmax_index) = unsafe { SSE::argminmax(data.view()) };
            assert_eq!(argmin_index, simd_argmin_index);
            assert_eq!(argmax_index, simd_argmax_index);
        }

        #[test]
        fn test_first_index_is_returned_when_identical_values_found() {
            let data = [
                10,
                std::u16::MIN,
                6,
                9,
                9,
                22,
                std::u16::MAX,
                4,
                std::u16::MAX,
            ];
            let data: Vec<u16> = data.iter().map(|x| *x).collect();
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
            let data = get_array_u16(n);

            let (argmin_index, argmax_index) = scalar_argminmax(data.view());
            let (argmin_simd_index, argmax_simd_index) = unsafe { SSE::argminmax(data.view()) };
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }

        #[test]
        fn test_many_random_runs() {
            for _ in 0..10_000 {
                let data = get_array_u16(32 * 2 + 1);
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

    const XOR_MASK: __m512i = unsafe { std::mem::transmute([XOR_VALUE; LANE_SIZE]) };

    //  - comparison swappen => dan moeten we opt einde niet meer swappen?

    #[inline(always)]
    unsafe fn _u16_to_i16decrord(u16: __m512i) -> __m512i {
        // on a scalar: v^ 0x7FFF
        // transforms to monotonically **decreasing** order
        _mm512_xor_si512(u16, XOR_MASK)
    }

    #[inline(always)]
    unsafe fn _reg_to_i16_arr(reg: __m512i) -> [i16; LANE_SIZE] {
        std::mem::transmute::<__m512i, [i16; LANE_SIZE]>(reg)
    }

    impl SIMD<u16, __m512i, u32, LANE_SIZE> for AVX512 {
        const INITIAL_INDEX: __m512i = unsafe {
            std::mem::transmute([
                0i16, 1i16, 2i16, 3i16, 4i16, 5i16, 6i16, 7i16, 8i16, 9i16, 10i16, 11i16, 12i16,
                13i16, 14i16, 15i16, 16i16, 17i16, 18i16, 19i16, 20i16, 21i16, 22i16, 23i16, 24i16,
                25i16, 26i16, 27i16, 28i16, 29i16, 30i16, 31i16,
            ])
        };
        const MAX_INDEX: usize = i16::MAX as usize;

        #[inline(always)]
        unsafe fn _reg_to_arr(_: __m512i) -> [u16; LANE_SIZE] {
            unimplemented!("We work with decrordi16 and override _get_min_index_value and _get_max_index_value")
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const u16) -> __m512i {
            _u16_to_i16decrord(_mm512_loadu_epi16(data as *const i16))
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
        unsafe fn argminmax(data: ArrayView1<u16>) -> (usize, usize) {
            Self::_argminmax(data)
        }

        #[inline(always)]
        unsafe fn _horiz_min(index: __m512i, value: __m512i) -> (usize, u16) {
            // 0. Find the minimum value
            let mut vmin: __m512i = value;
            vmin = _mm512_min_epi16(vmin, _mm512_alignr_epi32(vmin, vmin, 8));
            vmin = _mm512_min_epi16(vmin, _mm512_alignr_epi32(vmin, vmin, 4));
            vmin = _mm512_min_epi16(vmin, _mm512_alignr_epi8(vmin, vmin, 8));
            vmin = _mm512_min_epi16(vmin, _mm512_alignr_epi8(vmin, vmin, 4));
            vmin = _mm512_min_epi16(vmin, _mm512_alignr_epi8(vmin, vmin, 2));
            let min_value: i16 = _mm_extract_epi16(_mm512_castsi512_si128(vmin), 0) as i16;

            // Extract the index of the minimum value
            // 1. Create a mask with the index of the minimum value
            let mask = _mm512_cmpeq_epi16_mask(value, vmin);
            // 2. Blend the mask with the index
            let search_index = _mm512_mask_blend_epi16(
                mask,
                _mm512_set1_epi16(i16::MAX), // if mask is 0, use i16::MAX
                index,                       // if mask is 1, use index
            );
            // 3. Find the minimum index
            let mut imin: __m512i = search_index;
            imin = _mm512_min_epi16(imin, _mm512_alignr_epi32(imin, imin, 8));
            imin = _mm512_min_epi16(imin, _mm512_alignr_epi32(imin, imin, 4));
            imin = _mm512_min_epi16(imin, _mm512_alignr_epi8(imin, imin, 8));
            imin = _mm512_min_epi16(imin, _mm512_alignr_epi8(imin, imin, 4));
            imin = _mm512_min_epi16(imin, _mm512_alignr_epi8(imin, imin, 2));
            let min_index: usize = _mm_extract_epi16(_mm512_castsi512_si128(imin), 0) as usize;

            (min_index, _i16decrord_to_u16(min_value))
        }

        #[inline(always)]
        unsafe fn _horiz_max(index: __m512i, value: __m512i) -> (usize, u16) {
            // 0. Find the maximum value
            let mut vmax: __m512i = value;
            vmax = _mm512_max_epi16(vmax, _mm512_alignr_epi32(vmax, vmax, 8));
            vmax = _mm512_max_epi16(vmax, _mm512_alignr_epi32(vmax, vmax, 4));
            vmax = _mm512_max_epi16(vmax, _mm512_alignr_epi8(vmax, vmax, 8));
            vmax = _mm512_max_epi16(vmax, _mm512_alignr_epi8(vmax, vmax, 4));
            vmax = _mm512_max_epi16(vmax, _mm512_alignr_epi8(vmax, vmax, 2));
            let max_value: i16 = _mm_extract_epi16(_mm512_castsi512_si128(vmax), 0) as i16;

            // Extract the index of the maximum value
            // 1. Create a mask with the index of the maximum value
            let mask = _mm512_cmpeq_epi16_mask(value, vmax);
            // 2. Blend the mask with the index
            let search_index = _mm512_mask_blend_epi16(
                mask,
                _mm512_set1_epi16(i16::MAX), // if mask is 0, use i16::MAX
                index,                       // if mask is 1, use index
            );
            // 3. Find the maximum index
            let mut imin: __m512i = search_index;
            imin = _mm512_min_epi16(imin, _mm512_alignr_epi32(imin, imin, 8));
            imin = _mm512_min_epi16(imin, _mm512_alignr_epi32(imin, imin, 4));
            imin = _mm512_min_epi16(imin, _mm512_alignr_epi8(imin, imin, 8));
            imin = _mm512_min_epi16(imin, _mm512_alignr_epi8(imin, imin, 4));
            imin = _mm512_min_epi16(imin, _mm512_alignr_epi8(imin, imin, 2));
            let max_index: usize = _mm_extract_epi16(_mm512_castsi512_si128(imin), 0) as usize;

            (max_index, _i16decrord_to_u16(max_value))
        }

        #[inline(always)]
        unsafe fn _get_min_max_index_value(
            index_low: __m512i,
            values_low: __m512i,
            index_high: __m512i,
            values_high: __m512i,
        ) -> (usize, u16, usize, u16) {
            let (min_index, min_value) = Self::_horiz_min(index_low, values_low);
            let (max_index, max_value) = Self::_horiz_max(index_high, values_high);
            // Swap min and max here because we worked with i16ord in decreasing order (max => actual min, and vice versa)
            (max_index, max_value, min_index, min_value)
        }
    }

    // ----------------------------------------- TESTS -----------------------------------------

    #[cfg(test)]

    mod tests {
        use super::{AVX512, SIMD};
        use crate::scalar::generic::scalar_argminmax;

        use ndarray::Array1;

        extern crate dev_utils;
        use dev_utils::utils;

        fn get_array_u16(n: usize) -> Array1<u16> {
            utils::get_random_array(n, u16::MIN, u16::MAX)
        }

        #[test]
        fn test_both_versions_return_the_same_results() {
            if !is_x86_feature_detected!("avx512f") {
                return;
            }

            let data = get_array_u16(513);
            assert_eq!(data.len() % 16, 1);

            let (argmin_index, argmax_index) = scalar_argminmax(data.view());
            let (simd_argmin_index, simd_argmax_index) = unsafe { AVX512::argminmax(data.view()) };
            assert_eq!(argmin_index, simd_argmin_index);
            assert_eq!(argmax_index, simd_argmax_index);
        }

        #[test]
        fn test_first_index_is_returned_when_identical_values_found() {
            if !is_x86_feature_detected!("avx512f") {
                return;
            }

            let data = [
                10,
                std::u16::MIN,
                6,
                9,
                9,
                22,
                std::u16::MAX,
                4,
                std::u16::MAX,
            ];
            let data: Vec<u16> = data.iter().map(|x| *x).collect();
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
            if !is_x86_feature_detected!("avx512f") {
                return;
            }

            let n: usize = 1 << 18;
            let data = get_array_u16(n);

            let (argmin_index, argmax_index) = scalar_argminmax(data.view());
            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX512::argminmax(data.view()) };
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }

        #[test]
        fn test_many_random_runs() {
            if !is_x86_feature_detected!("avx512f") {
                return;
            }

            for _ in 0..10_000 {
                let data = get_array_u16(32 * 4 + 1);
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

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
mod neon {
    use super::super::config::NEON;
    use super::*;

    const LANE_SIZE: usize = NEON::LANE_SIZE_16;

    impl SIMD<u16, uint16x8_t, uint16x8_t, LANE_SIZE> for NEON {
        const INITIAL_INDEX: uint16x8_t =
            unsafe { std::mem::transmute([0i16, 1i16, 2i16, 3i16, 4i16, 5i16, 6i16, 7i16]) };
        const MAX_INDEX: usize = u16::MAX as usize;

        #[inline(always)]
        unsafe fn _reg_to_arr(reg: uint16x8_t) -> [u16; LANE_SIZE] {
            std::mem::transmute::<uint16x8_t, [u16; LANE_SIZE]>(reg)
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const u16) -> uint16x8_t {
            vld1q_u16(data as *const u16)
        }

        #[inline(always)]
        unsafe fn _mm_set1(a: usize) -> uint16x8_t {
            vdupq_n_u16(a as u16)
        }

        #[inline(always)]
        unsafe fn _mm_add(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
            vaddq_u16(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
            vcgtq_u16(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
            vcltq_u16(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: uint16x8_t, b: uint16x8_t, mask: uint16x8_t) -> uint16x8_t {
            vbslq_u16(mask, b, a)
        }

        // ------------------------------------ ARGMINMAX --------------------------------------

        #[target_feature(enable = "neon")]
        unsafe fn argminmax(data: ArrayView1<u16>) -> (usize, usize) {
            Self::_argminmax(data)
        }
    }

    // ----------------------------------------- TESTS -----------------------------------------

    #[cfg(test)]
    mod tests {
        use super::{NEON, SIMD};
        use crate::scalar::generic::scalar_argminmax;

        use ndarray::Array1;

        extern crate dev_utils;
        use dev_utils::utils;

        fn get_array_u16(n: usize) -> Array1<u16> {
            utils::get_random_array(n, u16::MIN, u16::MAX)
        }

        #[test]
        fn test_both_versions_return_the_same_results() {
            let data = get_array_u16(513);
            assert_eq!(data.len() % 16, 1);

            let (argmin_index, argmax_index) = scalar_argminmax(data.view());
            let (simd_argmin_index, simd_argmax_index) = unsafe { NEON::argminmax(data.view()) };
            assert_eq!(argmin_index, simd_argmin_index);
            assert_eq!(argmax_index, simd_argmax_index);
        }

        #[test]
        fn test_first_index_is_returned_when_identical_values_found() {
            let data = [
                10,
                std::u16::MIN,
                6,
                9,
                9,
                22,
                std::u16::MAX,
                4,
                std::u16::MAX,
            ];
            let data: Vec<u16> = data.iter().map(|x| *x).collect();
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
            let data = get_array_u16(n);

            let (argmin_index, argmax_index) = scalar_argminmax(data.view());
            let (argmin_simd_index, argmax_simd_index) = unsafe { NEON::argminmax(data.view()) };
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }

        #[test]
        fn test_many_random_runs() {
            for _ in 0..10_000 {
                let data = get_array_u16(32 * 2 + 1);
                let (argmin_index, argmax_index) = scalar_argminmax(data.view());
                let (argmin_simd_index, argmax_simd_index) =
                    unsafe { NEON::argminmax(data.view()) };
                assert_eq!(argmin_index, argmin_simd_index);
                assert_eq!(argmax_index, argmax_simd_index);
            }
        }
    }
}

use super::config::SIMDInstructionSet;
use super::generic::SIMD;
use ndarray::ArrayView1;
#[cfg(target_arch = "arm")]
use std::arch::arm::*;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::utils::{max_index_value, min_index_value};
#[cfg(feature = "half")]
use half::f16;
use num_traits::AsPrimitive;

const XOR_VALUE: i16 = 0x7FFF;

#[cfg(feature = "half")]
#[inline(always)]
fn _ord_i16_to_f16(ord_i16: i16) -> f16 {
    // TODO: more efficient transformation -> can be decreasing order as well
    let v = ((ord_i16 >> 15) & XOR_VALUE) ^ ord_i16;
    unsafe { std::mem::transmute::<i16, f16>(v) }
}

// ------------------------------------------ AVX2 ------------------------------------------

#[cfg(feature = "half")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx2 {
    use super::super::config::AVX2;
    use super::*;

    const LANE_SIZE: usize = AVX2::LANE_SIZE_16;

    // ------------------------------------ ARGMINMAX --------------------------------------

    #[inline(always)]
    unsafe fn _f16_as_m256i_to_i16ord(f16_as_m256i: __m256i) -> __m256i {
        // on a scalar: ((v >> 15) & 0x7FFF) ^ v
        let sign_bit_shifted = _mm256_srai_epi16(f16_as_m256i, 15);
        let sign_bit_masked = _mm256_and_si256(sign_bit_shifted, _mm256_set1_epi16(XOR_VALUE));
        _mm256_xor_si256(sign_bit_masked, f16_as_m256i)
    }

    #[inline(always)]
    unsafe fn _reg_to_i16_arr(reg: __m256i) -> [i16; LANE_SIZE] {
        std::mem::transmute::<__m256i, [i16; LANE_SIZE]>(reg)
    }

    impl SIMD<f16, __m256i, __m256i, LANE_SIZE> for AVX2 {
        const INITIAL_INDEX: __m256i = unsafe {
            std::mem::transmute([
                0i16, 1i16, 2i16, 3i16, 4i16, 5i16, 6i16, 7i16, 8i16, 9i16, 10i16, 11i16, 12i16,
                13i16, 14i16, 15i16,
            ])
        };

        #[inline(always)]
        unsafe fn _reg_to_arr(_: __m256i) -> [f16; LANE_SIZE] {
            // Not used because we work with i16ord and override _get_min_index_value and _get_max_index_value
            unimplemented!()
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const f16) -> __m256i {
            _f16_as_m256i_to_i16ord(_mm256_loadu_si256(data as *const __m256i))
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

        // #[inline]
        // fn _get_min_index_value(index_low: __m256i, values_low: __m256i) -> (usize, f16) {
        //     let index_low_arr = _reg_to_i16_arr(index_low);
        //     let values_low_arr = _reg_to_i16_arr(values_low);
        //     let (min_index, min_value) = min_index_value(&index_low_arr, &values_low_arr);
        //     (min_index.as_(), _ord_i16_to_f16(min_value))
        // }

        // #[inline]
        // fn _get_max_index_value(index_low: __m256i, values_low: __m256i) -> (usize, f16) {
        //     let index_low_arr = _reg_to_i16_arr(index_low);
        //     let values_low_arr = _reg_to_i16_arr(values_low);
        //     let (max_index, max_value) = max_index_value(&index_low_arr, &values_low_arr);
        //     (max_index.as_(), _ord_i16_to_f16(max_value))
        // }

        #[target_feature(enable = "avx2")]
        unsafe fn argminmax(data: ArrayView1<f16>) -> (usize, usize) {
            Self::_argminmax(data)
        }

        #[inline(always)]
        unsafe fn _get_min_max_index_value(
            index_low: __m256i,
            values_low: __m256i,
            index_high: __m256i,
            values_high: __m256i,
        ) -> (usize, f16, usize, f16) {
            let index_low_arr = _reg_to_i16_arr(index_low);
            let values_low_arr = _reg_to_i16_arr(values_low);
            let index_high_arr = _reg_to_i16_arr(index_high);
            let values_high_arr = _reg_to_i16_arr(values_high);
            let (min_index, min_value) = min_index_value(&index_low_arr, &values_low_arr);
            let (max_index, max_value) = max_index_value(&index_high_arr, &values_high_arr);
            (
                min_index.as_(),
                _ord_i16_to_f16(min_value),
                max_index.as_(),
                _ord_i16_to_f16(max_value),
            )
        }
    }

    //----- TESTS -----

    #[cfg(test)]
    mod tests {
        use super::{AVX2, SIMD};
        use crate::scalar::scalar_generic::scalar_argminmax;

        use half::f16;
        use ndarray::Array1;

        extern crate dev_utils;
        use dev_utils::utils;

        fn get_array_f16(n: usize) -> Array1<f16> {
            let arr = utils::get_random_array(n, i16::MIN, i16::MAX);
            let arr = arr.mapv(|x| f16::from_f32(x as f32));
            Array1::from(arr)
        }

        #[test]
        fn test_both_versions_return_the_same_results() {
            let data = get_array_f16(1025);
            assert_eq!(data.len() % 8, 1);

            let (argmin_index, argmax_index) = scalar_argminmax(data.view());
            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX2::argminmax(data.view()) };
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }

        #[test]
        fn test_first_index_is_returned_when_identical_values_found() {
            let data = [
                f16::from_f32(10.),
                f16::MAX,
                f16::from_f32(6.),
                f16::NEG_INFINITY,
                f16::NEG_INFINITY,
                f16::MAX,
                f16::from_f32(5_000.0),
            ];
            let data: Vec<f16> = data.iter().map(|x| *x).collect();
            let data = Array1::from(data);

            let (argmin_index, argmax_index) = scalar_argminmax(data.view());
            assert_eq!(argmin_index, 3);
            assert_eq!(argmax_index, 1);

            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX2::argminmax(data.view()) };
            assert_eq!(argmin_simd_index, 3);
            assert_eq!(argmax_simd_index, 1);
        }

        #[test]
        fn test_many_random_runs() {
            for _ in 0..10_000 {
                let data = get_array_f16(32 * 8 + 1);
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

#[cfg(feature = "half")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod sse {
    use super::super::config::SSE;
    use super::*;

    const LANE_SIZE: usize = SSE::LANE_SIZE_16;

    #[inline(always)]
    unsafe fn _f16_as_m128i_to_i16ord(f16_as_m128i: __m128i) -> __m128i {
        // on a scalar: ((v >> 15) & 0x7FFF) ^ v
        let sign_bit_shifted = _mm_srai_epi16(f16_as_m128i, 15);
        let sign_bit_masked = _mm_and_si128(sign_bit_shifted, _mm_set1_epi16(XOR_VALUE));
        _mm_xor_si128(sign_bit_masked, f16_as_m128i)
    }

    #[inline(always)]
    unsafe fn _reg_to_i16_arr(reg: __m128i) -> [i16; LANE_SIZE] {
        std::mem::transmute::<__m128i, [i16; LANE_SIZE]>(reg)
    }

    impl SIMD<f16, __m128i, __m128i, LANE_SIZE> for SSE {
        const INITIAL_INDEX: __m128i =
            unsafe { std::mem::transmute([0i16, 1i16, 2i16, 3i16, 4i16, 5i16, 6i16, 7i16]) };

        #[inline(always)]
        unsafe fn _reg_to_arr(_: __m128i) -> [f16; LANE_SIZE] {
            // Not used because we work with i16ord and override _get_min_index_value and _get_max_index_value
            unimplemented!()
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const f16) -> __m128i {
            _f16_as_m128i_to_i16ord(_mm_loadu_si128(data as *const __m128i))
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
        unsafe fn argminmax(data: ArrayView1<f16>) -> (usize, usize) {
            Self::_argminmax(data)
        }

        #[inline(always)]
        unsafe fn _get_min_max_index_value(
            index_low: __m128i,
            values_low: __m128i,
            index_high: __m128i,
            values_high: __m128i,
        ) -> (usize, f16, usize, f16) {
            let index_low_arr = _reg_to_i16_arr(index_low);
            let values_low_arr = _reg_to_i16_arr(values_low);
            let index_high_arr = _reg_to_i16_arr(index_high);
            let values_high_arr = _reg_to_i16_arr(values_high);
            let (min_index, min_value) = min_index_value(&index_low_arr, &values_low_arr);
            let (max_index, max_value) = max_index_value(&index_high_arr, &values_high_arr);
            (
                min_index.as_(),
                _ord_i16_to_f16(min_value),
                max_index.as_(),
                _ord_i16_to_f16(max_value),
            )
        }
    }

    // ----------------------------------------- TESTS -----------------------------------------

    #[cfg(test)]
    mod tests {
        use super::{SIMD, SSE};
        use crate::scalar::scalar_generic::scalar_argminmax;

        use half::f16;
        use ndarray::Array1;

        extern crate dev_utils;
        use dev_utils::utils;

        fn get_array_f16(n: usize) -> Array1<f16> {
            let arr = utils::get_random_array(n, i16::MIN, i16::MAX);
            let arr = arr.mapv(|x| f16::from_f32(x as f32));
            Array1::from(arr)
        }

        #[test]
        fn test_both_versions_return_the_same_results() {
            let data = get_array_f16(1025);
            assert_eq!(data.len() % 8, 1);

            let (argmin_index, argmax_index) = scalar_argminmax(data.view());
            let (argmin_simd_index, argmax_simd_index) = unsafe { SSE::argminmax(data.view()) };
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }

        #[test]
        fn test_first_index_is_returned_when_identical_values_found() {
            let data = [
                f16::from_f32(10.),
                f16::MAX,
                f16::from_f32(6.),
                f16::NEG_INFINITY,
                f16::NEG_INFINITY,
                f16::MAX,
                f16::from_f32(5_000.0),
            ];
            let data: Vec<f16> = data.iter().map(|x| *x).collect();
            let data = Array1::from(data);

            let (argmin_index, argmax_index) = scalar_argminmax(data.view());
            assert_eq!(argmin_index, 3);
            assert_eq!(argmax_index, 1);

            let (argmin_simd_index, argmax_simd_index) = unsafe { SSE::argminmax(data.view()) };
            assert_eq!(argmin_simd_index, 3);
            assert_eq!(argmax_simd_index, 1);
        }

        #[test]
        fn test_many_random_runs() {
            for _ in 0..10_000 {
                let data = get_array_f16(32 * 8 + 1);
                let (argmin_index, argmax_index) = scalar_argminmax(data.view());
                let (argmin_simd_index, argmax_simd_index) = unsafe { SSE::argminmax(data.view()) };
                assert_eq!(argmin_index, argmin_simd_index);
                assert_eq!(argmax_index, argmax_simd_index);
            }
        }
    }
}

// --------------------------------------- AVX512 ----------------------------------------

#[cfg(feature = "half")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx512 {
    use super::super::config::AVX512;
    use super::*;

    const LANE_SIZE: usize = AVX512::LANE_SIZE_16;

    #[inline(always)]
    unsafe fn _f16_as_m521i_to_i16ord(f16_as_m512i: __m512i) -> __m512i {
        // on a scalar: ((v >> 15) & 0x7FFF) ^ v
        let sign_bit_shifted = _mm512_srai_epi16(f16_as_m512i, 15);
        let sign_bit_masked = _mm512_and_si512(sign_bit_shifted, _mm512_set1_epi16(XOR_VALUE));
        _mm512_xor_si512(f16_as_m512i, sign_bit_masked)
    }

    #[inline(always)]
    unsafe fn _reg_to_i16_arr(reg: __m512i) -> [i16; LANE_SIZE] {
        std::mem::transmute::<__m512i, [i16; LANE_SIZE]>(reg)
    }

    impl SIMD<f16, __m512i, u32, LANE_SIZE> for AVX512 {
        const INITIAL_INDEX: __m512i = unsafe {
            std::mem::transmute([
                0i16, 1i16, 2i16, 3i16, 4i16, 5i16, 6i16, 7i16, 8i16, 9i16, 10i16, 11i16, 12i16,
                13i16, 14i16, 15i16, 16i16, 17i16, 18i16, 19i16, 20i16, 21i16, 22i16, 23i16, 24i16,
                25i16, 26i16, 27i16, 28i16, 29i16, 30i16, 31i16,
            ])
        };

        #[inline(always)]
        unsafe fn _reg_to_arr(_: __m512i) -> [f16; LANE_SIZE] {
            // Not used because we work with i16ord and override _get_min_index_value and _get_max_index_value
            unimplemented!()
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const f16) -> __m512i {
            _f16_as_m521i_to_i16ord(_mm512_loadu_epi16(data as *const i16))
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
        unsafe fn argminmax(data: ArrayView1<f16>) -> (usize, usize) {
            Self::_argminmax(data)
        }

        #[inline(always)]
        unsafe fn _get_min_max_index_value(
            index_low: __m512i,
            values_low: __m512i,
            index_high: __m512i,
            values_high: __m512i,
        ) -> (usize, f16, usize, f16) {
            let index_low_arr = _reg_to_i16_arr(index_low);
            let values_low_arr = _reg_to_i16_arr(values_low);
            let index_high_arr = _reg_to_i16_arr(index_high);
            let values_high_arr = _reg_to_i16_arr(values_high);
            let (min_index, min_value) = min_index_value(&index_low_arr, &values_low_arr);
            let (max_index, max_value) = max_index_value(&index_high_arr, &values_high_arr);
            (
                min_index.as_(),
                _ord_i16_to_f16(min_value),
                max_index.as_(),
                _ord_i16_to_f16(max_value),
            )
        }
    }

    // ----------------------------------------- TESTS -----------------------------------------

    #[cfg(test)]
    mod tests {
        use super::{AVX512, SIMD};
        use crate::scalar::scalar_generic::scalar_argminmax;

        use half::f16;
        use ndarray::Array1;

        extern crate dev_utils;
        use dev_utils::utils;

        fn get_array_f16(n: usize) -> Array1<f16> {
            let arr = utils::get_random_array(n, i16::MIN, i16::MAX);
            let arr = arr.mapv(|x| f16::from_f32(x as f32));
            Array1::from(arr)
        }

        #[test]
        fn test_both_versions_return_the_same_results() {
            let data = get_array_f16(1025);
            assert_eq!(data.len() % 8, 1);

            let (argmin_index, argmax_index) = scalar_argminmax(data.view());
            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX512::argminmax(data.view()) };
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }

        #[test]
        fn test_first_index_is_returned_when_identical_values_found() {
            let data = [
                f16::from_f32(10.),
                f16::MAX,
                f16::from_f32(6.),
                f16::NEG_INFINITY,
                f16::NEG_INFINITY,
                f16::MAX,
                f16::from_f32(5_000.0),
            ];
            let data: Vec<f16> = data.iter().map(|x| *x).collect();
            let data = Array1::from(data);

            let (argmin_index, argmax_index) = scalar_argminmax(data.view());
            assert_eq!(argmin_index, 3);
            assert_eq!(argmax_index, 1);

            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX512::argminmax(data.view()) };
            assert_eq!(argmin_simd_index, 3);
            assert_eq!(argmax_simd_index, 1);
        }

        #[test]
        fn test_many_random_runs() {
            for _ in 0..10_000 {
                let data = get_array_f16(32 * 8 + 1);
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

#[cfg(feature = "half")]
#[cfg(target_arch = "arm")]
mod neon {
    use super::super::config::NEON;
    use super::*;

    const LANE_SIZE: usize = NEON::LANE_SIZE_16;

    #[inline(always)]
    unsafe fn _f16_as_int16x8_to_i16ord(f16_as_int16x8: int16x8_t) -> int16x8_t {
        // on a scalar: ((v >> 15) & 0x7FFF) ^ v
        let sign_bit_shifted = vshrq_n_s16(f16_as_int16x8, 15);
        let sign_bit_masked = vandq_s16(sign_bit_shifted, vdupq_n_s16(XOR_VALUE));
        veorq_s16(f16_as_int16x8, sign_bit_masked)
    }

    #[inline(always)]
    unsafe fn _reg_to_i16_arr(reg: int16x8_t) -> [i16; LANE_SIZE] {
        std::mem::transmute::<int16x8_t, [i16; LANE_SIZE]>(reg)
    }

    impl SIMD<f16, int16x8_t, uint16x8_t, LANE_SIZE> for NEON {
        const INITIAL_INDEX: int16x8_t =
            unsafe { std::mem::transmute([0i16, 1i16, 2i16, 3i16, 4i16, 5i16, 6i16, 7i16]) };

        #[inline(always)]
        unsafe fn _reg_to_arr(reg: int16x8_t) -> [f16; LANE_SIZE] {
            // Not used because we work with i16ord and override _get_min_index_value and _get_max_index_value
            unimplemented!()
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const f16) -> int16x8_t {
            _f16_as_int16x8_to_i16ord(vld1q_s16(unsafe {
                std::mem::transmute::<*const f16, *const i16>(data)
            }))
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
        unsafe fn argminmax(data: ArrayView1<f16>) -> (usize, usize) {
            Self::_argminmax(data)
        }

        #[inline(always)]
        unsafe fn _get_min_max_index_value(
            index_low: int16x8_t,
            values_low: int16x8_t,
            index_high: int16x8_t,
            values_high: int16x8_t,
        ) -> (usize, f16, usize, f16) {
            let index_low_arr = _reg_to_i16_arr(index_low);
            let values_low_arr = _reg_to_i16_arr(values_low);
            let index_high_arr = _reg_to_i16_arr(index_high);
            let values_high_arr = _reg_to_i16_arr(values_high);
            let (min_index, min_value) = min_index_value(&index_low_arr, &values_low_arr);
            let (max_index, max_value) = max_index_value(&index_high_arr, &values_high_arr);
            (
                min_index.as_(),
                _ord_i16_to_f16(min_value),
                max_index.as_(),
                _ord_i16_to_f16(max_value),
            )
        }
    }

    // ----------------------------------------- TESTS -----------------------------------------

    #[cfg(test)]
    mod tests {
        use super::{NEON, SIMD};
        use crate::scalar::scalar_generic::scalar_argminmax;

        use half::f16;
        use ndarray::Array1;

        extern crate dev_utils;
        use dev_utils::utils;

        fn get_array_f16(n: usize) -> Array1<f16> {
            let arr = utils::get_random_array(n, i16::MIN, i16::MAX);
            let arr = arr.mapv(|x| f16::from_f32(x as f32));
            Array1::from(arr)
        }

        #[test]
        fn test_both_versions_return_the_same_results() {
            let data = get_array_f16(1025);
            assert_eq!(data.len() % 8, 1);

            let (argmin_index, argmax_index) = scalar_argminmax(data.view());
            let (argmin_simd_index, argmax_simd_index) = unsafe { NEON::argminmax(data.view()) };
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }

        #[test]
        fn test_first_index_is_returned_when_identical_values_found() {
            let data = [
                f16::from_f32(10.),
                f16::MAX,
                f16::from_f32(6.),
                f16::NEG_INFINITY,
                f16::NEG_INFINITY,
                f16::MAX,
                f16::from_f32(5_000.0),
            ];
            let data: Vec<f16> = data.iter().map(|x| *x).collect();
            let data = Array1::from(data);

            let (argmin_index, argmax_index) = scalar_argminmax(data.view());
            assert_eq!(argmin_index, 3);
            assert_eq!(argmax_index, 1);

            let (argmin_simd_index, argmax_simd_index) = unsafe { NEON::argminmax(data.view()) };
            assert_eq!(argmin_simd_index, 3);
            assert_eq!(argmax_simd_index, 1);
        }

        #[test]
        fn test_many_random_runs() {
            for _ in 0..10_000 {
                let data = get_array_f16(32 * 8 + 1);
                let (argmin_index, argmax_index) = scalar_argminmax(data.view());
                let (argmin_simd_index, argmax_simd_index) =
                    unsafe { NEON::argminmax(data.view()) };
                assert_eq!(argmin_index, argmin_simd_index);
                assert_eq!(argmax_index, argmax_simd_index);
            }
        }
    }
}

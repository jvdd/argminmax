use super::config::SIMDInstructionSet;
use super::generic::SIMD;
use crate::utils::{max_index_value, min_index_value};
use num_traits::AsPrimitive;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

const XOR_VALUE: i16 = 0x7FFF;

#[inline(always)]
fn _i16decrord_to_u16(decrord_i16: i16) -> u16 {
    // let v = ord_i16 ^ 0x7FFF;
    unsafe { std::mem::transmute::<i16, u16>(decrord_i16 ^ XOR_VALUE) }
}

// ------------------------------------------ AVX2 ------------------------------------------

use super::config::AVX2;

mod avx2 {
    use super::*;

    const LANE_SIZE: usize = AVX2::LANE_SIZE_16;
    const XOR_MASK: __m256i = unsafe { std::mem::transmute([XOR_VALUE; LANE_SIZE]) };
    // const ONES_MASK: __m256i = unsafe { std::mem::transmute([1i16; LANE_SIZE]) };

    // TODO: add vs xor? (also for other unsigned dtypes)
    // Fan van committen nr xor - maar denk dat implementatie nog cleaner kan
    //  - comparison swappen => dan moeten we opt einde niet meer swappen?

    #[inline(always)]
    unsafe fn _u16_to_i16decrord(u16: __m256i) -> __m256i {
        // on a scalar: v^ 0x7FFF
        // transforms to monotonically **decreasing** order
        _mm256_xor_si256(u16, XOR_MASK) // Only 1 operation
    }

    // #[inline(always)]
    // unsafe fn _u16_to_i16ord(u16: __m256i) -> __m256i {
    //     // on a scalar: v + 0x8000
    //     // transforms to monotonically **increasing** order
    //     _mm256_add_epi16(_mm256_add_epi16(u16, XOR_MASK), ONES_MASK)
    // }

    // #[inline(always)]
    // fn _ord_i16_to_u16(ord_i16: i16) -> u16 {
    //     // let v = ord_i16 - 0x8000;
    //     unsafe { std::mem::transmute::<i16, u16>(ord_i16.wrapping_add(0x7FFF).wrapping_add(1)) }
    // }

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

        #[inline(always)]
        unsafe fn _reg_to_arr(_: __m256i) -> [u16; LANE_SIZE] {
            // Not used because we work with i16ord and override _get_min_index_value and _get_max_index_value
            unimplemented!()
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const u16) -> __m256i {
            _u16_to_i16decrord(_mm256_loadu_si256(data as *const __m256i))
            // unsafe { _u16_to_i16ord(_mm256_loadu_si256(data as *const __m256i)) }
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
        unsafe fn argminmax(data: ndarray::ArrayView1<u16>) -> (usize, usize) {
            Self::_argminmax(data)
        }

        #[inline(always)]
        unsafe fn _get_min_max_index_value(
            index_low: __m256i,
            values_low: __m256i,
            index_high: __m256i,
            values_high: __m256i,
        ) -> (usize, u16, usize, u16) {
            let index_low_arr = _reg_to_i16_arr(index_low);
            let values_low_arr = _reg_to_i16_arr(values_low);
            let index_high_arr = _reg_to_i16_arr(index_high);
            let values_high_arr = _reg_to_i16_arr(values_high);
            let (min_index, min_value) = min_index_value(&index_low_arr, &values_low_arr);
            let (max_index, max_value) = max_index_value(&index_high_arr, &values_high_arr);
            // Swap min and max here because we worked with i16ord in decreasing order (max => actual min, and vice versa)
            (
                max_index.as_(),
                _i16decrord_to_u16(max_value),
                min_index.as_(),
                _i16decrord_to_u16(min_value),
            )
            // (min_index.as_(), _ord_i16_to_u16(min_value), max_index.as_(), _ord_i16_to_u16(max_value))
        }
    }

    // ------------------------------------ TESTS --------------------------------------

    #[cfg(test)]
    mod tests {
        use super::{AVX2, SIMD};
        use crate::scalar::scalar_generic::scalar_argminmax;

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
            let (simd_argmin_index, simd_argmax_index) = unsafe { AVX2::argminmax(data.view()) };
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

            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX2::argminmax(data.view()) };
            assert_eq!(argmin_simd_index, 1);
            assert_eq!(argmax_simd_index, 6);
        }

        #[test]
        fn test_many_random_runs() {
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

use super::config::SSE;

mod sse {
    use super::*;

    const LANE_SIZE: usize = SSE::LANE_SIZE_16;
    const XOR_MASK: __m128i = unsafe { std::mem::transmute([XOR_VALUE; LANE_SIZE]) };

    #[inline(always)]
    unsafe fn _u16_to_i16decrord(u16: __m128i) -> __m128i {
        _mm_xor_si128(u16, XOR_MASK)
    }

    #[inline(always)]
    unsafe fn _reg_to_i16_arr(reg: __m128i) -> [i16; LANE_SIZE] {
        std::mem::transmute::<__m128i, [i16; LANE_SIZE]>(reg)
    }

    impl SIMD<u16, __m128i, __m128i, LANE_SIZE> for SSE {
        const INITIAL_INDEX: __m128i =
            unsafe { std::mem::transmute([0i16, 1i16, 2i16, 3i16, 4i16, 5i16, 6i16, 7i16]) };

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
        unsafe fn argminmax(data: ndarray::ArrayView1<u16>) -> (usize, usize) {
            Self::_argminmax(data)
        }

        #[inline(always)]
        unsafe fn _get_min_max_index_value(
            index_low: __m128i,
            values_low: __m128i,
            index_high: __m128i,
            values_high: __m128i,
        ) -> (usize, u16, usize, u16) {
            let index_low_arr = _reg_to_i16_arr(index_low);
            let values_low_arr = _reg_to_i16_arr(values_low);
            let index_high_arr = _reg_to_i16_arr(index_high);
            let values_high_arr = _reg_to_i16_arr(values_high);
            let (min_index, min_value) = min_index_value(&index_low_arr, &values_low_arr);
            let (max_index, max_value) = max_index_value(&index_high_arr, &values_high_arr);
            // Swap min and max here because we worked with i16ord in decreasing order (max => actual min, and vice versa)
            (
                max_index.as_(),
                _i16decrord_to_u16(max_value),
                min_index.as_(),
                _i16decrord_to_u16(min_value),
            )
            // (min_index.as_(), _ord_i16_to_u16(min_value), max_index.as_(), _ord_i16_to_u16(max_value))
        }
    }

    // ----------------------------------------- TESTS -----------------------------------------

    #[cfg(test)]

    mod tests {
        use super::{SIMD, SSE};
        use crate::scalar::scalar_generic::scalar_argminmax;

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

use super::config::AVX512;

mod avx512 {
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
        unsafe fn argminmax(data: ndarray::ArrayView1<u16>) -> (usize, usize) {
            Self::_argminmax(data)
        }

        #[inline(always)]
        unsafe fn _get_min_max_index_value(
            index_low: __m512i,
            values_low: __m512i,
            index_high: __m512i,
            values_high: __m512i,
        ) -> (usize, u16, usize, u16) {
            let index_low_arr = _reg_to_i16_arr(index_low);
            let values_low_arr = _reg_to_i16_arr(values_low);
            let index_high_arr = _reg_to_i16_arr(index_high);
            let values_high_arr = _reg_to_i16_arr(values_high);
            let (min_index, min_value) = min_index_value(&index_low_arr, &values_low_arr);
            let (max_index, max_value) = max_index_value(&index_high_arr, &values_high_arr);
            // Swap min and max here because we worked with i16ord in decreasing order (max => actual min, and vice versa)
            (
                max_index.as_(),
                _i16decrord_to_u16(max_value),
                min_index.as_(),
                _i16decrord_to_u16(min_value),
            )
        }
    }

    // ----------------------------------------- TESTS -----------------------------------------

    #[cfg(test)]

    mod tests {
        use super::{AVX512, SIMD};
        use crate::scalar::scalar_generic::scalar_argminmax;

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
            let (simd_argmin_index, simd_argmax_index) = unsafe { AVX512::argminmax(data.view()) };
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

            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX512::argminmax(data.view()) };
            assert_eq!(argmin_simd_index, 1);
            assert_eq!(argmax_simd_index, 6);
        }

        #[test]
        fn test_many_random_runs() {
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

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use super::config::SIMDInstructionSet;
use super::generic::SIMD;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const XOR_VALUE: i64 = 0x7FFFFFFFFFFFFFFF;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
fn _i64decrord_to_u64(ord_i64: i64) -> u64 {
    // let v = ord_i64 ^ 0x7FFFFFFFFFFFFFFF;
    unsafe { std::mem::transmute::<i64, u64>(ord_i64 ^ XOR_VALUE) }
}

const MAX_INDEX: usize = i64::MAX as usize;

const MIN_VALUE: u64 = i64::MIN as u64; // 0x8000000000000000 -> as i64 = -9223372036854775808
const MAX_VALUE: u64 = i64::MAX as u64;

// ------------------------------------------ AVX2 ------------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx2 {
    use super::super::config::AVX2;
    use super::*;

    const LANE_SIZE: usize = AVX2::LANE_SIZE_64;
    const XOR_MASK: __m256i = unsafe { std::mem::transmute([XOR_VALUE; LANE_SIZE]) };

    #[inline(always)]
    unsafe fn _u64_to_i64decrord(u64: __m256i) -> __m256i {
        // on a scalar: v^ 0x7FFFFFFFFFFFFFFF
        // transforms to monotonically **decreasing** order
        _mm256_xor_si256(u64, XOR_MASK)
    }

    #[inline(always)]
    unsafe fn _reg_to_i64_arr(reg: __m256i) -> [i64; LANE_SIZE] {
        std::mem::transmute::<__m256i, [i64; LANE_SIZE]>(reg)
    }

    impl SIMD<u64, __m256i, __m256i, LANE_SIZE> for AVX2 {
        const INITIAL_INDEX: __m256i = unsafe { std::mem::transmute([0i64, 1i64, 2i64, 3i64]) };
        const INDEX_INCREMENT: __m256i =
            unsafe { std::mem::transmute([LANE_SIZE as i64; LANE_SIZE]) };
        const MAX_INDEX: usize = MAX_INDEX;

        const MIN_VALUE: u64 = MIN_VALUE;
        const MAX_VALUE: u64 = MAX_VALUE;

        #[inline(always)]
        unsafe fn _reg_to_arr(_: __m256i) -> [u64; LANE_SIZE] {
            // Not used because we work with i64ord and override _get_min_index_value and _get_max_index_value
            unimplemented!()
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const u64) -> __m256i {
            _u64_to_i64decrord(_mm256_loadu_si256(data as *const __m256i))
        }

        #[inline(always)]
        unsafe fn _mm_set1(a: u64) -> __m256i {
            _mm256_set1_epi64x(a as i64)
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
        unsafe fn argminmax(data: &[u64]) -> (usize, usize) {
            Self::_argminmax(data)
        }

        #[inline(always)]
        unsafe fn _get_min_max_index_value(
            index_low: __m256i,
            values_low: __m256i,
            index_high: __m256i,
            values_high: __m256i,
        ) -> (usize, u64, usize, u64) {
            let (min_index, min_value) = Self::_horiz_min(index_low, values_low);
            let (max_index, max_value) = Self::_horiz_max(index_high, values_high);
            // Swap min and max here because we worked with i64ord in decreasing order (max => actual min, and vice versa)
            (
                max_index,
                _i64decrord_to_u64(max_value),
                min_index,
                _i64decrord_to_u64(min_value),
            )
        }
    }

    // ------------------------------------ TESTS --------------------------------------

    #[cfg(test)]
    mod tests {
        use super::{AVX2, SIMD};
        use crate::scalar::generic::scalar_argminmax;

        extern crate dev_utils;
        use dev_utils::utils;

        fn get_array_u64(n: usize) -> Vec<u64> {
            utils::get_random_array(n, u64::MIN, u64::MAX)
        }

        #[test]
        fn test_both_versions_return_the_same_results() {
            if !is_x86_feature_detected!("avx2") {
                return;
            }

            let data: &[u64] = &get_array_u64(1025);
            assert_eq!(data.len() % 16, 1);

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            let (simd_argmin_index, simd_argmax_index) = unsafe { AVX2::argminmax(data) };
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
                std::u64::MIN,
                6,
                9,
                9,
                22,
                std::u64::MAX,
                4,
                std::u64::MAX,
            ];
            let data: Vec<u64> = data.iter().map(|x| *x).collect();
            let data: &[u64] = &data;

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            assert_eq!(argmin_index, 1);
            assert_eq!(argmax_index, 6);

            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX2::argminmax(data) };
            assert_eq!(argmin_simd_index, 1);
            assert_eq!(argmax_simd_index, 6);
        }

        #[test]
        fn test_many_random_runs() {
            if !is_x86_feature_detected!("avx2") {
                return;
            }

            for _ in 0..10_000 {
                let data: &[u64] = &get_array_u64(32 * 8 + 1);
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
    const XOR_MASK: __m128i = unsafe { std::mem::transmute([XOR_VALUE; LANE_SIZE]) };

    #[inline(always)]
    unsafe fn _u64_to_i64decrord(u64: __m128i) -> __m128i {
        // on a scalar: v^ 0x7FFFFFFF
        // transforms to monotonically **decreasing** order
        _mm_xor_si128(u64, XOR_MASK)
    }

    #[inline(always)]
    unsafe fn _reg_to_i64_arr(reg: __m128i) -> [i64; LANE_SIZE] {
        std::mem::transmute::<__m128i, [i64; LANE_SIZE]>(reg)
    }

    impl SIMD<u64, __m128i, __m128i, LANE_SIZE> for SSE {
        const INITIAL_INDEX: __m128i = unsafe { std::mem::transmute([0i64, 1i64]) };
        const INDEX_INCREMENT: __m128i =
            unsafe { std::mem::transmute([LANE_SIZE as i64; LANE_SIZE]) };
        const MAX_INDEX: usize = MAX_INDEX;

        const MIN_VALUE: u64 = MIN_VALUE;
        const MAX_VALUE: u64 = MAX_VALUE;

        #[inline(always)]
        unsafe fn _reg_to_arr(_: __m128i) -> [u64; LANE_SIZE] {
            // Not used because we work with i64ord and override _get_min_index_value and _get_max_index_value
            unimplemented!()
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const u64) -> __m128i {
            _u64_to_i64decrord(_mm_loadu_si128(data as *const __m128i))
        }

        #[inline(always)]
        unsafe fn _mm_set1(a: u64) -> __m128i {
            _mm_set1_epi64x(a as i64)
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
        unsafe fn argminmax(data: &[u64]) -> (usize, usize) {
            Self::_argminmax(data)
        }

        #[inline(always)]
        unsafe fn _get_min_max_index_value(
            index_low: __m128i,
            values_low: __m128i,
            index_high: __m128i,
            values_high: __m128i,
        ) -> (usize, u64, usize, u64) {
            let (min_index, min_value) = Self::_horiz_min(index_low, values_low);
            let (max_index, max_value) = Self::_horiz_max(index_high, values_high);
            // Swap min and max here because we worked with i64ord in decreasing order (max => actual min, and vice versa)
            (
                max_index,
                _i64decrord_to_u64(max_value),
                min_index,
                _i64decrord_to_u64(min_value),
            )
        }
    }

    // ----------------------------------------- TESTS -----------------------------------------

    #[cfg(test)]
    mod tests {
        use super::{SIMD, SSE};
        use crate::scalar::generic::scalar_argminmax;

        extern crate dev_utils;
        use dev_utils::utils;

        fn get_array_u64(n: usize) -> Vec<u64> {
            utils::get_random_array(n, u64::MIN, u64::MAX)
        }

        #[test]
        fn test_both_versions_return_the_same_results() {
            let data: &[u64] = &get_array_u64(1025);
            assert_eq!(data.len() % 16, 1);

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            let (simd_argmin_index, simd_argmax_index) = unsafe { SSE::argminmax(data) };
            assert_eq!(argmin_index, simd_argmin_index);
            assert_eq!(argmax_index, simd_argmax_index);
        }

        #[test]
        fn test_first_index_is_returned_when_identical_values_found() {
            let data = [
                10,
                std::u64::MIN,
                6,
                9,
                9,
                22,
                std::u64::MAX,
                4,
                std::u64::MAX,
            ];
            let data: Vec<u64> = data.iter().map(|x| *x).collect();
            let data: &[u64] = &data;

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            assert_eq!(argmin_index, 1);
            assert_eq!(argmax_index, 6);

            let (argmin_simd_index, argmax_simd_index) = unsafe { SSE::argminmax(data) };
            assert_eq!(argmin_simd_index, 1);
            assert_eq!(argmax_simd_index, 6);
        }

        #[test]
        fn test_many_random_runs() {
            for _ in 0..10_000 {
                let data: &[u64] = &get_array_u64(32 * 8 + 1);
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
    const XOR_MASK: __m512i = unsafe { std::mem::transmute([XOR_VALUE; LANE_SIZE]) };

    // TODO - comparison swappen => dan moeten we opt einde niet meer swappen?

    #[inline(always)]
    unsafe fn _u64_to_i64decrord(u64: __m512i) -> __m512i {
        // on a scalar: v^ 0x7FFFFFFFFFFFFFFF
        // transforms to monotonically **decreasing** order
        _mm512_xor_si512(u64, XOR_MASK)
    }

    #[inline(always)]
    unsafe fn _reg_to_i64_arr(reg: __m512i) -> [i64; LANE_SIZE] {
        std::mem::transmute::<__m512i, [i64; LANE_SIZE]>(reg)
    }

    impl SIMD<u64, __m512i, u8, LANE_SIZE> for AVX512 {
        const INITIAL_INDEX: __m512i =
            unsafe { std::mem::transmute([0i64, 1i64, 2i64, 3i64, 4i64, 5i64, 6i64, 7i64]) };
        const INDEX_INCREMENT: __m512i =
            unsafe { std::mem::transmute([LANE_SIZE as i64; LANE_SIZE]) };
        const MAX_INDEX: usize = MAX_INDEX;

        const MIN_VALUE: u64 = MIN_VALUE;
        const MAX_VALUE: u64 = MAX_VALUE;

        #[inline(always)]
        unsafe fn _reg_to_arr(_: __m512i) -> [u64; LANE_SIZE] {
            unimplemented!("We work with decrordi64 and override _get_min_index_value and _get_max_index_value")
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const u64) -> __m512i {
            _u64_to_i64decrord(_mm512_loadu_epi64(data as *const i64))
        }

        #[inline(always)]
        unsafe fn _mm_set1(a: u64) -> __m512i {
            _mm512_set1_epi64(a as i64)
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
            _mm512_cmplt_epi64_mask(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: __m512i, b: __m512i, mask: u8) -> __m512i {
            _mm512_mask_blend_epi64(mask, a, b)
        }

        // ------------------------------------ ARGMINMAX --------------------------------------

        #[target_feature(enable = "avx512f")]
        unsafe fn argminmax(data: &[u64]) -> (usize, usize) {
            Self::_argminmax(data)
        }

        #[inline(always)]
        unsafe fn _get_min_max_index_value(
            index_low: __m512i,
            values_low: __m512i,
            index_high: __m512i,
            values_high: __m512i,
        ) -> (usize, u64, usize, u64) {
            let (min_index, min_value) = Self::_horiz_min(index_low, values_low);
            let (max_index, max_value) = Self::_horiz_max(index_high, values_high);
            // Swap min and max here because we worked with i64ord in decreasing order (max => actual min, and vice versa)
            (
                max_index,
                _i64decrord_to_u64(max_value),
                min_index,
                _i64decrord_to_u64(min_value),
            )
        }
    }

    // ------------------------------------ TESTS --------------------------------------

    #[cfg(test)]
    mod tests {
        use super::{AVX512, SIMD};
        use crate::scalar::generic::scalar_argminmax;

        extern crate dev_utils;
        use dev_utils::utils;

        fn get_array_u64(n: usize) -> Vec<u64> {
            utils::get_random_array(n, u64::MIN, u64::MAX)
        }

        #[test]
        fn test_both_versions_return_the_same_results() {
            if !is_x86_feature_detected!("avx512f") {
                return;
            }

            let data: &[u64] = &get_array_u64(1025);
            assert_eq!(data.len() % 16, 1);

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            let (simd_argmin_index, simd_argmax_index) = unsafe { AVX512::argminmax(data) };
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
                std::u64::MIN,
                6,
                9,
                9,
                22,
                std::u64::MAX,
                4,
                std::u64::MAX,
            ];
            let data: Vec<u64> = data.iter().map(|x| *x).collect();
            let data: &[u64] = &data;

            let (argmin_index, argmax_index) = scalar_argminmax(data);
            assert_eq!(argmin_index, 1);
            assert_eq!(argmax_index, 6);

            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX512::argminmax(data) };
            assert_eq!(argmin_simd_index, 1);
            assert_eq!(argmax_simd_index, 6);
        }

        #[test]
        fn test_many_random_runs() {
            if !is_x86_feature_detected!("avx512f") {
                return;
            }

            for _ in 0..10_000 {
                let data: &[u64] = &get_array_u64(1025);
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
    unimplement_simd!(u64, usize, NEON);
}

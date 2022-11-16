use super::config::SIMDInstructionSet;
use super::generic::SIMD;
use ndarray::ArrayView1;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// ------------------------------------------ AVX2 ------------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx2 {
    use super::super::config::AVX2;
    use super::*;

    const LANE_SIZE: usize = AVX2::LANE_SIZE_64;

    impl SIMD<f64, __m256d, __m256d, LANE_SIZE> for AVX2 {
        const INITIAL_INDEX: __m256d =
            unsafe { std::mem::transmute([0.0f64, 1.0f64, 2.0f64, 3.0f64]) };
        // https://stackoverflow.com/a/3793950
        const MAX_INDEX: usize = 1 << f64::MANTISSA_DIGITS;

        #[inline(always)]
        unsafe fn _reg_to_arr(reg: __m256d) -> [f64; LANE_SIZE] {
            std::mem::transmute::<__m256d, [f64; LANE_SIZE]>(reg)
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const f64) -> __m256d {
            _mm256_loadu_pd(data as *const f64)
        }

        #[inline(always)]
        unsafe fn _mm_set1(a: usize) -> __m256d {
            _mm256_set1_pd(a as f64)
        }

        #[inline(always)]
        unsafe fn _mm_add(a: __m256d, b: __m256d) -> __m256d {
            _mm256_add_pd(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: __m256d, b: __m256d) -> __m256d {
            _mm256_cmp_pd(a, b, _CMP_GT_OQ)
        }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: __m256d, b: __m256d) -> __m256d {
            _mm256_cmp_pd(b, a, _CMP_GT_OQ)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: __m256d, b: __m256d, mask: __m256d) -> __m256d {
            _mm256_blendv_pd(a, b, mask)
        }

        // ------------------------------------ ARGMINMAX --------------------------------------

        #[target_feature(enable = "avx")]
        unsafe fn argminmax(data: ArrayView1<f64>) -> (usize, usize) {
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

        fn get_array_f64(n: usize) -> Array1<f64> {
            utils::get_random_array(n, f64::MIN, f64::MAX)
        }

        #[test]
        fn test_both_versions_return_the_same_results() {
            let data = get_array_f64(1025);
            assert_eq!(data.len() % 4, 1);

            let (argmin_index, argmax_index) = scalar_argminmax(data.view());
            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX2::argminmax(data.view()) };
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }

        #[test]
        fn test_first_index_is_returned_when_identical_values_found() {
            let data = [
                10.,
                std::f64::MAX,
                6.,
                std::f64::NEG_INFINITY,
                std::f64::NEG_INFINITY,
                std::f64::MAX,
                10_000.0,
            ];
            let data: Vec<f64> = data.iter().map(|x| *x).collect();
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
                let data = get_array_f64(32 * 8 + 1);
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

    const LANE_SIZE: usize = SSE::LANE_SIZE_64;

    impl SIMD<f64, __m128d, __m128d, LANE_SIZE> for SSE {
        const INITIAL_INDEX: __m128d = unsafe { std::mem::transmute([0.0f64, 1.0f64]) };
        // https://stackoverflow.com/a/3793950
        const MAX_INDEX: usize = 1 << f64::MANTISSA_DIGITS;

        #[inline(always)]
        unsafe fn _reg_to_arr(reg: __m128d) -> [f64; LANE_SIZE] {
            std::mem::transmute::<__m128d, [f64; LANE_SIZE]>(reg)
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const f64) -> __m128d {
            _mm_loadu_pd(data as *const f64)
        }

        #[inline(always)]
        unsafe fn _mm_set1(a: usize) -> __m128d {
            _mm_set1_pd(a as f64)
        }

        #[inline(always)]
        unsafe fn _mm_add(a: __m128d, b: __m128d) -> __m128d {
            _mm_add_pd(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: __m128d, b: __m128d) -> __m128d {
            _mm_cmpgt_pd(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: __m128d, b: __m128d) -> __m128d {
            _mm_cmplt_pd(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: __m128d, b: __m128d, mask: __m128d) -> __m128d {
            _mm_blendv_pd(a, b, mask)
        }

        // ------------------------------------ ARGMINMAX --------------------------------------

        #[target_feature(enable = "sse4.1")]
        unsafe fn argminmax(data: ArrayView1<f64>) -> (usize, usize) {
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

        fn get_array_f64(n: usize) -> Array1<f64> {
            utils::get_random_array(n, f64::MIN, f64::MAX)
        }

        #[test]
        fn test_both_versions_return_the_same_results() {
            let data = get_array_f64(1025);
            assert_eq!(data.len() % 2, 1);

            let (argmin_index, argmax_index) = scalar_argminmax(data.view());
            let (argmin_simd_index, argmax_simd_index) = unsafe { SSE::argminmax(data.view()) };
            assert_eq!(argmin_index, argmin_simd_index);
            assert_eq!(argmax_index, argmax_simd_index);
        }

        #[test]
        fn test_first_index_is_returned_when_identical_values_found() {
            let data = [
                10.,
                std::f64::MAX,
                6.,
                std::f64::NEG_INFINITY,
                std::f64::NEG_INFINITY,
                std::f64::MAX,
                10_000.0,
            ];
            let data: Vec<f64> = data.iter().map(|x| *x).collect();
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
                let data = get_array_f64(32 * 2 + 1);
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

    const LANE_SIZE: usize = AVX512::LANE_SIZE_64;

    impl SIMD<f64, __m512d, u8, LANE_SIZE> for AVX512 {
        const INITIAL_INDEX: __m512d = unsafe {
            std::mem::transmute([
                0.0f64, 1.0f64, 2.0f64, 3.0f64, 4.0f64, 5.0f64, 6.0f64, 7.0f64,
            ])
        };
        // https://stackoverflow.com/a/3793950
        const MAX_INDEX: usize = 1 << f64::MANTISSA_DIGITS;

        #[inline(always)]
        unsafe fn _reg_to_arr(reg: __m512d) -> [f64; LANE_SIZE] {
            std::mem::transmute::<__m512d, [f64; LANE_SIZE]>(reg)
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const f64) -> __m512d {
            _mm512_loadu_pd(data as *const f64)
        }

        #[inline(always)]
        unsafe fn _mm_set1(a: usize) -> __m512d {
            _mm512_set1_pd(a as f64)
        }

        #[inline(always)]
        unsafe fn _mm_add(a: __m512d, b: __m512d) -> __m512d {
            _mm512_add_pd(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: __m512d, b: __m512d) -> u8 {
            _mm512_cmp_pd_mask(a, b, _CMP_GT_OQ)
        }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: __m512d, b: __m512d) -> u8 {
            _mm512_cmp_pd_mask(a, b, _CMP_LT_OQ)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: __m512d, b: __m512d, mask: u8) -> __m512d {
            _mm512_mask_blend_pd(mask, a, b)
        }

        // ------------------------------------ ARGMINMAX --------------------------------------

        #[target_feature(enable = "avx512f")]
        unsafe fn argminmax(data: ArrayView1<f64>) -> (usize, usize) {
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

        fn get_array_f64(n: usize) -> Array1<f64> {
            utils::get_random_array(n, f64::MIN, f64::MAX)
        }

        #[test]
        fn test_both_versions_return_the_same_results() {
            if !is_x86_feature_detected!("avx512f") {
                return;
            }

            let data = get_array_f64(1025);
            assert_eq!(data.len() % 2, 1);

            let (argmin_index, argmax_index) = scalar_argminmax(data.view());
            let (argmin_simd_index, argmax_simd_index) = unsafe { AVX512::argminmax(data.view()) };
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
                std::f64::MAX,
                6.,
                std::f64::NEG_INFINITY,
                std::f64::NEG_INFINITY,
                std::f64::MAX,
                10_000.0,
            ];
            let data: Vec<f64> = data.iter().map(|x| *x).collect();
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
            if !is_x86_feature_detected!("avx512f") {
                return;
            }

            for _ in 0..10_000 {
                let data = get_array_f64(32 * 2 + 1);
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

// There are no NEON intrinsics for f64, so we need to use the scalar version.
//   although NEON intrinsics exist for i64 and u64, we cannot use them as
//   they there is no 64-bit variant (of any data type) for the following three
//   intrinsics: vadd_, vcgt_, vclt_

#[cfg(target_arch = "arm")]
mod neon {
    use super::super::config::NEON;
    use super::super::generic::unimplement_simd;
    use super::*;

    // We need to (un)implement the SIMD trait for the NEON struct as otherwise the
    // compiler will complain that the trait is not implemented for the struct -
    // even though we are not using the trait for the NEON struct when dealing with
    // > 64 bit data types.
    unimplement_simd!(f64, usize, NEON);
}

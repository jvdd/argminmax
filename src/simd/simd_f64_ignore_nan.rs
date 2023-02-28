/// Implementation of the argminmax operations for f64 that ignores NaN values.
/// This implementation returns the index of the minimum and maximum values.
/// However, unexpected behavior may occur when there are
/// - *only* NaN values in the array
/// - *only* +/- infinity values in the array
/// - *only* NaN and +/- infinity values in the array
/// In these cases, index 0 is returned.
///
/// NaN values are ignored and treated as if they are not present in the array.
/// To realize this we create an initial SIMD register with values +/- infinity.
/// As comparisons with NaN always return false, it is guaranteed that no NaN values
/// are added to the accumulating SIMD register.
///

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use super::config::SIMDInstructionSet;
use super::generic::{SIMDArgMinMaxIgnoreNaN, SIMDOps, SIMDSetOps};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// https://stackoverflow.com/a/3793950
#[cfg(target_arch = "x86_64")]
const MAX_INDEX: usize = 1 << f64::MANTISSA_DIGITS;
#[cfg(target_arch = "x86")] // https://stackoverflow.com/a/29592369
const MAX_INDEX: usize = u32::MAX as usize;

// --------------------------------------- AVX2 ----------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx_ignore_nan {
    use super::super::config::{AVX2IgnoreNaN, AVX2};
    use super::*;

    const LANE_SIZE: usize = AVX2::LANE_SIZE_64;

    impl SIMDOps<f64, __m256d, __m256d, LANE_SIZE> for AVX2IgnoreNaN {
        const INITIAL_INDEX: __m256d =
            unsafe { std::mem::transmute([0.0f64, 1.0f64, 2.0f64, 3.0f64]) };
        const INDEX_INCREMENT: __m256d =
            unsafe { std::mem::transmute([LANE_SIZE as f64; LANE_SIZE]) };
        const MAX_INDEX: usize = MAX_INDEX;

        #[inline(always)]
        unsafe fn _reg_to_arr(reg: __m256d) -> [f64; LANE_SIZE] {
            std::mem::transmute::<__m256d, [f64; LANE_SIZE]>(reg)
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const f64) -> __m256d {
            _mm256_loadu_pd(data as *const f64)
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
    }

    impl SIMDSetOps<f64, __m256d> for AVX2IgnoreNaN {
        #[inline(always)]
        unsafe fn _mm_set1(a: f64) -> __m256d {
            _mm256_set1_pd(a)
        }
    }

    impl SIMDArgMinMaxIgnoreNaN<f64, __m256d, __m256d, LANE_SIZE> for AVX2IgnoreNaN {
        #[target_feature(enable = "avx")]
        unsafe fn argminmax(data: &[f64]) -> (usize, usize) {
            Self::_argminmax(data)
        }
    }
}

// ---------------------------------------- SSE ----------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod sse_ignore_nan {
    use super::super::config::{SSEIgnoreNaN, SSE};
    use super::*;

    const LANE_SIZE: usize = SSE::LANE_SIZE_64;

    impl SIMDOps<f64, __m128d, __m128d, LANE_SIZE> for SSEIgnoreNaN {
        const INITIAL_INDEX: __m128d = unsafe { std::mem::transmute([0.0f64, 1.0f64]) };
        const INDEX_INCREMENT: __m128d =
            unsafe { std::mem::transmute([LANE_SIZE as f64; LANE_SIZE]) };
        const MAX_INDEX: usize = MAX_INDEX;

        #[inline(always)]
        unsafe fn _reg_to_arr(reg: __m128d) -> [f64; LANE_SIZE] {
            std::mem::transmute::<__m128d, [f64; LANE_SIZE]>(reg)
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const f64) -> __m128d {
            _mm_loadu_pd(data as *const f64)
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
    }

    impl SIMDSetOps<f64, __m128d> for SSEIgnoreNaN {
        #[inline(always)]
        unsafe fn _mm_set1(a: f64) -> __m128d {
            _mm_set1_pd(a)
        }
    }

    impl SIMDArgMinMaxIgnoreNaN<f64, __m128d, __m128d, LANE_SIZE> for SSEIgnoreNaN {
        #[target_feature(enable = "sse4.1")] // TODO: check if this is correct
        unsafe fn argminmax(data: &[f64]) -> (usize, usize) {
            Self::_argminmax(data)
        }
    }
}

// -------------------------------------- AVX512 ---------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx512_ignore_nan {
    use super::super::config::{AVX512IgnoreNaN, AVX512};
    use super::*;

    const LANE_SIZE: usize = AVX512::LANE_SIZE_64;

    impl SIMDOps<f64, __m512d, u8, LANE_SIZE> for AVX512IgnoreNaN {
        const INITIAL_INDEX: __m512d = unsafe {
            std::mem::transmute([
                0.0f64, 1.0f64, 2.0f64, 3.0f64, 4.0f64, 5.0f64, 6.0f64, 7.0f64,
            ])
        };
        const INDEX_INCREMENT: __m512d =
            unsafe { std::mem::transmute([LANE_SIZE as f64; LANE_SIZE]) };
        const MAX_INDEX: usize = MAX_INDEX;

        #[inline(always)]
        unsafe fn _reg_to_arr(reg: __m512d) -> [f64; LANE_SIZE] {
            std::mem::transmute::<__m512d, [f64; LANE_SIZE]>(reg)
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const f64) -> __m512d {
            _mm512_loadu_pd(data as *const f64)
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
    }

    impl SIMDSetOps<f64, __m512d> for AVX512IgnoreNaN {
        #[inline(always)]
        unsafe fn _mm_set1(a: f64) -> __m512d {
            _mm512_set1_pd(a)
        }
    }

    impl SIMDArgMinMaxIgnoreNaN<f64, __m512d, u8, LANE_SIZE> for AVX512IgnoreNaN {
        #[target_feature(enable = "avx512f")]
        unsafe fn argminmax(data: &[f64]) -> (usize, usize) {
            Self::_argminmax(data)
        }
    }
}

// --------------------------------------- NEON ----------------------------------------

// There are no NEON intrinsics for f64, so we need to use the scalar version.
//   although NEON intrinsics exist for i64 and u64, we cannot use them as
//   they there is no 64-bit variant (of any data type) for the following three
//   intrinsics: vadd_, vcgt_, vclt_

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
mod neon_ignore_nan {
    use super::super::config::NEONIgnoreNaN;
    use super::super::generic::{unimpl_SIMDArgMinMaxIgnoreNaN, unimpl_SIMDOps};
    use super::*;

    // We need to (un)implement the SIMD trait for the NEON struct as otherwise the
    // compiler will complain that the trait is not implemented for the struct -
    // even though we are not using the trait for the NEON struct when dealing with
    // > 64 bit data types.
    unimpl_SIMDOps!(f64, usize, NEONIgnoreNaN);
    unimpl_SIMDArgMinMaxIgnoreNaN!(f64, usize, NEONIgnoreNaN);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(test)]
mod tests {
    use rstest::rstest;
    use rstest_reuse::{self, *};

    use crate::scalar::generic::scalar_argminmax_ignore_nans as scalar_argminmax;
    use crate::simd::config::{AVX2IgnoreNaN, AVX512IgnoreNaN, SSEIgnoreNaN};
    use crate::SIMDArgMinMaxIgnoreNaN; // todo use type-state pattern

    use super::super::test_utils::{test_long_array_argminmax, test_random_runs_argminmax};
    // Float specific tests
    use super::super::test_utils::{test_ignore_nans_argminmax, test_return_infs_argminmax};

    use dev_utils::utils;

    fn get_array_f64(n: usize) -> Vec<f64> {
        utils::get_random_array(n, f64::MIN, f64::MAX)
    }

    // ------------ Template for x86 / x86_64 -------------

    #[template]
    #[rstest]
    #[case::sse(SSEIgnoreNaN, is_x86_feature_detected!("sse4.1"))]
    #[case::avx2(AVX2IgnoreNaN, is_x86_feature_detected!("avx"))]
    #[case::avx512(AVX512IgnoreNaN, is_x86_feature_detected!("avx512f"))]
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
        T: SIMDArgMinMaxIgnoreNaN<f64, SIMDV, SIMDM, LANE_SIZE>,
        SIMDV: Copy,
        SIMDM: Copy,
    {
        if !simd_available {
            return;
        }

        // TODO: make sure that this is a proper multiple of the LANE_SIZE
        // I think we - ideally - centralize this just as the other tests
        let data = [
            10.,
            f64::MAX,
            6.,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
            f64::MAX,
            10_000.0,
        ];
        let data: &[f64] = &data;

        let (argmin_index, argmax_index) = scalar_argminmax(data);
        assert_eq!(argmin_index, 3);
        assert_eq!(argmax_index, 1);

        let (argmin_simd_index, argmax_simd_index) = unsafe { T::argminmax(data) };
        assert_eq!(argmin_simd_index, 3);
        assert_eq!(argmax_simd_index, 1);
    }

    #[apply(simd_implementations)]
    fn test_return_same_result<T, SIMDV, SIMDM, const LANE_SIZE: usize>(
        #[case] _simd: T, // This is just to make sure the template is applied
        #[case] simd_available: bool,
    ) where
        T: SIMDArgMinMaxIgnoreNaN<f64, SIMDV, SIMDM, LANE_SIZE>,
        SIMDV: Copy,
        SIMDM: Copy,
    {
        if !simd_available {
            return;
        }
        test_long_array_argminmax(get_array_f64, scalar_argminmax, T::argminmax);
        test_random_runs_argminmax(get_array_f64, scalar_argminmax, T::argminmax);
    }

    #[apply(simd_implementations)]
    fn test_return_infs<T, SIMDV, SIMDM, const LANE_SIZE: usize>(
        #[case] _simd: T, // This is just to make sure the template is applied
        #[case] simd_available: bool,
    ) where
        T: SIMDArgMinMaxIgnoreNaN<f64, SIMDV, SIMDM, LANE_SIZE>,
        SIMDV: Copy,
        SIMDM: Copy,
    {
        if !simd_available {
            return;
        }
        test_return_infs_argminmax(get_array_f64, scalar_argminmax, T::argminmax);
    }

    #[apply(simd_implementations)]
    fn test_ignore_nans<T, SIMDV, SIMDM, const LANE_SIZE: usize>(
        #[case] _simd: T, // This is just to make sure the template is applied
        #[case] simd_available: bool,
    ) where
        T: SIMDArgMinMaxIgnoreNaN<f64, SIMDV, SIMDM, LANE_SIZE>,
        SIMDV: Copy,
        SIMDM: Copy,
    {
        if !simd_available {
            return;
        }
        test_ignore_nans_argminmax(get_array_f64, scalar_argminmax, T::argminmax);
    }
}

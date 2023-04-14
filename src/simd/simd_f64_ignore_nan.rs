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
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use super::generic::{impl_SIMDArgMinMax, impl_SIMDInit_FloatIgnoreNaN};
#[cfg(any(target_arch = "x86", target_arch = "x86_64", feature = "nightly_simd"))]
use super::generic::{SIMDArgMinMax, SIMDInit, SIMDOps};
#[cfg(any(target_arch = "x86", target_arch = "x86_64", feature = "nightly_simd"))]
use crate::SCALAR;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use num_traits::Zero;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// The dtype-strategy for performing operations on f64 data: ignore NaN values
#[cfg(any(target_arch = "x86", target_arch = "x86_64", feature = "nightly_simd"))]
use super::super::dtype_strategy::FloatIgnoreNaN;

// https://stackoverflow.com/a/3793950
#[cfg(target_arch = "x86_64")]
const MAX_INDEX: usize = 1 << f64::MANTISSA_DIGITS;
#[cfg(target_arch = "x86")] // https://stackoverflow.com/a/29592369
const MAX_INDEX: usize = u32::MAX as usize;

// --------------------------------------- AVX2 ----------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx_ignore_nan {
    use super::super::config::AVX2;
    use super::*;

    const LANE_SIZE: usize = AVX2::<FloatIgnoreNaN>::LANE_SIZE_64;

    impl SIMDOps<f64, __m256d, __m256d, LANE_SIZE> for AVX2<FloatIgnoreNaN> {
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

        // --- Necessary for impl_SIMDInit_FloatIgnoreNaN

        #[inline(always)]
        unsafe fn _mm_set1(val: f64) -> __m256d {
            _mm256_set1_pd(val)
        }
    }

    impl_SIMDInit_FloatIgnoreNaN!(f64, __m256d, __m256d, LANE_SIZE, AVX2<FloatIgnoreNaN>);

    // Requires just AVX (and thus not necessarily AVX2)
    impl_SIMDArgMinMax!(
        f64,
        __m256d,
        __m256d,
        LANE_SIZE,
        SCALAR<FloatIgnoreNaN>,
        AVX2<FloatIgnoreNaN>,
        "avx"
    );
}

// ---------------------------------------- SSE ----------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod sse_ignore_nan {
    use super::super::config::SSE;
    use super::*;

    const LANE_SIZE: usize = SSE::<FloatIgnoreNaN>::LANE_SIZE_64;

    impl SIMDOps<f64, __m128d, __m128d, LANE_SIZE> for SSE<FloatIgnoreNaN> {
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

        // --- Necessary for impl_SIMDInit_FloatIgnoreNaN

        #[inline(always)]
        unsafe fn _mm_set1(val: f64) -> __m128d {
            _mm_set1_pd(val)
        }
    }

    impl_SIMDInit_FloatIgnoreNaN!(f64, __m128d, __m128d, LANE_SIZE, SSE<FloatIgnoreNaN>);

    impl_SIMDArgMinMax!(
        f64,
        __m128d,
        __m128d,
        LANE_SIZE,
        SCALAR<FloatIgnoreNaN>,
        SSE<FloatIgnoreNaN>,
        "sse4.1"
    );
}

// -------------------------------------- AVX512 ---------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly_simd")]
mod avx512_ignore_nan {
    use super::super::config::AVX512;
    use super::*;

    const LANE_SIZE: usize = AVX512::<FloatIgnoreNaN>::LANE_SIZE_64;

    impl SIMDOps<f64, __m512d, u8, LANE_SIZE> for AVX512<FloatIgnoreNaN> {
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

        // --- Necessary for impl_SIMDInit_FloatIgnoreNaN

        #[inline(always)]
        unsafe fn _mm_set1(val: f64) -> __m512d {
            _mm512_set1_pd(val)
        }
    }

    impl_SIMDInit_FloatIgnoreNaN!(f64, __m512d, u8, LANE_SIZE, AVX512<FloatIgnoreNaN>);

    impl_SIMDArgMinMax!(
        f64,
        __m512d,
        u8,
        LANE_SIZE,
        SCALAR<FloatIgnoreNaN>,
        AVX512<FloatIgnoreNaN>,
        "avx512f"
    );
}

// --------------------------------------- NEON ----------------------------------------

// There are no NEON intrinsics for f64, so we need to use the scalar version.
//   although NEON intrinsics exist for i64 and u64, we cannot use them as
//   they there is no 64-bit variant (of any data type) for the following three
//   intrinsics: vadd_, vcgt_, vclt_

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
#[cfg(feature = "nightly_simd")]
mod neon_ignore_nan {
    use super::super::config::NEON;
    use super::super::generic::{unimpl_SIMDArgMinMax, unimpl_SIMDInit, unimpl_SIMDOps};
    use super::*;

    // We need to (un)implement the SIMD trait for the NEON struct as otherwise the
    // compiler will complain that the trait is not implemented for the struct -
    // even though we are not using the trait for the NEON struct when dealing with
    // > 64 bit data types.
    unimpl_SIMDOps!(f64, usize, NEON<FloatIgnoreNaN>);
    unimpl_SIMDInit!(f64, usize, NEON<FloatIgnoreNaN>);
    unimpl_SIMDArgMinMax!(f64, usize, SCALAR<FloatIgnoreNaN>, NEON<FloatIgnoreNaN>);
}

// ======================================= TESTS =======================================

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(test)]
mod tests {
    use rstest::rstest;
    use rstest_reuse::{self, *};
    use std::marker::PhantomData;

    #[cfg(feature = "nightly_simd")]
    use crate::simd::config::AVX512;
    use crate::simd::config::{AVX2, SSE};
    use crate::{FloatIgnoreNaN, SIMDArgMinMax, SCALAR};

    use super::super::test_utils::{
        test_first_index_identical_values_argminmax, test_return_same_result_argminmax,
    };
    // Float specific tests
    use super::super::test_utils::{test_ignore_nans_argminmax, test_return_infs_argminmax};

    use dev_utils::utils;

    fn get_array_f64(n: usize) -> Vec<f64> {
        utils::get_random_array(n, f64::MIN, f64::MAX)
    }

    // The scalar implementation
    const SCALAR_STRATEGY: SCALAR<FloatIgnoreNaN> = SCALAR {
        _dtype_strategy: PhantomData::<FloatIgnoreNaN>,
    };

    // ------------ Template for x86 / x86_64 -------------

    #[template]
    #[rstest]
    #[case::sse(SSE {_dtype_strategy: PhantomData::<FloatIgnoreNaN>}, is_x86_feature_detected!("sse4.1"))]
    #[case::avx2(AVX2 {_dtype_strategy: PhantomData::<FloatIgnoreNaN>}, is_x86_feature_detected!("avx"))]
    #[cfg_attr(feature = "nightly_simd", case::avx512(AVX512 {_dtype_strategy: PhantomData::<FloatIgnoreNaN>}, is_x86_feature_detected!("avx512f")))]
    fn simd_implementations<T, SIMDV, SIMDM, const LANE_SIZE: usize>(
        #[case] simd: T,
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
        #[case] simd: T,
        #[case] simd_available: bool,
    ) where
        T: SIMDArgMinMax<f64, SIMDV, SIMDM, LANE_SIZE, SCALAR<FloatIgnoreNaN>>,
        SIMDV: Copy,
        SIMDM: Copy,
    {
        if !simd_available {
            return;
        }
        test_first_index_identical_values_argminmax(SCALAR_STRATEGY, simd);
    }

    #[apply(simd_implementations)]
    fn test_return_same_result<T, SIMDV, SIMDM, const LANE_SIZE: usize>(
        #[case] simd: T,
        #[case] simd_available: bool,
    ) where
        T: SIMDArgMinMax<f64, SIMDV, SIMDM, LANE_SIZE, SCALAR<FloatIgnoreNaN>>,
        SIMDV: Copy,
        SIMDM: Copy,
    {
        if !simd_available {
            return;
        }
        test_return_same_result_argminmax(get_array_f64, SCALAR_STRATEGY, simd);
    }

    #[apply(simd_implementations)]
    fn test_return_infs<T, SIMDV, SIMDM, const LANE_SIZE: usize>(
        #[case] simd: T,
        #[case] simd_available: bool,
    ) where
        T: SIMDArgMinMax<f64, SIMDV, SIMDM, LANE_SIZE, SCALAR<FloatIgnoreNaN>>,
        SIMDV: Copy,
        SIMDM: Copy,
    {
        if !simd_available {
            return;
        }
        test_return_infs_argminmax(get_array_f64, SCALAR_STRATEGY, simd);
    }

    #[apply(simd_implementations)]
    fn test_ignore_nans<T, SIMDV, SIMDM, const LANE_SIZE: usize>(
        #[case] simd: T,
        #[case] simd_available: bool,
    ) where
        T: SIMDArgMinMax<f64, SIMDV, SIMDM, LANE_SIZE, SCALAR<FloatIgnoreNaN>>,
        SIMDV: Copy,
        SIMDM: Copy,
    {
        if !simd_available {
            return;
        }
        test_ignore_nans_argminmax(get_array_f64, SCALAR_STRATEGY, simd);
    }
}

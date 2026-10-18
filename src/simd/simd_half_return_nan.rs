//! Implementation of the argminmax operations for f16 and bf16 where NaN values take
//! precedence.
//! This implementation returns the index of the first* NaN value if any are present,
//! otherwise it returns the index of the minimum and maximum values.
//!
//! To serve this functionality we transform the values (a sign bit followed by the
//! magnitude) to ordinal i16 values:
//!     ord_i16 = v               if v >= 0 (as i16)
//!     ord_i16 = i16::MIN - v    otherwise (i.e., minus the magnitude)
//!
//! This transformation is reversible (with the same formula), except that -0.0 and 0.0
//! are both mapped to 0 (as these are equal).
//!
//! Through this transformation we can perform the argminmax operations on the ordinal
//! integer values and then transform the result back to the original f16 / bf16 values.
//! This transformation is necessary because comparisons with NaN values are always false.
//! So unless we perform ! <=  as gt and ! >=  as lt the argminmax operations will not
//! add NaN values to the accumulating SIMD register. And as le and ge are significantly
//! more expensive than lt and gt we use this efficient bitwise transformation.
//!
//! `_horiz_min` and `_horiz_max` call the i16 ones of the same instruction set and only
//! transform the returned value back to f16 / bf16. This is correct because the values
//! are i16 ordinals and the index lanes are the same as for i16 (the same `MAX_INDEX`).
//!
//! Note that most x86 CPUs do not support f16 and bf16 instructions - making this implementation
//! multitudes (up to 300x for f16) faster than trying to use a vanilla scalar implementation.
//!
//!
//! ---
//!
//! *Note: the first NaN value is only returned iff all NaN values have the same bit
//! representation. When NaN values have different bit representations then the index of
//! the highest / lowest ord_i16 is returned for the
//! SIMDOps::_get_overflow_lane_size_limit() chunk of the data - which is not
//! necessarily the index of the first NaN value.
//!
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "arm", feature = "nightly_simd"),
    target_arch = "aarch64",
))]
use super::config::SIMDInstructionSet;
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "arm", feature = "nightly_simd"),
    target_arch = "aarch64",
))]
use super::generic::{
    impl_SIMDArgMinMax_FloatReturnNaN, impl_SIMDInit_FloatReturnNaN, SIMDArgMinMax, SIMDInit,
    SIMDMasked, SIMDOps,
};
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "arm", feature = "nightly_simd"),
    target_arch = "aarch64",
))]
use crate::{
    scalar::scalar_half::{i16ord_to_half, Half},
    Int, SCALAR,
};
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(all(target_arch = "arm", feature = "nightly_simd"))]
use std::arch::arm::*;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "arm", feature = "nightly_simd"),
    target_arch = "aarch64",
))]
use half::{bf16, f16};

/// The dtype-strategy for performing operations on f16 and bf16 data: return NaN index
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "arm", feature = "nightly_simd"),
    target_arch = "aarch64",
))]
use super::super::dtype_strategy::FloatReturnNaN;

#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "arm", feature = "nightly_simd"),
    target_arch = "aarch64",
))]
const MAX_INDEX: usize = i16::MAX as usize;

// --------------------------------------- AVX2 ----------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx2 {
    use super::super::config::AVX2;
    use super::*;

    const LANE_SIZE: usize = AVX2::<FloatReturnNaN>::LANE_SIZE_16;
    const LOWER_15_MASK: __m256i = unsafe { std::mem::transmute([i16::MAX; LANE_SIZE]) };

    #[inline(always)]
    unsafe fn _half_as_m256i_to_i16ord(half_as_m256i: __m256i) -> __m256i {
        // on a scalar: v if v >= 0 else i16::MIN - v (-0.0 and 0.0 are both 0)
        _mm256_sign_epi16(
            _mm256_and_si256(half_as_m256i, LOWER_15_MASK),
            half_as_m256i,
        )
    }

    impl<T: Half> SIMDOps<T, __m256i, __m256i, LANE_SIZE> for AVX2<FloatReturnNaN> {
        const INITIAL_INDEX: __m256i = unsafe {
            std::mem::transmute([
                0i16, 1i16, 2i16, 3i16, 4i16, 5i16, 6i16, 7i16, 8i16, 9i16, 10i16, 11i16, 12i16,
                13i16, 14i16, 15i16,
            ])
        };
        const INDEX_INCREMENT: __m256i =
            unsafe { std::mem::transmute([LANE_SIZE as i16; LANE_SIZE]) };
        const MAX_INDEX: usize = MAX_INDEX;

        #[inline(always)]
        unsafe fn _reg_to_arr(_: __m256i) -> [T; LANE_SIZE] {
            // Not implemented because we will perform the horizontal operations on the
            // signed integer values instead of trying to retransform **only** the values
            // (and thus not the indices) to floats.
            unimplemented!()
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const T) -> __m256i {
            _half_as_m256i_to_i16ord(_mm256_loadu_si256(data as *const __m256i))
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

        #[inline(always)]
        unsafe fn _mm_min(a: __m256i, b: __m256i) -> __m256i {
            _mm256_min_epi16(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_max(a: __m256i, b: __m256i) -> __m256i {
            _mm256_max_epi16(a, b)
        }

        #[inline(always)]
        unsafe fn _horiz_min(index: __m256i, value: __m256i) -> (usize, T) {
            let (min_index, min_value) =
                <AVX2<Int> as SIMDOps<i16, _, _, LANE_SIZE>>::_horiz_min(index, value);
            (min_index, i16ord_to_half(min_value))
        }

        #[inline(always)]
        unsafe fn _horiz_max(index: __m256i, value: __m256i) -> (usize, T) {
            let (max_index, max_value) =
                <AVX2<Int> as SIMDOps<i16, _, _, LANE_SIZE>>::_horiz_max(index, value);
            (max_index, i16ord_to_half(max_value))
        }
    }

    impl_SIMDInit_FloatReturnNaN!(f16, __m256i, __m256i, LANE_SIZE, AVX2<FloatReturnNaN>);
    impl_SIMDInit_FloatReturnNaN!(bf16, __m256i, __m256i, LANE_SIZE, AVX2<FloatReturnNaN>);

    impl_SIMDArgMinMax_FloatReturnNaN!(
        f16,
        __m256i,
        __m256i,
        LANE_SIZE,
        AVX2<FloatReturnNaN>,
        "avx2"
    );
    impl_SIMDArgMinMax_FloatReturnNaN!(
        bf16,
        __m256i,
        __m256i,
        LANE_SIZE,
        AVX2<FloatReturnNaN>,
        "avx2"
    );
}

// ---------------------------------------- SSE ----------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod sse {
    use super::super::config::SSE;
    use super::*;

    const LANE_SIZE: usize = SSE::<FloatReturnNaN>::LANE_SIZE_16;
    const LOWER_15_MASK: __m128i = unsafe { std::mem::transmute([i16::MAX; LANE_SIZE]) };

    #[inline(always)]
    unsafe fn _half_as_m128i_to_i16ord(half_as_m128i: __m128i) -> __m128i {
        // on a scalar: v if v >= 0 else i16::MIN - v (-0.0 and 0.0 are both 0)
        _mm_sign_epi16(_mm_and_si128(half_as_m128i, LOWER_15_MASK), half_as_m128i)
    }

    impl<T: Half> SIMDOps<T, __m128i, __m128i, LANE_SIZE> for SSE<FloatReturnNaN> {
        const INITIAL_INDEX: __m128i =
            unsafe { std::mem::transmute([0i16, 1i16, 2i16, 3i16, 4i16, 5i16, 6i16, 7i16]) };
        const INDEX_INCREMENT: __m128i =
            unsafe { std::mem::transmute([LANE_SIZE as i16; LANE_SIZE]) };
        const MAX_INDEX: usize = MAX_INDEX;

        #[inline(always)]
        unsafe fn _reg_to_arr(_: __m128i) -> [T; LANE_SIZE] {
            // Not implemented because we will perform the horizontal operations on the
            // signed integer values instead of trying to retransform **only** the values
            // (and thus not the indices) to floats.
            unimplemented!()
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const T) -> __m128i {
            _half_as_m128i_to_i16ord(_mm_loadu_si128(data as *const __m128i))
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

        #[inline(always)]
        unsafe fn _horiz_min(index: __m128i, value: __m128i) -> (usize, T) {
            let (min_index, min_value) =
                <SSE<Int> as SIMDOps<i16, _, _, LANE_SIZE>>::_horiz_min(index, value);
            (min_index, i16ord_to_half(min_value))
        }

        #[inline(always)]
        unsafe fn _horiz_max(index: __m128i, value: __m128i) -> (usize, T) {
            let (max_index, max_value) =
                <SSE<Int> as SIMDOps<i16, _, _, LANE_SIZE>>::_horiz_max(index, value);
            (max_index, i16ord_to_half(max_value))
        }
    }

    impl_SIMDInit_FloatReturnNaN!(f16, __m128i, __m128i, LANE_SIZE, SSE<FloatReturnNaN>);
    impl_SIMDInit_FloatReturnNaN!(bf16, __m128i, __m128i, LANE_SIZE, SSE<FloatReturnNaN>);

    impl_SIMDArgMinMax_FloatReturnNaN!(
        f16,
        __m128i,
        __m128i,
        LANE_SIZE,
        SSE<FloatReturnNaN>,
        "sse4.1"
    );
    impl_SIMDArgMinMax_FloatReturnNaN!(
        bf16,
        __m128i,
        __m128i,
        LANE_SIZE,
        SSE<FloatReturnNaN>,
        "sse4.1"
    );
}

// -------------------------------------- AVX512 ---------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx512 {
    use super::super::config::AVX512;
    use super::*;

    const LANE_SIZE: usize = AVX512::<FloatReturnNaN>::LANE_SIZE_16;
    const SIGN_BIT: __m512i = unsafe { std::mem::transmute([i16::MIN; LANE_SIZE]) };

    #[inline(always)]
    unsafe fn _half_as_m512i_to_i16ord(half_as_m512i: __m512i) -> __m512i {
        // on a scalar: v if v >= 0 else i16::MIN - v (-0.0 and 0.0 are both 0)
        let negative = _mm512_cmplt_epi16_mask(half_as_m512i, _mm512_setzero_si512());
        _mm512_mask_sub_epi16(half_as_m512i, negative, SIGN_BIT, half_as_m512i)
    }

    impl<T: Half> SIMDOps<T, __m512i, u32, LANE_SIZE> for AVX512<FloatReturnNaN> {
        const INITIAL_INDEX: __m512i = unsafe {
            std::mem::transmute([
                0i16, 1i16, 2i16, 3i16, 4i16, 5i16, 6i16, 7i16, 8i16, 9i16, 10i16, 11i16, 12i16,
                13i16, 14i16, 15i16, 16i16, 17i16, 18i16, 19i16, 20i16, 21i16, 22i16, 23i16, 24i16,
                25i16, 26i16, 27i16, 28i16, 29i16, 30i16, 31i16,
            ])
        };
        const INDEX_INCREMENT: __m512i =
            unsafe { std::mem::transmute([LANE_SIZE as i16; LANE_SIZE]) };
        const MAX_INDEX: usize = MAX_INDEX;

        #[inline(always)]
        unsafe fn _reg_to_arr(_: __m512i) -> [T; LANE_SIZE] {
            // Not implemented because we will perform the horizontal operations on the
            // signed integer values instead of trying to retransform **only** the values
            // (and thus not the indices) to floats.
            unimplemented!()
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const T) -> __m512i {
            _half_as_m512i_to_i16ord(_mm512_loadu_epi16(data as *const i16))
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

        #[inline(always)]
        unsafe fn _horiz_min(index: __m512i, value: __m512i) -> (usize, T) {
            let (min_index, min_value) =
                <AVX512<Int> as SIMDOps<i16, _, _, LANE_SIZE>>::_horiz_min(index, value);
            (min_index, i16ord_to_half(min_value))
        }

        #[inline(always)]
        unsafe fn _horiz_max(index: __m512i, value: __m512i) -> (usize, T) {
            let (max_index, max_value) =
                <AVX512<Int> as SIMDOps<i16, _, _, LANE_SIZE>>::_horiz_max(index, value);
            (max_index, i16ord_to_half(max_value))
        }
    }

    impl_SIMDInit_FloatReturnNaN!(f16, __m512i, u32, LANE_SIZE, AVX512<FloatReturnNaN>);
    impl_SIMDInit_FloatReturnNaN!(bf16, __m512i, u32, LANE_SIZE, AVX512<FloatReturnNaN>);

    impl_SIMDArgMinMax_FloatReturnNaN!(
        f16,
        __m512i,
        u32,
        LANE_SIZE,
        AVX512<FloatReturnNaN>,
        "avx512bw"
    );
    impl_SIMDArgMinMax_FloatReturnNaN!(
        bf16,
        __m512i,
        u32,
        LANE_SIZE,
        AVX512<FloatReturnNaN>,
        "avx512bw"
    );
}

// --------------------------------------- NEON ----------------------------------------

#[cfg(any(
    all(target_arch = "arm", feature = "nightly_simd"),
    target_arch = "aarch64" // stable for AArch64
))]
mod neon {
    use super::super::config::NEON;
    use super::*;

    const LANE_SIZE: usize = NEON::<FloatReturnNaN>::LANE_SIZE_16;
    const SIGN_BIT: int16x8_t = unsafe { std::mem::transmute([i16::MIN; LANE_SIZE]) };

    #[inline(always)]
    unsafe fn _half_as_int16x8_to_i16ord(half_as_int16x8: int16x8_t) -> int16x8_t {
        // on a scalar: v if v >= 0 else i16::MIN - v (-0.0 and 0.0 are both 0)
        // = abs(v) ^ (v & i16::MIN), as abs wraps around (abs(i16::MIN) = i16::MIN)
        veorq_s16(
            vabsq_s16(half_as_int16x8),
            vandq_s16(half_as_int16x8, SIGN_BIT),
        )
    }

    impl<T: Half> SIMDOps<T, int16x8_t, uint16x8_t, LANE_SIZE> for NEON<FloatReturnNaN> {
        const INITIAL_INDEX: int16x8_t =
            unsafe { std::mem::transmute([0i16, 1i16, 2i16, 3i16, 4i16, 5i16, 6i16, 7i16]) };
        const INDEX_INCREMENT: int16x8_t =
            unsafe { std::mem::transmute([LANE_SIZE as i16; LANE_SIZE]) };
        const MAX_INDEX: usize = MAX_INDEX;

        #[inline(always)]
        unsafe fn _reg_to_arr(_: int16x8_t) -> [T; LANE_SIZE] {
            // Not implemented because we will perform the horizontal operations on the
            // signed integer values instead of trying to retransform **only** the values
            // (and thus not the indices) to floats.
            unimplemented!()
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const T) -> int16x8_t {
            _half_as_int16x8_to_i16ord(vld1q_s16(data as *const i16))
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

        #[inline(always)]
        unsafe fn _horiz_min(index: int16x8_t, value: int16x8_t) -> (usize, T) {
            let (min_index, min_value) =
                <NEON<Int> as SIMDOps<i16, _, _, LANE_SIZE>>::_horiz_min(index, value);
            (min_index, i16ord_to_half(min_value))
        }

        #[inline(always)]
        unsafe fn _horiz_max(index: int16x8_t, value: int16x8_t) -> (usize, T) {
            let (max_index, max_value) =
                <NEON<Int> as SIMDOps<i16, _, _, LANE_SIZE>>::_horiz_max(index, value);
            (max_index, i16ord_to_half(max_value))
        }
    }

    impl_SIMDInit_FloatReturnNaN!(f16, int16x8_t, uint16x8_t, LANE_SIZE, NEON<FloatReturnNaN>);
    impl_SIMDInit_FloatReturnNaN!(bf16, int16x8_t, uint16x8_t, LANE_SIZE, NEON<FloatReturnNaN>);

    impl_SIMDArgMinMax_FloatReturnNaN!(
        f16,
        int16x8_t,
        uint16x8_t,
        LANE_SIZE,
        NEON<FloatReturnNaN>,
        "neon"
    );
    impl_SIMDArgMinMax_FloatReturnNaN!(
        bf16,
        int16x8_t,
        uint16x8_t,
        LANE_SIZE,
        NEON<FloatReturnNaN>,
        "neon"
    );
}

// ======================================= TESTS =======================================

#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "arm", feature = "nightly_simd"),
    target_arch = "aarch64",
))]
#[cfg(test)]
mod tests {
    use rstest::rstest;
    use rstest_reuse::{self, *};
    use std::marker::PhantomData;

    use half::{bf16, f16};

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    use crate::simd::config::AVX512;
    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    use crate::simd::config::NEON;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    use crate::simd::config::{AVX2, SSE};
    use crate::{FloatReturnNaN, SIMDArgMinMax, SCALAR};

    use super::super::test_utils::{
        test_first_index_identical_values_argminmax, test_no_overflow_argminmax,
        test_return_same_result_argminmax,
    };
    // Float specific tests
    use super::super::test_utils::{
        test_nan_payloads_argminmax, test_return_infs_argminmax, test_return_nans_argminmax,
        test_signed_zeros_argminmax,
    };
    // Masked tests
    use super::super::test_utils::{
        test_adversarial_masked_argminmax, test_nans_masked_argminmax,
        test_no_overflow_masked_argminmax, test_return_same_result_masked_argminmax,
    };

    use crate::scalar::scalar_half::Half;
    use dev_utils::utils::SampleUniformFullRange;

    // The scalar implementation
    const SCALAR_STRATEGY: SCALAR<FloatReturnNaN> = SCALAR {
        _dtype_strategy: PhantomData::<FloatReturnNaN>,
    };

    // ------------ Template for x86 / x86_64 -------------

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[template]
    #[rstest]
    #[case::sse_f16(SSE {_dtype_strategy: PhantomData::<FloatReturnNaN>}, f16::ZERO, is_x86_feature_detected!("sse4.1"))]
    #[case::sse_bf16(SSE {_dtype_strategy: PhantomData::<FloatReturnNaN>}, bf16::ZERO, is_x86_feature_detected!("sse4.1"))]
    #[case::avx2_f16(AVX2 {_dtype_strategy: PhantomData::<FloatReturnNaN>}, f16::ZERO, is_x86_feature_detected!("avx2"))]
    #[case::avx2_bf16(AVX2 {_dtype_strategy: PhantomData::<FloatReturnNaN>}, bf16::ZERO, is_x86_feature_detected!("avx2"))]
    #[case::avx512_f16(AVX512 {_dtype_strategy: PhantomData::<FloatReturnNaN>}, f16::ZERO, is_x86_feature_detected!("avx512bw"))]
    #[case::avx512_bf16(AVX512 {_dtype_strategy: PhantomData::<FloatReturnNaN>}, bf16::ZERO, is_x86_feature_detected!("avx512bw"))]
    fn simd_implementations<DType, T, SIMDV, SIMDM, const LANE_SIZE: usize>(
        #[case] simd: T,
        #[case] _dtype: DType,
        #[case] simd_available: bool,
    ) {
    }

    // ------------ Template for ARM / AArch64 ------------

    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    #[template]
    #[rstest]
    #[case::neon_f16(NEON {_dtype_strategy: PhantomData::<FloatReturnNaN>}, f16::ZERO, true)]
    #[case::neon_bf16(NEON {_dtype_strategy: PhantomData::<FloatReturnNaN>}, bf16::ZERO, true)]
    fn simd_implementations<DType, T, SIMDV, SIMDM, const LANE_SIZE: usize>(
        #[case] simd: T,
        #[case] _dtype: DType,
        #[case] simd_available: bool,
    ) {
    }

    // ----------------- The actual tests -----------------

    #[apply(simd_implementations)]
    fn test_first_index_is_returned_when_identical_values_found<
        DType,
        T,
        SIMDV,
        SIMDM,
        const LANE_SIZE: usize,
    >(
        #[case] simd: T,
        #[case] _dtype: DType,
        #[case] simd_available: bool,
    ) where
        DType: Half + SampleUniformFullRange,
        T: SIMDArgMinMax<DType, SIMDV, SIMDM, LANE_SIZE, SCALAR<FloatReturnNaN>>,
        SIMDV: Copy,
        SIMDM: Copy,
    {
        if !simd_available {
            return;
        }
        test_first_index_identical_values_argminmax(SCALAR_STRATEGY, simd);
    }

    #[apply(simd_implementations)]
    fn test_return_same_result<DType, T, SIMDV, SIMDM, const LANE_SIZE: usize>(
        #[case] simd: T,
        #[case] _dtype: DType,
        #[case] simd_available: bool,
    ) where
        DType: Half + SampleUniformFullRange,
        T: SIMDArgMinMax<DType, SIMDV, SIMDM, LANE_SIZE, SCALAR<FloatReturnNaN>>,
        SIMDV: Copy,
        SIMDM: Copy,
    {
        if !simd_available {
            return;
        }
        test_return_same_result_argminmax(DType::get_random_array, SCALAR_STRATEGY, simd);
    }

    #[apply(simd_implementations)]
    fn test_no_overflow<DType, T, SIMDV, SIMDM, const LANE_SIZE: usize>(
        #[case] simd: T,
        #[case] _dtype: DType,
        #[case] simd_available: bool,
    ) where
        DType: Half + SampleUniformFullRange,
        T: SIMDArgMinMax<DType, SIMDV, SIMDM, LANE_SIZE, SCALAR<FloatReturnNaN>>,
        SIMDV: Copy,
        SIMDM: Copy,
    {
        if !simd_available {
            return;
        }
        test_no_overflow_argminmax(DType::get_random_array, SCALAR_STRATEGY, simd, None);
    }

    #[apply(simd_implementations)]
    fn test_return_infs<DType, T, SIMDV, SIMDM, const LANE_SIZE: usize>(
        #[case] simd: T,
        #[case] _dtype: DType,
        #[case] simd_available: bool,
    ) where
        DType: Half + SampleUniformFullRange,
        T: SIMDArgMinMax<DType, SIMDV, SIMDM, LANE_SIZE, SCALAR<FloatReturnNaN>>,
        SIMDV: Copy,
        SIMDM: Copy,
    {
        if !simd_available {
            return;
        }
        test_return_infs_argminmax(DType::get_random_array, SCALAR_STRATEGY, simd);
    }

    #[apply(simd_implementations)]
    fn test_return_nans<DType, T, SIMDV, SIMDM, const LANE_SIZE: usize>(
        #[case] simd: T,
        #[case] _dtype: DType,
        #[case] simd_available: bool,
    ) where
        DType: Half + SampleUniformFullRange,
        T: SIMDArgMinMax<DType, SIMDV, SIMDM, LANE_SIZE, SCALAR<FloatReturnNaN>>,
        SIMDV: Copy,
        SIMDM: Copy,
    {
        if !simd_available {
            return;
        }
        test_return_nans_argminmax(DType::get_random_array, SCALAR_STRATEGY, simd);
    }

    #[apply(simd_implementations)]
    fn test_signed_zeros<DType, T, SIMDV, SIMDM, const LANE_SIZE: usize>(
        #[case] simd: T,
        #[case] _dtype: DType,
        #[case] simd_available: bool,
    ) where
        DType: Half + SampleUniformFullRange,
        T: SIMDArgMinMax<DType, SIMDV, SIMDM, LANE_SIZE, SCALAR<FloatReturnNaN>>,
        SIMDV: Copy,
        SIMDM: Copy,
    {
        if !simd_available {
            return;
        }
        test_signed_zeros_argminmax(SCALAR_STRATEGY, simd);
    }

    #[apply(simd_implementations)]
    fn test_nan_payloads<DType, T, SIMDV, SIMDM, const LANE_SIZE: usize>(
        #[case] simd: T,
        #[case] _dtype: DType,
        #[case] simd_available: bool,
    ) where
        DType: Half + SampleUniformFullRange,
        T: SIMDArgMinMax<DType, SIMDV, SIMDM, LANE_SIZE, SCALAR<FloatReturnNaN>>,
        SIMDV: Copy,
        SIMDM: Copy,
    {
        if !simd_available {
            return;
        }
        test_nan_payloads_argminmax(SCALAR_STRATEGY, simd);
    }

    #[apply(simd_implementations)]
    fn test_return_same_result_masked<DType, T, SIMDV, SIMDM, const LANE_SIZE: usize>(
        #[case] simd: T,
        #[case] _dtype: DType,
        #[case] simd_available: bool,
    ) where
        DType: Half + SampleUniformFullRange,
        T: SIMDArgMinMax<DType, SIMDV, SIMDM, LANE_SIZE, SCALAR<FloatReturnNaN>>,
        SIMDV: Copy,
        SIMDM: Copy,
    {
        if !simd_available {
            return;
        }
        test_return_same_result_masked_argminmax(DType::get_random_array, SCALAR_STRATEGY, simd);
    }

    #[apply(simd_implementations)]
    fn test_no_overflow_masked<DType, T, SIMDV, SIMDM, const LANE_SIZE: usize>(
        #[case] simd: T,
        #[case] _dtype: DType,
        #[case] simd_available: bool,
    ) where
        DType: Half + SampleUniformFullRange,
        T: SIMDArgMinMax<DType, SIMDV, SIMDM, LANE_SIZE, SCALAR<FloatReturnNaN>>,
        SIMDV: Copy,
        SIMDM: Copy,
    {
        if !simd_available {
            return;
        }
        test_no_overflow_masked_argminmax(DType::get_random_array, SCALAR_STRATEGY, simd, None);
    }

    #[apply(simd_implementations)]
    fn test_nans_masked<DType, T, SIMDV, SIMDM, const LANE_SIZE: usize>(
        #[case] simd: T,
        #[case] _dtype: DType,
        #[case] simd_available: bool,
    ) where
        DType: Half + SampleUniformFullRange,
        T: SIMDArgMinMax<DType, SIMDV, SIMDM, LANE_SIZE, SCALAR<FloatReturnNaN>>,
        SIMDV: Copy,
        SIMDM: Copy,
    {
        if !simd_available {
            return;
        }
        test_nans_masked_argminmax(DType::get_random_array, SCALAR_STRATEGY, simd);
    }
    #[apply(simd_implementations)]
    fn test_adversarial_masked<DType, T, SIMDV, SIMDM, const LANE_SIZE: usize>(
        #[case] simd: T,
        #[case] _dtype: DType,
        #[case] simd_available: bool,
    ) where
        DType: Half + SampleUniformFullRange,
        T: SIMDArgMinMax<DType, SIMDV, SIMDM, LANE_SIZE, SCALAR<FloatReturnNaN>>,
        SIMDV: Copy,
        SIMDM: Copy,
    {
        if !simd_available {
            return;
        }
        test_adversarial_masked_argminmax(SCALAR_STRATEGY, simd, true);
    }
}

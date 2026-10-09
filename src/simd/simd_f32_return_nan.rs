//! Implementation of the argminmax operations for f32 where NaN values take precedence.
//! This implementation returns the index of the first* NaN value if any are present,
//! otherwise it returns the index of the minimum and maximum values.
//!
//! To serve this functionality we transform the f32 values to ordinal i32 values, by
//! applying the sign bit to the magnitude (the other bits):
//!     ord_i32 = v               if v >= 0 (as i32)
//!     ord_i32 = i32::MIN - v    otherwise (i.e., minus the magnitude)
//!
//! This transformation is reversible (with the same formula), except that -0.0 and 0.0
//! are both mapped to 0 (as these are equal).
//!
//! Through this transformation we can perform the argminmax operations on the ordinal
//! integer values and then transform the result back to the original f32 values.
//! This transformation is necessary because comparisons with NaN values are always false.
//! So unless we perform ! <=  as gt and ! >=  as lt the argminmax operations will not
//! add NaN values to the accumulating SIMD register. And as le and ge are significantly
//! more expensive than lt and gt we use this efficient bitwise transformation.
//!
//! Also comparing integers is faster than comparing floats:
//!   - <https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_cmp_ps&ig_expand=902>
//!   - <https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_cmpgt_epi32&ig_expand=1084>
//!
//!
//! ---
//!
//! *Note: the first NaN value is only returned iff all NaN values have the same bit
//! representation. When NaN values have different bit representations then the index of
//! the highest / lowest ord_i32 is returned for the
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
use super::generic::{impl_SIMDInit_FloatReturnNaN, SIMDArgMinMax, SIMDInit, SIMDMasked, SIMDOps};
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "arm", feature = "nightly_simd"),
    target_arch = "aarch64",
))]
use crate::SCALAR;
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(all(target_arch = "arm", feature = "nightly_simd"))]
use std::arch::arm::*;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// The dtype-strategy for performing operations on f32 data: return NaN index
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
use super::task::{max_index_value, min_index_value};

#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "arm", feature = "nightly_simd"),
    target_arch = "aarch64",
))]
#[inline(always)]
fn _i32ord_to_f32(ord_i32: i32) -> f32 {
    // The same formula as the transformation (0.0 is returned for -0.0)
    let v = if ord_i32 < 0 {
        i32::MIN - ord_i32
    } else {
        ord_i32
    };
    f32::from_bits(v as u32)
}

#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "arm", feature = "nightly_simd"),
    target_arch = "aarch64",
))]
const MAX_INDEX: usize = i32::MAX as usize;

// --------------------------------------- AVX2 ----------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx2 {
    use super::super::config::AVX2;
    use super::*;

    const LANE_SIZE: usize = AVX2::<FloatReturnNaN>::LANE_SIZE_32;
    const LOWER_31_MASK: __m256i = unsafe { std::mem::transmute([i32::MAX; LANE_SIZE]) };

    #[inline(always)]
    unsafe fn _f32_as_m256i_to_i32ord(f32_as_m256i: __m256i) -> __m256i {
        // on a scalar: v if v >= 0 else i32::MIN - v (-0.0 and 0.0 are both 0)
        _mm256_sign_epi32(_mm256_and_si256(f32_as_m256i, LOWER_31_MASK), f32_as_m256i)
    }

    #[inline(always)]
    unsafe fn _reg_to_i32_arr(reg: __m256i) -> [i32; LANE_SIZE] {
        std::mem::transmute::<__m256i, [i32; LANE_SIZE]>(reg)
    }

    impl SIMDOps<f32, __m256i, __m256i, LANE_SIZE> for AVX2<FloatReturnNaN> {
        const INITIAL_INDEX: __m256i =
            unsafe { std::mem::transmute([0i32, 1i32, 2i32, 3i32, 4i32, 5i32, 6i32, 7i32]) };
        const INDEX_INCREMENT: __m256i =
            unsafe { std::mem::transmute([LANE_SIZE as i32; LANE_SIZE]) };
        const MAX_INDEX: usize = MAX_INDEX;

        #[inline(always)]
        unsafe fn _reg_to_arr(_: __m256i) -> [f32; LANE_SIZE] {
            // Not implemented because we will perform the horizontal operations on the
            // signed integer values instead of trying to retransform **only** the values
            // (and thus not the indices) to floats.
            unimplemented!()
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const f32) -> __m256i {
            _f32_as_m256i_to_i32ord(_mm256_loadu_si256(data as *const __m256i))
        }

        #[inline(always)]
        unsafe fn _mm_add(a: __m256i, b: __m256i) -> __m256i {
            _mm256_add_epi32(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: __m256i, b: __m256i) -> __m256i {
            _mm256_cmpgt_epi32(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: __m256i, b: __m256i) -> __m256i {
            _mm256_cmpgt_epi32(b, a)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: __m256i, b: __m256i, mask: __m256i) -> __m256i {
            _mm256_blendv_epi8(a, b, mask)
        }

        #[inline(always)]
        unsafe fn _horiz_min(index: __m256i, value: __m256i) -> (usize, f32) {
            let index_arr: [i32; LANE_SIZE] = _reg_to_i32_arr(index);
            let value_arr: [i32; LANE_SIZE] = _reg_to_i32_arr(value);
            let (min_index, min_value) = min_index_value(&index_arr, &value_arr);
            (min_index as usize, _i32ord_to_f32(min_value))
        }

        #[inline(always)]
        unsafe fn _horiz_max(index: __m256i, value: __m256i) -> (usize, f32) {
            let index_arr: [i32; LANE_SIZE] = _reg_to_i32_arr(index);
            let value_arr: [i32; LANE_SIZE] = _reg_to_i32_arr(value);
            let (max_index, max_value) = max_index_value(&index_arr, &value_arr);
            (max_index as usize, _i32ord_to_f32(max_value))
        }
    }

    impl_SIMDInit_FloatReturnNaN!(f32, __m256i, __m256i, LANE_SIZE, AVX2<FloatReturnNaN>);

    impl SIMDArgMinMax<f32, __m256i, __m256i, LANE_SIZE, SCALAR<FloatReturnNaN>>
        for AVX2<FloatReturnNaN>
    {
        #[target_feature(enable = "avx2")]
        unsafe fn argminmax(data: &[f32]) -> (usize, usize) {
            Self::_argminmax(data)
        }

        unsafe fn argmin(data: &[f32]) -> usize {
            Self::argminmax(data).0
        }

        unsafe fn argmax(data: &[f32]) -> usize {
            Self::argminmax(data).1
        }

        #[target_feature(enable = "avx2")]
        unsafe fn argminmax_masked(
            data: &[f32],
            validity: &[u8],
            offset: usize,
        ) -> Option<(usize, usize)> {
            Self::_argminmax_masked(data, validity, offset)
        }
    }
}

// ---------------------------------------- SSE ----------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod sse {
    use super::super::config::SSE;
    use super::*;

    const LANE_SIZE: usize = SSE::<FloatReturnNaN>::LANE_SIZE_32;
    const LOWER_31_MASK: __m128i = unsafe { std::mem::transmute([i32::MAX; LANE_SIZE]) };

    #[inline(always)]
    unsafe fn _f32_as_m128i_to_i32ord(f32_as_m128i: __m128i) -> __m128i {
        // on a scalar: v if v >= 0 else i32::MIN - v (-0.0 and 0.0 are both 0)
        _mm_sign_epi32(_mm_and_si128(f32_as_m128i, LOWER_31_MASK), f32_as_m128i)
    }

    #[inline(always)]
    unsafe fn _reg_to_i32_arr(reg: __m128i) -> [i32; LANE_SIZE] {
        std::mem::transmute::<__m128i, [i32; LANE_SIZE]>(reg)
    }

    impl SIMDOps<f32, __m128i, __m128i, LANE_SIZE> for SSE<FloatReturnNaN> {
        const INITIAL_INDEX: __m128i = unsafe { std::mem::transmute([0i32, 1i32, 2i32, 3i32]) };
        const INDEX_INCREMENT: __m128i =
            unsafe { std::mem::transmute([LANE_SIZE as i32; LANE_SIZE]) };
        const MAX_INDEX: usize = MAX_INDEX;

        #[inline(always)]
        unsafe fn _reg_to_arr(_: __m128i) -> [f32; LANE_SIZE] {
            // Not implemented because we will perform the horizontal operations on the
            // signed integer values instead of trying to retransform **only** the values
            // (and thus not the indices) to floats.
            unimplemented!()
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const f32) -> __m128i {
            _f32_as_m128i_to_i32ord(_mm_loadu_si128(data as *const __m128i))
        }

        #[inline(always)]
        unsafe fn _mm_add(a: __m128i, b: __m128i) -> __m128i {
            _mm_add_epi32(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: __m128i, b: __m128i) -> __m128i {
            _mm_cmpgt_epi32(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: __m128i, b: __m128i) -> __m128i {
            _mm_cmplt_epi32(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: __m128i, b: __m128i, mask: __m128i) -> __m128i {
            _mm_blendv_epi8(a, b, mask)
        }

        #[inline(always)]
        unsafe fn _horiz_min(index: __m128i, value: __m128i) -> (usize, f32) {
            let index_arr: [i32; LANE_SIZE] = _reg_to_i32_arr(index);
            let value_arr: [i32; LANE_SIZE] = _reg_to_i32_arr(value);
            let (min_index, min_value) = min_index_value(&index_arr, &value_arr);
            (min_index as usize, _i32ord_to_f32(min_value))
        }

        #[inline(always)]
        unsafe fn _horiz_max(index: __m128i, value: __m128i) -> (usize, f32) {
            let index_arr: [i32; LANE_SIZE] = _reg_to_i32_arr(index);
            let value_arr: [i32; LANE_SIZE] = _reg_to_i32_arr(value);
            let (max_index, max_value) = max_index_value(&index_arr, &value_arr);
            (max_index as usize, _i32ord_to_f32(max_value))
        }
    }

    impl_SIMDInit_FloatReturnNaN!(f32, __m128i, __m128i, LANE_SIZE, SSE<FloatReturnNaN>);

    impl SIMDArgMinMax<f32, __m128i, __m128i, LANE_SIZE, SCALAR<FloatReturnNaN>>
        for SSE<FloatReturnNaN>
    {
        #[target_feature(enable = "sse4.1")]
        unsafe fn argminmax(data: &[f32]) -> (usize, usize) {
            Self::_argminmax(data)
        }

        unsafe fn argmin(data: &[f32]) -> usize {
            Self::argminmax(data).0
        }

        unsafe fn argmax(data: &[f32]) -> usize {
            Self::argminmax(data).1
        }

        #[target_feature(enable = "sse4.1")]
        unsafe fn argminmax_masked(
            data: &[f32],
            validity: &[u8],
            offset: usize,
        ) -> Option<(usize, usize)> {
            Self::_argminmax_masked(data, validity, offset)
        }
    }
}

// -------------------------------------- AVX512 ---------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx512 {
    use super::super::config::AVX512;
    use super::*;

    const LANE_SIZE: usize = AVX512::<FloatReturnNaN>::LANE_SIZE_32;
    const SIGN_BIT: __m512i = unsafe { std::mem::transmute([i32::MIN; LANE_SIZE]) };

    #[inline(always)]
    unsafe fn _f32_as_m512i_to_i32ord(f32_as_m512i: __m512i) -> __m512i {
        // on a scalar: v if v >= 0 else i32::MIN - v (-0.0 and 0.0 are both 0)
        let negative = _mm512_cmplt_epi32_mask(f32_as_m512i, _mm512_setzero_si512());
        _mm512_mask_sub_epi32(f32_as_m512i, negative, SIGN_BIT, f32_as_m512i)
    }

    #[inline(always)]
    unsafe fn _reg_to_i32_arr(reg: __m512i) -> [i32; LANE_SIZE] {
        std::mem::transmute::<__m512i, [i32; LANE_SIZE]>(reg)
    }

    impl SIMDOps<f32, __m512i, u16, LANE_SIZE> for AVX512<FloatReturnNaN> {
        const INITIAL_INDEX: __m512i = unsafe {
            std::mem::transmute([
                0i32, 1i32, 2i32, 3i32, 4i32, 5i32, 6i32, 7i32, 8i32, 9i32, 10i32, 11i32, 12i32,
                13i32, 14i32, 15i32,
            ])
        };
        const INDEX_INCREMENT: __m512i =
            unsafe { std::mem::transmute([LANE_SIZE as i32; LANE_SIZE]) };
        const MAX_INDEX: usize = MAX_INDEX;

        #[inline(always)]
        unsafe fn _reg_to_arr(_: __m512i) -> [f32; LANE_SIZE] {
            // Not implemented because we will perform the horizontal operations on the
            // signed integer values instead of trying to retransform **only** the values
            // (and thus not the indices) to floats.
            unimplemented!()
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const f32) -> __m512i {
            _f32_as_m512i_to_i32ord(_mm512_loadu_si512(data as *const _))
        }

        #[inline(always)]
        unsafe fn _mm_add(a: __m512i, b: __m512i) -> __m512i {
            _mm512_add_epi32(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: __m512i, b: __m512i) -> u16 {
            _mm512_cmpgt_epi32_mask(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: __m512i, b: __m512i) -> u16 {
            _mm512_cmplt_epi32_mask(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: __m512i, b: __m512i, mask: u16) -> __m512i {
            _mm512_mask_blend_epi32(mask, a, b)
        }

        #[inline(always)]
        unsafe fn _horiz_min(index: __m512i, value: __m512i) -> (usize, f32) {
            let index_arr: [i32; LANE_SIZE] = _reg_to_i32_arr(index);
            let value_arr: [i32; LANE_SIZE] = _reg_to_i32_arr(value);
            let (min_index, min_value) = min_index_value(&index_arr, &value_arr);
            (min_index as usize, _i32ord_to_f32(min_value))
        }

        #[inline(always)]
        unsafe fn _horiz_max(index: __m512i, value: __m512i) -> (usize, f32) {
            let index_arr: [i32; LANE_SIZE] = _reg_to_i32_arr(index);
            let value_arr: [i32; LANE_SIZE] = _reg_to_i32_arr(value);
            let (max_index, max_value) = max_index_value(&index_arr, &value_arr);
            (max_index as usize, _i32ord_to_f32(max_value))
        }
    }

    impl_SIMDInit_FloatReturnNaN!(f32, __m512i, u16, LANE_SIZE, AVX512<FloatReturnNaN>);

    impl SIMDArgMinMax<f32, __m512i, u16, LANE_SIZE, SCALAR<FloatReturnNaN>>
        for AVX512<FloatReturnNaN>
    {
        #[target_feature(enable = "avx512f")]
        unsafe fn argminmax(data: &[f32]) -> (usize, usize) {
            Self::_argminmax(data)
        }

        unsafe fn argmin(data: &[f32]) -> usize {
            Self::argminmax(data).0
        }

        unsafe fn argmax(data: &[f32]) -> usize {
            Self::argminmax(data).1
        }

        #[target_feature(enable = "avx512f")]
        unsafe fn argminmax_masked(
            data: &[f32],
            validity: &[u8],
            offset: usize,
        ) -> Option<(usize, usize)> {
            Self::_argminmax_masked(data, validity, offset)
        }
    }
}

// --------------------------------------- NEON ----------------------------------------

#[cfg(any(
    all(target_arch = "arm", feature = "nightly_simd"),
    target_arch = "aarch64" // stable for AArch64
))]
mod neon {
    use super::super::config::NEON;
    use super::*;

    const LANE_SIZE: usize = NEON::<FloatReturnNaN>::LANE_SIZE_32;
    const SIGN_BIT: int32x4_t = unsafe { std::mem::transmute([i32::MIN; LANE_SIZE]) };

    #[inline(always)]
    unsafe fn _f32_as_int32x4_to_i32ord(f32_as_int32x4: int32x4_t) -> int32x4_t {
        // on a scalar: v if v >= 0 else i32::MIN - v (-0.0 and 0.0 are both 0)
        // = abs(v) ^ (v & i32::MIN), as abs wraps around (abs(i32::MIN) = i32::MIN)
        veorq_s32(
            vabsq_s32(f32_as_int32x4),
            vandq_s32(f32_as_int32x4, SIGN_BIT),
        )
    }

    #[inline(always)]
    unsafe fn _reg_to_i32_arr(reg: int32x4_t) -> [i32; LANE_SIZE] {
        std::mem::transmute::<int32x4_t, [i32; LANE_SIZE]>(reg)
    }

    impl SIMDOps<f32, int32x4_t, uint32x4_t, LANE_SIZE> for NEON<FloatReturnNaN> {
        const INITIAL_INDEX: int32x4_t = unsafe { std::mem::transmute([0i32, 1i32, 2i32, 3i32]) };
        const INDEX_INCREMENT: int32x4_t =
            unsafe { std::mem::transmute([LANE_SIZE as i32; LANE_SIZE]) };
        const MAX_INDEX: usize = MAX_INDEX;

        #[inline(always)]
        unsafe fn _reg_to_arr(_: int32x4_t) -> [f32; LANE_SIZE] {
            // Not implemented because we will perform the horizontal operations on the
            // signed integer values instead of trying to retransform **only** the values
            // (and thus not the indices) to floats.
            unimplemented!()
        }

        #[inline(always)]
        unsafe fn _mm_loadu(data: *const f32) -> int32x4_t {
            _f32_as_int32x4_to_i32ord(vld1q_s32(data as *const i32))
        }

        #[inline(always)]
        unsafe fn _mm_add(a: int32x4_t, b: int32x4_t) -> int32x4_t {
            vaddq_s32(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmpgt(a: int32x4_t, b: int32x4_t) -> uint32x4_t {
            vcgtq_s32(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_cmplt(a: int32x4_t, b: int32x4_t) -> uint32x4_t {
            vcltq_s32(a, b)
        }

        #[inline(always)]
        unsafe fn _mm_blendv(a: int32x4_t, b: int32x4_t, mask: uint32x4_t) -> int32x4_t {
            vbslq_s32(mask, b, a)
        }

        #[inline(always)]
        unsafe fn _horiz_min(index: int32x4_t, value: int32x4_t) -> (usize, f32) {
            let index_arr: [i32; LANE_SIZE] = _reg_to_i32_arr(index);
            let value_arr: [i32; LANE_SIZE] = _reg_to_i32_arr(value);
            let (min_index, min_value) = min_index_value(&index_arr, &value_arr);
            (min_index as usize, _i32ord_to_f32(min_value))
        }

        #[inline(always)]
        unsafe fn _horiz_max(index: int32x4_t, value: int32x4_t) -> (usize, f32) {
            let index_arr: [i32; LANE_SIZE] = _reg_to_i32_arr(index);
            let value_arr: [i32; LANE_SIZE] = _reg_to_i32_arr(value);
            let (max_index, max_value) = max_index_value(&index_arr, &value_arr);
            (max_index as usize, _i32ord_to_f32(max_value))
        }
    }

    impl_SIMDInit_FloatReturnNaN!(f32, int32x4_t, uint32x4_t, LANE_SIZE, NEON<FloatReturnNaN>);

    impl SIMDArgMinMax<f32, int32x4_t, uint32x4_t, LANE_SIZE, SCALAR<FloatReturnNaN>>
        for NEON<FloatReturnNaN>
    {
        #[target_feature(enable = "neon")]
        unsafe fn argminmax(data: &[f32]) -> (usize, usize) {
            Self::_argminmax(data)
        }

        unsafe fn argmin(data: &[f32]) -> usize {
            Self::argminmax(data).0
        }

        unsafe fn argmax(data: &[f32]) -> usize {
            Self::argminmax(data).1
        }

        #[target_feature(enable = "neon")]
        unsafe fn argminmax_masked(
            data: &[f32],
            validity: &[u8],
            offset: usize,
        ) -> Option<(usize, usize)> {
            Self::_argminmax_masked(data, validity, offset)
        }
    }
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

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    use crate::simd::config::AVX512;
    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    use crate::simd::config::NEON;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    use crate::simd::config::{AVX2, SSE};
    use crate::{FloatReturnNaN, SIMDArgMinMax, SCALAR};

    use super::super::test_utils::{
        test_first_index_identical_values_argminmax, test_return_same_result_argminmax,
    };
    // Float specific tests
    use super::super::test_utils::{
        test_return_infs_argminmax, test_return_nans_argminmax, test_signed_zeros_argminmax,
    };
    // Masked tests
    use super::super::test_utils::{
        test_adversarial_masked_argminmax, test_nans_masked_argminmax,
        test_return_same_result_masked_argminmax,
    };

    use dev_utils::utils;

    fn get_array_f32(n: usize) -> Vec<f32> {
        utils::SampleUniformFullRange::get_random_array(n)
    }

    // The scalar implementation
    const SCALAR_STRATEGY: SCALAR<FloatReturnNaN> = SCALAR {
        _dtype_strategy: PhantomData::<FloatReturnNaN>,
    };

    // ------------ Template for x86 / x86_64 -------------

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[template]
    #[rstest]
    #[case::sse(SSE {_dtype_strategy: PhantomData::<FloatReturnNaN>}, is_x86_feature_detected!("sse4.1"))]
    #[case::avx2(AVX2 {_dtype_strategy: PhantomData::<FloatReturnNaN>}, is_x86_feature_detected!("avx2"))]
    #[case::avx512(AVX512 {_dtype_strategy: PhantomData::<FloatReturnNaN>}, is_x86_feature_detected!("avx512f"))]
    fn simd_implementations<T, SIMDV, SIMDM, const LANE_SIZE: usize>(
        #[case] simd: T,
        #[case] simd_available: bool,
    ) {
    }

    // ------------ Template for ARM / AArch64 ------------

    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    #[template]
    #[rstest]
    #[case::neon(NEON { _dtype_strategy: PhantomData::<FloatReturnNaN>}, true)]
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
        T: SIMDArgMinMax<f32, SIMDV, SIMDM, LANE_SIZE, SCALAR<FloatReturnNaN>>,
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
        T: SIMDArgMinMax<f32, SIMDV, SIMDM, LANE_SIZE, SCALAR<FloatReturnNaN>>,
        SIMDV: Copy,
        SIMDM: Copy,
    {
        if !simd_available {
            return;
        }
        test_return_same_result_argminmax(get_array_f32, SCALAR_STRATEGY, simd);
    }

    #[apply(simd_implementations)]
    fn test_return_infs<T, SIMDV, SIMDM, const LANE_SIZE: usize>(
        #[case] simd: T,
        #[case] simd_available: bool,
    ) where
        T: SIMDArgMinMax<f32, SIMDV, SIMDM, LANE_SIZE, SCALAR<FloatReturnNaN>>,
        SIMDV: Copy,
        SIMDM: Copy,
    {
        if !simd_available {
            return;
        }
        test_return_infs_argminmax(get_array_f32, SCALAR_STRATEGY, simd);
    }

    #[apply(simd_implementations)]
    fn test_signed_zeros<T, SIMDV, SIMDM, const LANE_SIZE: usize>(
        #[case] simd: T,
        #[case] simd_available: bool,
    ) where
        T: SIMDArgMinMax<f32, SIMDV, SIMDM, LANE_SIZE, SCALAR<FloatReturnNaN>>,
        SIMDV: Copy,
        SIMDM: Copy,
    {
        if !simd_available {
            return;
        }
        test_signed_zeros_argminmax(SCALAR_STRATEGY, simd);
    }

    #[apply(simd_implementations)]
    fn test_return_nans<T, SIMDV, SIMDM, const LANE_SIZE: usize>(
        #[case] simd: T,
        #[case] simd_available: bool,
    ) where
        T: SIMDArgMinMax<f32, SIMDV, SIMDM, LANE_SIZE, SCALAR<FloatReturnNaN>>,
        SIMDV: Copy,
        SIMDM: Copy,
    {
        if !simd_available {
            return;
        }
        test_return_nans_argminmax(get_array_f32, SCALAR_STRATEGY, simd);
    }

    #[apply(simd_implementations)]
    fn test_return_same_result_masked<T, SIMDV, SIMDM, const LANE_SIZE: usize>(
        #[case] simd: T,
        #[case] simd_available: bool,
    ) where
        T: SIMDArgMinMax<f32, SIMDV, SIMDM, LANE_SIZE, SCALAR<FloatReturnNaN>>,
        SIMDV: Copy,
        SIMDM: Copy,
    {
        if !simd_available {
            return;
        }
        test_return_same_result_masked_argminmax(get_array_f32, SCALAR_STRATEGY, simd);
    }

    #[apply(simd_implementations)]
    fn test_nans_masked<T, SIMDV, SIMDM, const LANE_SIZE: usize>(
        #[case] simd: T,
        #[case] simd_available: bool,
    ) where
        T: SIMDArgMinMax<f32, SIMDV, SIMDM, LANE_SIZE, SCALAR<FloatReturnNaN>>,
        SIMDV: Copy,
        SIMDM: Copy,
    {
        if !simd_available {
            return;
        }
        test_nans_masked_argminmax(get_array_f32, SCALAR_STRATEGY, simd);
    }
    #[apply(simd_implementations)]
    fn test_adversarial_masked<T, SIMDV, SIMDM, const LANE_SIZE: usize>(
        #[case] simd: T,
        #[case] simd_available: bool,
    ) where
        T: SIMDArgMinMax<f32, SIMDV, SIMDM, LANE_SIZE, SCALAR<FloatReturnNaN>>,
        SIMDV: Copy,
        SIMDM: Copy,
    {
        if !simd_available {
            return;
        }
        test_adversarial_masked_argminmax(SCALAR_STRATEGY, simd, true);
    }
}

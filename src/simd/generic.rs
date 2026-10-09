use num_traits::{AsPrimitive, Bounded};

use super::task::*;
use super::validity::SIMDValidity;
use crate::scalar::ScalarArgMinMax;
use crate::validity::{assert_validity_len, validity_word, FoundMinMax};

// ---------------------------------- SIMD operations ----------------------------------

/// Raw SIMD operations
/// These operations are used by the SIMD algorithm and have to be implemented for each
/// data type - SIMD instruction set combination.
/// The operations are implemented in the `simd_*.rs` files.
///
/// Note that for floating point dataypes two implementations are required:
/// - one for the ignore NaN case (uses a floating point SIMDVecDtype)
///   (see the `simd_f*_ignore_nan.rs` files)
/// - one for the return NaN case (uses an integer SIMDVecDtype - as we use the
///   ord_transform to view the floating point data as ordinal integer data).
///   (see the `simd_f*_return_nan.rs` files)
///
#[doc(hidden)]
pub trait SIMDOps<ScalarDType, SIMDVecDtype, SIMDMaskDtype, const LANE_SIZE: usize>
where
    ScalarDType: Copy + PartialOrd + AsPrimitive<usize>,
    SIMDVecDtype: Copy,
    SIMDMaskDtype: Copy,
{
    /// Integers > this value **cannot** be accurately represented in SIMDVecDtype
    const MAX_INDEX: usize;
    /// Initial index value for the SIMD vector
    const INITIAL_INDEX: SIMDVecDtype;
    /// Increment value for the SIMD vector
    const INDEX_INCREMENT: SIMDVecDtype;
    /// Whether `SIMDCore::_core_argminmax` reduces each group of vectors to one
    /// candidate before comparing it with the running min / max. Disable this where
    /// measurements show that the groups do not help, e.g., when the loop is limited by
    /// its throughput rather than by that comparison, or when the overflow-safe loop
    /// restarts after a few vectors.
    /// This is a setting of the algorithm rather than an operation, but it is defined
    /// here as it depends on both the instruction set and the data type.
    const GROUP_VECTORS: bool = true;

    /// Convert a SIMD register to array
    unsafe fn _reg_to_arr(reg: SIMDVecDtype) -> [ScalarDType; LANE_SIZE];

    /// Load a SIMD register from memory
    unsafe fn _mm_loadu(data: *const ScalarDType) -> SIMDVecDtype;

    /// Add two SIMD registers
    unsafe fn _mm_add(a: SIMDVecDtype, b: SIMDVecDtype) -> SIMDVecDtype;

    /// Compare two SIMD registers for greater-than (gt): a > b
    /// Returns a SIMD mask
    /// For FloatIgnoreNaN, `b` must contain no NaNs; NaN lanes in `a` return false.
    unsafe fn _mm_cmpgt(a: SIMDVecDtype, b: SIMDVecDtype) -> SIMDMaskDtype;

    /// Compare two SIMD registers for less-than (lt): a < b
    /// For FloatIgnoreNaN, `b` must contain no NaNs; NaN lanes in `a` return false.
    unsafe fn _mm_cmplt(a: SIMDVecDtype, b: SIMDVecDtype) -> SIMDMaskDtype;

    /// Blend two SIMD registers using a SIMD mask (selects elements from a or b)
    unsafe fn _mm_blendv(a: SIMDVecDtype, b: SIMDVecDtype, mask: SIMDMaskDtype) -> SIMDVecDtype;

    /// Horizontal min: get the minimum value from the value SIMD register and its
    /// corresponding index from the index SIMD register
    #[inline(always)]
    unsafe fn _horiz_min(index: SIMDVecDtype, value: SIMDVecDtype) -> (usize, ScalarDType) {
        // This becomes the bottleneck when using 8-bit data types, as for  every 2**7
        // or 2**8 elements, the SIMD inner loop is executed (& thus also terminated)
        // to avoid overflow.
        // To tackle this bottleneck, we use a different approach for 8-bit data types:
        // -> we overwrite this method to perform (in SIMD) the horizontal min
        //    see: https://stackoverflow.com/a/9798369
        // Note: this is not a bottleneck for 16-bit data types, as the termination of
        // the SIMD inner loop is 2**8 times less frequent.
        let index_arr = Self::_reg_to_arr(index);
        let value_arr = Self::_reg_to_arr(value);
        let (min_index, min_value) = min_index_value(&index_arr, &value_arr);
        (min_index.as_(), min_value)
    }

    /// Horizontal max: get the maximum value from the value SIMD register and its
    /// corresponding index from the index SIMD register
    #[inline(always)]
    unsafe fn _horiz_max(index: SIMDVecDtype, value: SIMDVecDtype) -> (usize, ScalarDType) {
        // This becomes the bottleneck when using 8-bit data types, as for  every 2**7
        // or 2**8 elements, the SIMD inner loop is executed (& thus also terminated)
        // to avoid overflow.
        // To tackle this bottleneck, we use a different approach for 8-bit data types:
        // -> we overwrite this method to perform (in SIMD) the horizontal max
        //    see: https://stackoverflow.com/a/9798369
        // Note: this is not a bottleneck for 16-bit data types, as the termination of
        // the SIMD inner loop is 2**8 times less frequent.
        let index_arr = Self::_reg_to_arr(index);
        let value_arr = Self::_reg_to_arr(value);
        let (max_index, max_value) = max_index_value(&index_arr, &value_arr);
        (max_index.as_(), max_value)
    }

    /// Get the largest multiple of LANE_SIZE that is <= MAX_INDEX
    #[inline(always)]
    fn _get_overflow_lane_size_limit() -> usize {
        Self::MAX_INDEX - Self::MAX_INDEX % LANE_SIZE
    }

    // ----------------- SIMD operations necessary for ignoring NaNs ------------------

    #[inline(always)]
    unsafe fn _mm_set1(_value: ScalarDType) -> SIMDVecDtype {
        // This is a dummy method that is only used for the ignore NaN case.
        // For the Integer and Float return NaN case, this method is not used.
        unreachable!()
    }
}

/// SIMD initialization operations
/// These operations are used by the SIMD algorithm and have to be implemented for each
/// data type - SIMD instruction set combination.
///
/// These operations are implemented in the `simd_*.rs` files through calling one of the
/// three macros below:
/// - `impl_SIMDInit_Int!`
///     - called in the `simd_i*.rs` files
///     - called in the `simd_u*.rs` files
/// - `impl_SIMDInit_FloatReturnNaN!`
///     - see the `simd_f*_return_nan.rs` files
/// - `impl_SIMDInit_FloatIgnoreNaN!`
///     - see the `simd_f*_ignore_nan.rs` files
///
/// The current (default) implementation is for the Int case - see `impl_SIMDInit_Int!`
/// macro below for more details.
/// For the Float Return NaN case,only the _return_check method is changed - see
/// `impl_SIMDInit_FloatReturnNaN!` macro below for more details.
/// For the Float Ignore NaN case, all the initialization methods are changed - see
/// `impl_SIMDInit_FloatIgnoreNaN!` macro below for more details. Note that for this
/// case, the SIMDOps should implement the `_mm_set1` method.
///
#[doc(hidden)]
pub trait SIMDInit<ScalarDType, SIMDVecDtype, SIMDMaskDtype, const LANE_SIZE: usize>:
    SIMDOps<ScalarDType, SIMDVecDtype, SIMDMaskDtype, LANE_SIZE>
where
    ScalarDType: Copy + PartialOrd + AsPrimitive<usize>,
    SIMDVecDtype: Copy,
    SIMDMaskDtype: Copy,
{
    const IGNORE_NAN: bool = false;

    // Initialization for _core_argminmax

    #[inline(always)]
    unsafe fn _initialize_index_values_low(
        arr_ptr: *const ScalarDType,
    ) -> (SIMDVecDtype, SIMDVecDtype) {
        // Initialize the index and value SIMD registers
        (Self::INITIAL_INDEX, Self::_mm_loadu(arr_ptr))
    }

    #[inline(always)]
    unsafe fn _initialize_index_values_high(
        arr_ptr: *const ScalarDType,
    ) -> (SIMDVecDtype, SIMDVecDtype) {
        // Initialize the index and value SIMD registers
        (Self::INITIAL_INDEX, Self::_mm_loadu(arr_ptr))
    }

    // Initialization for _overflow_safe_core_argminmax

    #[inline(always)]
    fn _initialize_min_value(arr: &[ScalarDType]) -> ScalarDType {
        unsafe { *arr.get_unchecked(0) }
    }

    #[inline(always)]
    fn _initialize_max_value(arr: &[ScalarDType]) -> ScalarDType {
        unsafe { *arr.get_unchecked(0) }
    }

    // Checks

    /// Return case for the algorithm
    #[inline(always)]
    fn _return_check(_v: ScalarDType) -> bool {
        false
    }

    /// Check if the value is NaN
    #[inline(always)]
    fn _nan_check(_v: ScalarDType) -> bool {
        false
    }
}

// --------------- Int (signed and unsigned)

#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "arm", feature = "nightly_simd"),
    target_arch = "aarch64",
))]
macro_rules! impl_SIMDInit_Int {
    ($scalar_dtype:ty, $simd_vec_dtype:ty, $simd_mask_dtype:ty, $lane_size:expr, $simd_struct:ty) => {
        impl SIMDInit<$scalar_dtype, $simd_vec_dtype, $simd_mask_dtype, $lane_size>
            for $simd_struct
        {
            // Use the default implementation
        }
    };
}

#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "arm", feature = "nightly_simd"),
    target_arch = "aarch64",
))]
pub(crate) use impl_SIMDInit_Int; // Now classic paths Just Work™

// --------------- Float Return NaNs

#[cfg(any(feature = "float", feature = "half"))]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "arm", feature = "nightly_simd"),
    target_arch = "aarch64",
))]
macro_rules! impl_SIMDInit_FloatReturnNaN {
    ($scalar_dtype:ty, $simd_vec_dtype:ty, $simd_mask_dtype:ty, $lane_size:expr, $simd_struct:ty) => {
        impl SIMDInit<$scalar_dtype, $simd_vec_dtype, $simd_mask_dtype, $lane_size>
            for $simd_struct
        {
            // Use all initialization methods from the default implementation

            /// Return when a NaN is found
            #[inline(always)]
            fn _return_check(v: $scalar_dtype) -> bool {
                v.is_nan()
            }

            #[inline(always)]
            fn _nan_check(v: $scalar_dtype) -> bool {
                v.is_nan()
            }
        }
    };
}

#[cfg(any(feature = "float", feature = "half"))]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "arm", feature = "nightly_simd"),
    target_arch = "aarch64",
))]
pub(crate) use impl_SIMDInit_FloatReturnNaN; // Now classic paths Just Work™

// --------------- Float Ignore NaNs

#[cfg(any(feature = "float", feature = "half"))]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "arm", feature = "nightly_simd"),
    target_arch = "aarch64",
))]
macro_rules! impl_SIMDInit_FloatIgnoreNaN {
    ($scalar_dtype:ty, $simd_vec_dtype:ty, $simd_mask_dtype:ty, $lane_size:expr, $simd_struct:ty) => {
        impl SIMDInit<$scalar_dtype, $simd_vec_dtype, $simd_mask_dtype, $lane_size>
            for $simd_struct
        {
            // Use the _return_check method from the default implementation

            const IGNORE_NAN: bool = true;

            #[inline(always)]
            unsafe fn _initialize_index_values_low(
                arr_ptr: *const $scalar_dtype,
            ) -> ($simd_vec_dtype, $simd_vec_dtype) {
                // Initialize the index and value SIMD registers
                let new_values = Self::_mm_loadu(arr_ptr);
                let mask_low =
                    Self::_mm_cmplt(new_values, Self::_mm_set1(<$scalar_dtype>::INFINITY));
                let values_low = Self::_mm_blendv(
                    Self::_mm_set1(<$scalar_dtype>::INFINITY),
                    new_values,
                    mask_low,
                );
                let index_low = Self::_mm_blendv(
                    Self::_mm_set1(<$scalar_dtype>::zero()),
                    Self::INITIAL_INDEX,
                    mask_low,
                );
                (index_low, values_low)
            }

            #[inline(always)]
            unsafe fn _initialize_index_values_high(
                arr_ptr: *const $scalar_dtype,
            ) -> ($simd_vec_dtype, $simd_vec_dtype) {
                // Initialize the index and value SIMD registers
                let new_values = Self::_mm_loadu(arr_ptr);
                let mask_high =
                    Self::_mm_cmpgt(new_values, Self::_mm_set1(<$scalar_dtype>::NEG_INFINITY));
                let values_high = Self::_mm_blendv(
                    Self::_mm_set1(<$scalar_dtype>::NEG_INFINITY),
                    new_values,
                    mask_high,
                );
                let index_high = Self::_mm_blendv(
                    Self::_mm_set1(<$scalar_dtype>::zero()),
                    Self::INITIAL_INDEX,
                    mask_high,
                );
                (index_high, values_high)
            }

            #[inline(always)]
            fn _initialize_min_value(_: &[$scalar_dtype]) -> $scalar_dtype {
                <$scalar_dtype>::INFINITY
            }

            #[inline(always)]
            fn _initialize_max_value(_: &[$scalar_dtype]) -> $scalar_dtype {
                <$scalar_dtype>::NEG_INFINITY
            }

            #[inline(always)]
            fn _nan_check(v: $scalar_dtype) -> bool {
                v.is_nan()
            }
        }
    };
}

#[cfg(any(feature = "float", feature = "half"))]
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "arm", feature = "nightly_simd"),
    target_arch = "aarch64",
))]
pub(crate) use impl_SIMDInit_FloatIgnoreNaN; // Now classic paths Just Work™

// ---------------------------------- SIMD algorithm -----------------------------------

/// Number of vectors that `SIMDCore::_core_argminmax` reduces to one candidate before
/// comparing it with the running min / max.
pub(crate) const VECTORS_PER_GROUP: usize = 4;

/// The SIMDCore trait (for all data types).
/// This trait contains the core of the argminmax algorithm.
///
/// This trait is auto-implemented below for all structs - iff the SIMDOps and the
/// SIMDInit traits are implemented for the struct
///
#[doc(hidden)]
pub trait SIMDCore<ScalarDType, SIMDVecDtype, SIMDMaskDtype, const LANE_SIZE: usize>:
    SIMDOps<ScalarDType, SIMDVecDtype, SIMDMaskDtype, LANE_SIZE>
    + SIMDInit<ScalarDType, SIMDVecDtype, SIMDMaskDtype, LANE_SIZE>
where
    ScalarDType: Copy + PartialOrd + AsPrimitive<usize>,
    SIMDVecDtype: Copy,
    SIMDMaskDtype: Copy,
{
    /// Core argminmax algorithm - returns (argmin, min, argmax, max)
    ///
    /// This method asserts:
    /// - the array length is a multiple of LANE_SIZE
    ///
    /// This method assumes:
    /// - the array length is <= MAX_INDEX
    ///
    /// Note that this method is not overflow safe, as it assumes that the array length
    /// is <= MAX_INDEX. The `_overflow_safe_core_argminmax` method is overflow safe.
    ///
    /// Comparing every vector with the running min / max makes each comparison wait for
    /// the previous one. Instead (unless `GROUP_VECTORS` is false), every group of
    /// `VECTORS_PER_GROUP` vectors is first reduced to one candidate (which does not
    /// depend on the running min / max), and only that candidate is compared with the
    /// running min / max. This allows the CPU to process the vectors of several groups
    /// at the same time.
    ///
    #[inline(always)]
    unsafe fn _core_argminmax(arr: &[ScalarDType]) -> (usize, ScalarDType, usize, ScalarDType) {
        assert_eq!(arr.len() % LANE_SIZE, 0);
        // Efficient calculation of argmin and argmax together

        let mut arr_ptr = arr.as_ptr(); // Array pointer we will increment in the loops
        let mut new_index = Self::INITIAL_INDEX; // Index we will increment in the loops
        let mut low = Self::_initialize_index_values_low(arr_ptr);
        let mut high = Self::_initialize_index_values_high(arr_ptr);

        let nb_vectors = arr.len() / LANE_SIZE - 1; // the vectors after the first one
        let nb_groups = if Self::GROUP_VECTORS {
            nb_vectors / VECTORS_PER_GROUP
        } else {
            0
        };
        // The index increment from one group to the next. Adding this once per group,
        // instead of continuing from the last index within the group, keeps the index
        // additions within a group out of the dependency chain between the groups (the
        // compiler can not reorder float additions, e.g., for f32 indices).
        let mut group_index_increment = Self::INDEX_INCREMENT;
        for _ in 1..VECTORS_PER_GROUP {
            group_index_increment = Self::_mm_add(group_index_increment, Self::INDEX_INCREMENT);
        }
        for _ in 0..nb_groups {
            // Reduce the next group to its lowest and highest values (and their index)
            let group_ptr = arr_ptr.add(LANE_SIZE);
            let mut group_index = Self::_mm_add(new_index, Self::INDEX_INCREMENT);
            // Start from the first vector. When ignoring NaNs, its NaNs become +inf
            // (-inf), which never replaces the running min (max).
            let (_, values_low) = Self::_initialize_index_values_low(group_ptr);
            let (_, values_high) = Self::_initialize_index_values_high(group_ptr);
            let mut group_low = (group_index, values_low);
            let mut group_high = (group_index, values_high);
            for i in 1..VECTORS_PER_GROUP {
                group_index = Self::_mm_add(group_index, Self::INDEX_INCREMENT);
                let new_values = Self::_mm_loadu(group_ptr.add(i * LANE_SIZE));
                group_low = Self::_update_low(group_low, (group_index, new_values));
                group_high = Self::_update_high(group_high, (group_index, new_values));
            }
            // Only these candidates are compared with the running min / max
            low = Self::_update_low(low, group_low);
            high = Self::_update_high(high, group_high);
            arr_ptr = arr_ptr.add(VECTORS_PER_GROUP * LANE_SIZE);
            new_index = Self::_mm_add(new_index, group_index_increment);
        }
        // The remaining vectors
        for _ in 0..nb_vectors - nb_groups * VECTORS_PER_GROUP {
            // Increment the index
            new_index = Self::_mm_add(new_index, Self::INDEX_INCREMENT);
            // Load the next chunk of data
            arr_ptr = arr_ptr.add(LANE_SIZE);
            let new_values = Self::_mm_loadu(arr_ptr);
            low = Self::_update_low(low, (new_index, new_values));
            high = Self::_update_high(high, (new_index, new_values));
        }

        // Get the min/max index and corresponding value from the SIMD vectors and return
        let ((index_low, values_low), (index_high, values_high)) = (low, high);
        let (min_index, min_value) = Self::_horiz_min(index_low, values_low);
        let (max_index, max_value) = Self::_horiz_max(index_high, values_high);
        (min_index, min_value, max_index, max_value)
    }

    /// Core argmin algorithm - returns (argmin, min)
    ///
    /// This method asserts:
    /// - the array length is a multiple of LANE_SIZE
    ///
    /// This method assumes:
    /// - the array length is <= MAX_INDEX
    ///
    /// Note that this method is not overflow safe, as it assumes that the array length
    /// is <= MAX_INDEX. The `_overflow_safe_core_argmin` method is overflow safe.
    ///
    /// This method calls `_core_argminmax`: as both are inlined, the compiler removes
    /// the computations for the maximum.
    ///
    #[inline(always)]
    unsafe fn _core_argmin(arr: &[ScalarDType]) -> (usize, ScalarDType) {
        let (min_index, min_value, _, _) = Self::_core_argminmax(arr);
        (min_index, min_value)
    }

    /// Core argmax algorithm - returns (argmax, max)
    ///
    /// This method asserts:
    /// - the array length is a multiple of LANE_SIZE
    ///
    /// This method assumes:
    /// - the array length is <= MAX_INDEX
    ///
    /// Note that this method is not overflow safe, as it assumes that the array length
    /// is <= MAX_INDEX. The `_overflow_safe_core_argmax` method is overflow safe.
    ///
    /// This method calls `_core_argminmax`: as both are inlined, the compiler removes
    /// the computations for the minimum.
    ///
    #[inline(always)]
    unsafe fn _core_argmax(arr: &[ScalarDType]) -> (usize, ScalarDType) {
        let (_, _, max_index, max_value) = Self::_core_argminmax(arr);
        (max_index, max_value)
    }

    /// Update the lowest (index, values) with the new (index, values) where these are
    /// lower. On ties the current values are kept, so the first occurrence wins as long
    /// as the new values come later in the array.
    #[inline(always)]
    unsafe fn _update_low(
        (index_low, values_low): (SIMDVecDtype, SIMDVecDtype),
        (new_index, new_values): (SIMDVecDtype, SIMDVecDtype),
    ) -> (SIMDVecDtype, SIMDVecDtype) {
        let mask_low = Self::_mm_cmplt(new_values, values_low);
        // Blend the values first, as the next comparison waits for them
        let values_low = Self::_mm_blendv(values_low, new_values, mask_low);
        let index_low = Self::_mm_blendv(index_low, new_index, mask_low);
        (index_low, values_low)
    }

    /// Update the highest (index, values) with the new (index, values) where these are
    /// higher. On ties the current values are kept, so the first occurrence wins as
    /// long as the new values come later in the array.
    #[inline(always)]
    unsafe fn _update_high(
        (index_high, values_high): (SIMDVecDtype, SIMDVecDtype),
        (new_index, new_values): (SIMDVecDtype, SIMDVecDtype),
    ) -> (SIMDVecDtype, SIMDVecDtype) {
        let mask_high = Self::_mm_cmpgt(new_values, values_high);
        // Blend the values first, as the next comparison waits for them
        let values_high = Self::_mm_blendv(values_high, new_values, mask_high);
        let index_high = Self::_mm_blendv(index_high, new_index, mask_high);
        (index_high, values_high)
    }

    /// Overflow-safe core argminmax algorithm - returns (argmin, min, argmax, max)
    ///
    /// This method asserts:
    /// - the array is not empty
    /// - the array length is a multiple of LANE_SIZE
    ///
    #[inline(always)]
    unsafe fn _overflow_safe_core_argminmax(
        arr: &[ScalarDType],
    ) -> (usize, ScalarDType, usize, ScalarDType) {
        assert!(!arr.is_empty());
        assert_eq!(arr.len() % LANE_SIZE, 0);
        // 0. Get the max value of the data type - which needs to be divided by LANE_SIZE
        let dtype_max = Self::_get_overflow_lane_size_limit();

        // 1. Determine the number of loops needed
        // let n_loops = (arr.len() + dtype_max - 1) / dtype_max; // ceil division
        let n_loops = arr.len() / dtype_max; // floor division

        // 2. Perform overflow-safe _core_argminmax
        let mut min_index: usize = 0;
        let mut min_value: ScalarDType = Self::_initialize_min_value(arr);
        let mut max_index: usize = 0;
        let mut max_value: ScalarDType = Self::_initialize_max_value(arr);
        let mut start: usize = 0;
        // 2.0 Perform the full loops
        for _ in 0..n_loops {
            if Self::_return_check(min_value) || Self::_return_check(max_value) {
                // We can return immediately
                return (min_index, min_value, max_index, max_value);
            }
            let (min_index_, min_value_, max_index_, max_value_) =
                Self::_core_argminmax(&arr[start..start + dtype_max]);
            if min_value_ < min_value || Self::_return_check(min_value_) {
                min_index = start + min_index_;
                min_value = min_value_;
            }
            if max_value_ > max_value || Self::_return_check(max_value_) {
                max_index = start + max_index_;
                max_value = max_value_;
            }
            start += dtype_max;
        }
        // 2.1 Handle the remainder
        if start < arr.len() {
            if Self::_return_check(min_value) || Self::_return_check(max_value) {
                // We can return immediately
                return (min_index, min_value, max_index, max_value);
            }
            let (min_index_, min_value_, max_index_, max_value_) =
                Self::_core_argminmax(&arr[start..]);
            if min_value_ < min_value || Self::_return_check(min_value_) {
                min_index = start + min_index_;
                min_value = min_value_;
            }
            if max_value_ > max_value || Self::_return_check(max_value_) {
                max_index = start + max_index_;
                max_value = max_value_;
            }
        }

        // 3. Return the min/max index and corresponding value
        (min_index, min_value, max_index, max_value)
    }

    /// Overflow-safe core argmin algorithm - returns (argmin, min)
    ///
    /// This method asserts:
    /// - the array is not empty
    /// - the array length is a multiple of LANE_SIZE
    ///
    #[inline(always)]
    unsafe fn _overflow_safe_core_argmin(arr: &[ScalarDType]) -> (usize, ScalarDType) {
        assert!(!arr.is_empty());
        assert_eq!(arr.len() % LANE_SIZE, 0);
        // 0. Get the max value of the data type - which needs to be divided by LANE_SIZE
        let dtype_max = Self::_get_overflow_lane_size_limit();

        // 1. Determine the number of loops needed
        let n_loops = arr.len() / dtype_max; // floor division

        // 2. Perform overflow-safe _core_argminmax
        let mut min_index: usize = 0;
        let mut min_value: ScalarDType = Self::_initialize_min_value(arr);
        let mut start: usize = 0;
        // 2.0 Perform the full loops
        for _ in 0..n_loops {
            if Self::_return_check(min_value) {
                // We can return immediately
                return (min_index, min_value);
            }
            let (min_index_, min_value_) = Self::_core_argmin(&arr[start..start + dtype_max]);
            if min_value_ < min_value || Self::_return_check(min_value_) {
                min_index = start + min_index_;
                min_value = min_value_;
            }
            start += dtype_max;
        }
        // 2.1 Handle the remainder
        if start < arr.len() {
            if Self::_return_check(min_value) {
                // We can return immediately
                return (min_index, min_value);
            }
            let (min_index_, min_value_) = Self::_core_argmin(&arr[start..]);
            if min_value_ < min_value || Self::_return_check(min_value_) {
                min_index = start + min_index_;
                min_value = min_value_;
            }
        }

        // 3. Return the min/max index and corresponding value
        (min_index, min_value)
    }

    /// Overflow-safe core argmax algorithm - returns (argmax, max)
    ///
    /// This method asserts:
    /// - the array is not empty
    /// - the array length is a multiple of LANE_SIZE
    ///
    #[inline(always)]
    unsafe fn _overflow_safe_core_argmax(arr: &[ScalarDType]) -> (usize, ScalarDType) {
        assert!(!arr.is_empty());
        assert_eq!(arr.len() % LANE_SIZE, 0);
        // 0. Get the max value of the data type - which needs to be divided by LANE_SIZE
        let dtype_max = Self::_get_overflow_lane_size_limit();

        // 1. Determine the number of loops needed
        let n_loops = arr.len() / dtype_max; // floor division

        // 2. Perform overflow-safe _core_argminmax
        let mut max_index: usize = 0;
        let mut max_value: ScalarDType = Self::_initialize_max_value(arr);
        let mut start: usize = 0;
        // 2.0 Perform the full loops
        for _ in 0..n_loops {
            if Self::_return_check(max_value) {
                // We can return immediately
                return (max_index, max_value);
            }
            let (max_index_, max_value_) = Self::_core_argmax(&arr[start..start + dtype_max]);
            if max_value_ > max_value || Self::_return_check(max_value_) {
                max_index = start + max_index_;
                max_value = max_value_;
            }
            start += dtype_max;
        }
        // 2.1 Handle the remainder
        if start < arr.len() {
            if Self::_return_check(max_value) {
                // We can return immediately
                return (max_index, max_value);
            }
            let (max_index_, max_value_) = Self::_core_argmax(&arr[start..]);
            if max_value_ > max_value || Self::_return_check(max_value_) {
                max_index = start + max_index_;
                max_value = max_value_;
            }
        }

        // 3. Return the min/max index and corresponding value
        (max_index, max_value)
    }
}

/// Implement SIMDCore where SIMDOps & SIMDInit are implemented
impl<T, ScalarDType, SIMDVecDtype, SIMDMaskDtype, const LANE_SIZE: usize>
    SIMDCore<ScalarDType, SIMDVecDtype, SIMDMaskDtype, LANE_SIZE> for T
where
    ScalarDType: Copy + PartialOrd + AsPrimitive<usize>,
    SIMDVecDtype: Copy,
    SIMDMaskDtype: Copy,
    T: SIMDOps<ScalarDType, SIMDVecDtype, SIMDMaskDtype, LANE_SIZE>
        + SIMDInit<ScalarDType, SIMDVecDtype, SIMDMaskDtype, LANE_SIZE>,
{
    // Implement the SIMDCore trait
}

// -------------------------------- ArgMinMax SIMD TRAIT -------------------------------

/// A trait providing the SIMD implementation of the argminmax operations.
///
// This trait its methods should be implemented for all structs that implement `SIMDOps`
// for the same generics.
// This trait is implemented in the `simd_*.rs` files calling the `impl_SIMDArgMinMax!`
// macro. With the exception of the `simd_f*_return_nan.rs` files, which implement this
// trait themselves (as these return .argminmax().0 and .argminmax().1 respectively
// instead of .argmin() and .argmax()).
//
pub trait SIMDArgMinMax<ScalarDType, SIMDVecDtype, SIMDMaskDtype, const LANE_SIZE: usize, SCALAR>:
    SIMDCore<ScalarDType, SIMDVecDtype, SIMDMaskDtype, LANE_SIZE>
where
    ScalarDType: Copy + PartialOrd + AsPrimitive<usize>,
    SIMDVecDtype: Copy,
    SIMDMaskDtype: Copy,
    SCALAR: ScalarArgMinMax<ScalarDType>,
{
    /// Get the index of the minimum and maximum values in the slice.
    ///
    /// # Arguments
    /// - `data` - the slice of data.
    ///
    /// # Returns
    /// A tuple of the index of the minimum and maximum values in the slice
    /// `(min_index, max_index)`.
    ///
    /// # Safety
    /// The caller must ensure that the CPU supports the target features that this
    /// instruction set requires for the data type (see the docs of the instruction set
    /// struct), e.g. with `is_x86_feature_detected!`. Calling this function on a CPU
    /// without these features is undefined behavior.
    /// See SIMD operations for more information:
    /// - [`x86` SIMD docs](https://doc.rust-lang.org/core/arch/x86/index.html)
    /// - [`x86_64` SIMD docs](https://doc.rust-lang.org/core/arch/x86_64/index.html)
    /// - [`arm` SIMD docs](https://doc.rust-lang.org/core/arch/arm/index.html)
    /// - [`aarch64` SIMD docs](https://doc.rust-lang.org/core/arch/aarch64/index.html)
    ///
    unsafe fn argminmax(data: &[ScalarDType]) -> (usize, usize);

    // Is necessary to have a separate function for this so we can call it in the
    // argminmax function when we add the target feature to the function.
    #[doc(hidden)]
    #[inline(always)]
    unsafe fn _argminmax(data: &[ScalarDType]) -> (usize, usize)
    where
        SCALAR: ScalarArgMinMax<ScalarDType>,
    {
        argminmax_generic(
            data,
            LANE_SIZE,
            Self::_overflow_safe_core_argminmax, // SIMD operation
            SCALAR::argminmax,                   // Scalar operation
            Self::_nan_check,                    // NaN check - true if value is NaN
            Self::IGNORE_NAN,                    // Ignore NaNs - if false -> return NaN
        )
    }

    /// Get the index of the minimum value in the slice.
    ///
    /// # Arguments
    /// - `data` - the slice of data.
    ///
    /// # Returns
    /// The index of the minimum value in the slice.
    ///
    /// # Safety
    /// The caller must ensure that the CPU supports the target features that this
    /// instruction set requires for the data type (see the docs of the instruction set
    /// struct), e.g. with `is_x86_feature_detected!`. Calling this function on a CPU
    /// without these features is undefined behavior.
    /// See SIMD operations for more information:
    /// - [`x86` SIMD docs](https://doc.rust-lang.org/core/arch/x86/index.html)
    /// - [`x86_64` SIMD docs](https://doc.rust-lang.org/core/arch/x86_64/index.html)
    /// - [`arm` SIMD docs](https://doc.rust-lang.org/core/arch/arm/index.html)
    /// - [`aarch64` SIMD docs](https://doc.rust-lang.org/core/arch/aarch64/index.html)
    ///
    unsafe fn argmin(data: &[ScalarDType]) -> usize;

    // Is necessary to have a separate function for this so we can call it in the
    // argmin function when we add the target feature to the function.
    #[doc(hidden)]
    #[inline(always)]
    unsafe fn _argmin(data: &[ScalarDType]) -> usize
    where
        SCALAR: ScalarArgMinMax<ScalarDType>,
    {
        argmin_generic(
            data,
            LANE_SIZE,
            Self::_overflow_safe_core_argmin, // SIMD operation
            SCALAR::argmin,                   // Scalar operation
            Self::_nan_check,                 // NaN check - true if value is NaN
            Self::IGNORE_NAN,                 // Ignore NaNs - if false -> return NaN
        )
    }

    /// Get the index of the maximum value in the slice.
    ///
    /// # Arguments
    /// - `data` - the slice of data.
    ///
    /// # Returns
    /// The index of the maximum value in the slice.
    ///
    /// # Safety
    /// The caller must ensure that the CPU supports the target features that this
    /// instruction set requires for the data type (see the docs of the instruction set
    /// struct), e.g. with `is_x86_feature_detected!`. Calling this function on a CPU
    /// without these features is undefined behavior.
    /// See SIMD operations for more information:
    /// - [`x86` SIMD docs](https://doc.rust-lang.org/core/arch/x86/index.html)
    /// - [`x86_64` SIMD docs](https://doc.rust-lang.org/core/arch/x86_64/index.html)
    /// - [`arm` SIMD docs](https://doc.rust-lang.org/core/arch/arm/index.html)
    /// - [`aarch64` SIMD docs](https://doc.rust-lang.org/core/arch/aarch64/index.html)
    ///
    unsafe fn argmax(data: &[ScalarDType]) -> usize;

    // Is necessary to have a separate function for this so we can call it in the
    // argmax function when we add the target feature to the function.
    #[doc(hidden)]
    #[inline(always)]
    unsafe fn _argmax(data: &[ScalarDType]) -> usize
    where
        SCALAR: ScalarArgMinMax<ScalarDType>,
    {
        argmax_generic(
            data,
            LANE_SIZE,
            Self::_overflow_safe_core_argmax, // SIMD operation
            SCALAR::argmax,                   // Scalar operation
            Self::_nan_check,                 // NaN check - true if value is NaN
            Self::IGNORE_NAN,                 // Ignore NaNs - if false -> return NaN
        )
    }

    /// Get the index of the minimum and maximum values in the slice, skipping the null
    /// elements.
    ///
    /// # Arguments
    /// - `data` - the slice of data.
    /// - `validity` - the validity bitmap: element `i` is valid (not null) iff bit
    ///   `offset + i` is set (in the [Arrow format](https://arrow.apache.org/docs/format/Columnar.html#validity-bitmaps)).
    /// - `offset` - the bit offset of the first element in the validity bitmap.
    ///
    /// # Returns
    /// A tuple of the index of the minimum and maximum valid values in the slice
    /// `(min_index, max_index)`, or `None` when there are no valid values.
    ///
    /// # Panics
    /// When the validity bitmap has less than `offset + data.len()` bits.
    ///
    /// # Safety
    /// The caller must ensure that the CPU supports the target features that this
    /// instruction set requires for the data type (see the docs of the instruction set
    /// struct), e.g. with `is_x86_feature_detected!`. Calling this function on a CPU
    /// without these features is undefined behavior.
    ///
    unsafe fn argminmax_masked(
        data: &[ScalarDType],
        validity: &[u8],
        offset: usize,
    ) -> Option<(usize, usize)>;

    /// Get the index of the minimum value in the slice, skipping the null elements.
    ///
    /// See [`argminmax_masked`](SIMDArgMinMax::argminmax_masked) for the arguments and
    /// the panics.
    ///
    /// # Returns
    /// The index of the minimum valid value in the slice, or `None` when there are no
    /// valid values.
    ///
    /// # Safety
    /// See [`argminmax_masked`](SIMDArgMinMax::argminmax_masked).
    ///
    unsafe fn argmin_masked(data: &[ScalarDType], validity: &[u8], offset: usize) -> Option<usize> {
        // Used by the return NaN implementations (see above), the others override this
        Self::argminmax_masked(data, validity, offset).map(|(min_index, _)| min_index)
    }

    /// Get the index of the maximum value in the slice, skipping the null elements.
    ///
    /// See [`argminmax_masked`](SIMDArgMinMax::argminmax_masked) for the arguments and
    /// the panics.
    ///
    /// # Returns
    /// The index of the maximum valid value in the slice, or `None` when there are no
    /// valid values.
    ///
    /// # Safety
    /// See [`argminmax_masked`](SIMDArgMinMax::argminmax_masked).
    ///
    unsafe fn argmax_masked(data: &[ScalarDType], validity: &[u8], offset: usize) -> Option<usize> {
        // Used by the return NaN implementations (see above), the others override this
        Self::argminmax_masked(data, validity, offset).map(|(_, max_index)| max_index)
    }
}

// ------------------------------ Masked ArgMinMax SIMD TRAIT --------------------------

/// The SIMD implementation of the masked argminmax operations, which the SIMD
/// implementations of `SIMDArgMinMax::argminmax_masked` (and of `argmin_masked` and
/// `argmax_masked`) call (see `impl_SIMDArgMinMax!` and the `simd_f*_return_nan.rs`
/// files). This trait is crate-private, as is the `SIMDValidity` trait it requires.
///
// Without SIMD (arm without the nightly_simd feature), nothing implements this trait
#[cfg_attr(
    not(any(
        target_arch = "x86",
        target_arch = "x86_64",
        all(target_arch = "arm", feature = "nightly_simd"),
        target_arch = "aarch64",
    )),
    allow(dead_code)
)]
pub(crate) trait SIMDMasked<
    ScalarDType,
    SIMDVecDtype,
    SIMDMaskDtype,
    const LANE_SIZE: usize,
    SCALAR,
> where
    Self: SIMDArgMinMax<ScalarDType, SIMDVecDtype, SIMDMaskDtype, LANE_SIZE, SCALAR>
        + SIMDValidity<SIMDMaskDtype, LANE_SIZE>,
    ScalarDType: Copy + PartialOrd + AsPrimitive<usize> + Bounded,
    SIMDVecDtype: Copy,
    SIMDMaskDtype: Copy,
    SCALAR: ScalarArgMinMax<ScalarDType>,
{
    /// Core masked algorithm - returns the (argmin, min) if `MIN` and the (argmax, max)
    /// if `MAX` of the valid elements (see `crate::validity` for the validity bitmap).
    ///
    /// The null lanes are replaced by a neutral value (the max value of the data type
    /// for the min, the min value for the max), which never replaces the running min /
    /// max. Thus, the returned value is the neutral value iff no valid value is below
    /// (above) it, in which case the returned index is meaningless.
    ///
    /// This method asserts:
    /// - the array length is a multiple of LANE_SIZE
    ///
    /// This method assumes:
    /// - the array length is <= MAX_INDEX
    /// - the validity bitmap has (at least) `offset + arr.len()` bits
    ///
    #[inline(always)]
    unsafe fn _core_masked<const MIN: bool, const MAX: bool>(
        arr: &[ScalarDType],
        validity: &[u8],
        offset: usize,
    ) -> FoundMinMax<ScalarDType> {
        assert_eq!(arr.len() % LANE_SIZE, 0);
        let neutral_low = Self::_mm_loadu([ScalarDType::max_value(); LANE_SIZE].as_ptr());
        let neutral_high = Self::_mm_loadu([ScalarDType::min_value(); LANE_SIZE].as_ptr());

        let mut arr_ptr = arr.as_ptr(); // Array pointer we will increment in the loop
        let mut new_index = Self::INITIAL_INDEX; // Index we will increment in the loop
        let (mut index_low, mut values_low) = (Self::INITIAL_INDEX, neutral_low);
        let (mut index_high, mut values_high) = (Self::INITIAL_INDEX, neutral_high);

        for start in (0..arr.len()).step_by(64) {
            let bits = validity_word(validity, offset, start);
            for lane in (0..64.min(arr.len() - start)).step_by(LANE_SIZE) {
                let valid = Self::_mm_validity_mask(bits >> lane);
                let new_values = Self::_mm_loadu(arr_ptr);

                if MIN {
                    // Update the lowest values and index
                    let new_low = Self::_mm_blendv(neutral_low, new_values, valid);
                    let mask_low = Self::_mm_cmplt(new_low, values_low);
                    values_low = Self::_mm_blendv(values_low, new_low, mask_low);
                    index_low = Self::_mm_blendv(index_low, new_index, mask_low);
                }
                if MAX {
                    // Update the highest values and index
                    let new_high = Self::_mm_blendv(neutral_high, new_values, valid);
                    let mask_high = Self::_mm_cmpgt(new_high, values_high);
                    values_high = Self::_mm_blendv(values_high, new_high, mask_high);
                    index_high = Self::_mm_blendv(index_high, new_index, mask_high);
                }

                // Increment the index and the array pointer
                new_index = Self::_mm_add(new_index, Self::INDEX_INCREMENT);
                arr_ptr = arr_ptr.add(LANE_SIZE);
            }
        }

        // Get the min/max index and corresponding value from the SIMD vectors and return
        let min = MIN.then(|| Self::_horiz_min(index_low, values_low));
        let max = MAX.then(|| Self::_horiz_max(index_high, values_high));
        (min, max)
    }

    /// Returns the (argmin, min) if `MIN` and the (argmax, max) if `MAX` of the valid
    /// elements (see `argminmax_masked` for the arguments and the panics).
    #[inline(always)]
    unsafe fn _masked<const MIN: bool, const MAX: bool>(
        data: &[ScalarDType],
        validity: &[u8],
        offset: usize,
    ) -> FoundMinMax<ScalarDType> {
        assert_validity_len(validity, offset, data.len());
        masked_generic::<_, SCALAR, MIN, MAX>(
            data,
            validity,
            offset,
            LANE_SIZE,
            Self::_get_overflow_lane_size_limit(),
            Self::_core_masked::<MIN, MAX>, // SIMD operation
        )
    }

    #[inline(always)]
    unsafe fn _argminmax_masked(
        data: &[ScalarDType],
        validity: &[u8],
        offset: usize,
    ) -> Option<(usize, usize)> {
        let (min, max) = Self::_masked::<true, true>(data, validity, offset);
        let ((min_index, min_value), (max_index, max_value)) = (min?, max?);
        Some(get_correct_argminmax_result(
            min_index,
            min_value,
            max_index,
            max_value,
            Self::_nan_check, // NaN check - true if value is NaN
            Self::IGNORE_NAN, // Ignore NaNs - if false -> return NaN
        ))
    }

    #[inline(always)]
    unsafe fn _argmin_masked(
        data: &[ScalarDType],
        validity: &[u8],
        offset: usize,
    ) -> Option<usize> {
        let (min, _) = Self::_masked::<true, false>(data, validity, offset);
        Some(min?.0)
    }

    #[inline(always)]
    unsafe fn _argmax_masked(
        data: &[ScalarDType],
        validity: &[u8],
        offset: usize,
    ) -> Option<usize> {
        let (_, max) = Self::_masked::<false, true>(data, validity, offset);
        Some(max?.0)
    }
}

/// Implement SIMDMasked where SIMDArgMinMax & SIMDValidity are implemented
impl<T, ScalarDType, SIMDVecDtype, SIMDMaskDtype, const LANE_SIZE: usize, SCALAR>
    SIMDMasked<ScalarDType, SIMDVecDtype, SIMDMaskDtype, LANE_SIZE, SCALAR> for T
where
    ScalarDType: Copy + PartialOrd + AsPrimitive<usize> + Bounded,
    SIMDVecDtype: Copy,
    SIMDMaskDtype: Copy,
    SCALAR: ScalarArgMinMax<ScalarDType>,
    T: SIMDArgMinMax<ScalarDType, SIMDVecDtype, SIMDMaskDtype, LANE_SIZE, SCALAR>
        + SIMDValidity<SIMDMaskDtype, LANE_SIZE>,
{
    // Implement the SIMDMasked trait
}

#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "arm", feature = "nightly_simd"),
    target_arch = "aarch64",
))]
macro_rules! impl_SIMDArgMinMax {
    ($scalar_dtype:ty, $simd_vec_dtype:ty, $simd_mask_dtype:ty, $lane_size:expr, $scalar_struct:ty, $simd_struct:ty, $target:expr) => {
        impl
            SIMDArgMinMax<
                $scalar_dtype,
                $simd_vec_dtype,
                $simd_mask_dtype,
                $lane_size,
                $scalar_struct,
            > for $simd_struct
        {
            #[target_feature(enable = $target)]
            unsafe fn argminmax(data: &[$scalar_dtype]) -> (usize, usize) {
                Self::_argminmax(data)
            }

            #[target_feature(enable = $target)]
            unsafe fn argmin(data: &[$scalar_dtype]) -> usize {
                Self::_argmin(data)
                // TODO: test if this is same speed as _argmin
                // Self::_argminmax(data).0
            }

            #[target_feature(enable = $target)]
            unsafe fn argmax(data: &[$scalar_dtype]) -> usize {
                Self::_argmax(data)
                // Self::_argminmax(data).1
            }

            #[target_feature(enable = $target)]
            unsafe fn argminmax_masked(
                data: &[$scalar_dtype],
                validity: &[u8],
                offset: usize,
            ) -> Option<(usize, usize)> {
                Self::_argminmax_masked(data, validity, offset)
            }

            #[target_feature(enable = $target)]
            unsafe fn argmin_masked(
                data: &[$scalar_dtype],
                validity: &[u8],
                offset: usize,
            ) -> Option<usize> {
                Self::_argmin_masked(data, validity, offset)
            }

            #[target_feature(enable = $target)]
            unsafe fn argmax_masked(
                data: &[$scalar_dtype],
                validity: &[u8],
                offset: usize,
            ) -> Option<usize> {
                Self::_argmax_masked(data, validity, offset)
            }
        }
    };
}

#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "arm", feature = "nightly_simd"),
    target_arch = "aarch64",
))]
pub(crate) use impl_SIMDArgMinMax; // Now classic paths Just Work™

// --------------------------------- Unimplement Macros --------------------------------

#[cfg(all(target_arch = "arm", feature = "nightly_simd"))]
macro_rules! unimpl_SIMDOps {
    ($scalar_type:ty, $reg:ty, $simd_struct:ty) => {
        impl SIMDOps<$scalar_type, $reg, $reg, 0> for $simd_struct {
            const INITIAL_INDEX: $reg = 0;
            const INDEX_INCREMENT: $reg = 0;
            const MAX_INDEX: usize = 0;

            unsafe fn _reg_to_arr(_reg: $reg) -> [$scalar_type; 0] {
                unimplemented!()
            }

            unsafe fn _mm_loadu(_data: *const $scalar_type) -> $reg {
                unimplemented!()
            }

            unsafe fn _mm_add(_a: $reg, _b: $reg) -> $reg {
                unimplemented!()
            }

            unsafe fn _mm_cmpgt(_a: $reg, _b: $reg) -> $reg {
                unimplemented!()
            }

            unsafe fn _mm_cmplt(_a: $reg, _b: $reg) -> $reg {
                unimplemented!()
            }

            unsafe fn _mm_blendv(_a: $reg, _b: $reg, _mask: $reg) -> $reg {
                unimplemented!()
            }
        }
    };
}

#[cfg(all(target_arch = "arm", feature = "nightly_simd"))]
macro_rules! unimpl_SIMDInit {
    ($scalar_type:ty, $reg:ty, $simd_struct:ty) => {
        impl SIMDInit<$scalar_type, $reg, $reg, 0> for $simd_struct {
            // Use the default implementation
        }
    };
}

#[cfg(all(target_arch = "arm", feature = "nightly_simd"))]
macro_rules! unimpl_SIMDArgMinMax {
    ($scalar_type:ty, $reg:ty, $scalar:ty, $simd_struct:ty) => {
        impl SIMDArgMinMax<$scalar_type, $reg, $reg, 0, $scalar> for $simd_struct {
            unsafe fn argminmax(_data: &[$scalar_type]) -> (usize, usize) {
                unimplemented!()
            }

            unsafe fn argmin(_data: &[$scalar_type]) -> usize {
                unimplemented!()
            }

            unsafe fn argmax(_data: &[$scalar_type]) -> usize {
                unimplemented!()
            }

            unsafe fn argminmax_masked(
                _data: &[$scalar_type],
                _validity: &[u8],
                _offset: usize,
            ) -> Option<(usize, usize)> {
                unimplemented!()
            }
        }
    };
}

#[cfg(all(target_arch = "arm", feature = "nightly_simd"))]
pub(crate) use unimpl_SIMDArgMinMax; // Now classic paths Just Work™

#[cfg(all(target_arch = "arm", feature = "nightly_simd"))]
pub(crate) use unimpl_SIMDInit; // Now classic paths Just Work™

#[cfg(all(target_arch = "arm", feature = "nightly_simd"))]
pub(crate) use unimpl_SIMDOps; // Now classic paths Just Work™

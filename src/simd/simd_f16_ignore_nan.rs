/// Currently not supported. Should give this some more thought.
///
// use super::config::SIMDInstructionSet;
use super::generic::{unimpl_SIMDArgMinMax, unimpl_SIMDInit, unimpl_SIMDOps};
use super::generic::{SIMDArgMinMax, SIMDInit, SIMDOps};
use crate::SCALAR;

use half::f16;

/// The dtype-strategy for performing operations on f16 data: ignore NaN values
use super::super::dtype_strategy::FloatIgnoreNaN;

// --------------------------------------- AVX2 ----------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx2_ignore_nan {
    use super::super::config::AVX2;
    use super::*;

    unimpl_SIMDOps!(f16, usize, AVX2<FloatIgnoreNaN>);
    unimpl_SIMDInit!(f16, usize, AVX2<FloatIgnoreNaN>);
    unimpl_SIMDArgMinMax!(f16, usize, SCALAR<FloatIgnoreNaN>, AVX2<FloatIgnoreNaN>);
}

// ---------------------------------------- SSE ----------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod sse_ignore_nan {
    use super::super::config::SSE;
    use super::*;

    unimpl_SIMDOps!(f16, usize, SSE<FloatIgnoreNaN>);
    unimpl_SIMDInit!(f16, usize, SSE<FloatIgnoreNaN>);
    unimpl_SIMDArgMinMax!(f16, usize, SCALAR<FloatIgnoreNaN>, SSE<FloatIgnoreNaN>);
}

// -------------------------------------- AVX512 ---------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx512_ignore_nan {
    use super::super::config::AVX512;
    use super::*;

    unimpl_SIMDOps!(f16, usize, AVX512<FloatIgnoreNaN>);
    unimpl_SIMDInit!(f16, usize, AVX512<FloatIgnoreNaN>);
    unimpl_SIMDArgMinMax!(f16, usize, SCALAR<FloatIgnoreNaN>, AVX512<FloatIgnoreNaN>);
}

// --------------------------------------- NEON ----------------------------------------

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
mod neon_ignore_nan {
    use super::super::config::NEON;
    use super::*;

    unimpl_SIMDOps!(f16, usize, NEON<FloatIgnoreNaN>);
    unimpl_SIMDInit!(f16, usize, NEON<FloatIgnoreNaN>);
    unimpl_SIMDArgMinMax!(f16, usize, SCALAR<FloatIgnoreNaN>, NEON<FloatIgnoreNaN>);
}

/// Currently not supported. Should give this some more thought.
///

// #[cfg(feature = "half")]
// use super::config::SIMDInstructionSet;
#[cfg(feature = "half")]
use super::generic::{unimpl_SIMDArgMinMaxIgnoreNaN, unimpl_SIMDOps};
#[cfg(feature = "half")]
use super::generic::{SIMDArgMinMaxIgnoreNaN, SIMDOps, SIMDSetOps};

#[cfg(feature = "half")]
use half::f16;

// ------------------------------------------ AVX2 ------------------------------------------

#[cfg(feature = "half")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx2_ignore_nan {
    use super::super::config::AVX2IgnoreNaN;
    use super::*;

    unimpl_SIMDOps!(f16, usize, AVX2IgnoreNaN);
    unimpl_SIMDArgMinMaxIgnoreNaN!(f16, usize, AVX2IgnoreNaN);
}

// ----------------------------------------- SSE -----------------------------------------

#[cfg(feature = "half")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod sse_ignore_nan {
    use super::super::config::SSEIgnoreNaN;
    use super::*;

    unimpl_SIMDOps!(f16, usize, SSEIgnoreNaN);
    unimpl_SIMDArgMinMaxIgnoreNaN!(f16, usize, SSEIgnoreNaN);
}

// --------------------------------------- AVX512 ----------------------------------------

#[cfg(feature = "half")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx512_ignore_nan {
    use super::super::config::AVX512IgnoreNaN;
    use super::*;

    unimpl_SIMDOps!(f16, usize, AVX512IgnoreNaN);
    unimpl_SIMDArgMinMaxIgnoreNaN!(f16, usize, AVX512IgnoreNaN);
}

// ---------------------------------------- NEON -----------------------------------------

#[cfg(feature = "half")]
#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
mod neon_ignore_nan {
    use super::super::config::NEONIgnoreNaN;
    use super::*;

    unimpl_SIMDOps!(f16, usize, NEONIgnoreNaN);
    unimpl_SIMDArgMinMaxIgnoreNaN!(f16, usize, NEONIgnoreNaN);
}

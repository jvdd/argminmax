// TODO: fix simd package private vs pub crate etc.

// FLOAT
pub(crate) mod simd_f64;
// pub use simd_f64::*;
pub(crate) mod simd_f32;
// pub use simd_f32::*;
pub(crate) mod simd_f16;
// pub use simd_f16::*;
// SIGNED INT
pub(crate) mod simd_i64;
// pub use simd_i64::*;
pub(crate) mod simd_i32;
// pub use simd_i32::*;
pub(crate) mod simd_i16;
// pub use simd_i16::*;
// UNSIGNED INT
mod config;
pub use config::*;
mod generic;
pub use generic::*;
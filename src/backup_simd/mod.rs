// FLOAT
pub mod simd_f64;
pub use simd_f64::*;
pub mod simd_f32;
pub use simd_f32::*;
pub mod simd_f16;
pub use simd_f16::*;
// SIGNED INT
pub mod simd_i64;
pub use simd_i64::*;
pub mod simd_i32;
pub use simd_i32::*;
pub mod simd_i16;
pub use simd_i16::*;
// UNSIGNED INT
mod config;
mod generic;
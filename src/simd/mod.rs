mod config;
pub use config::*;
mod generic;
pub use generic::*;
// FLOAT
mod simd_f16;
mod simd_f32;
mod simd_f64;
// SIGNED INT
mod simd_i16;
mod simd_i32;
mod simd_i64;
// UNSIGNED INT
mod simd_u16;
mod simd_u32;
mod simd_u64;

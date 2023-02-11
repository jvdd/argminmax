// Helper mod
mod task;
// Generic mods
mod config;
pub use config::*;
mod generic;
pub use generic::*;
// FLOAT
mod simd_f16_ignore_nans;
// mod simd_f16_return_nans; // TODO: not supported yet
mod simd_f32;
mod simd_f32_ignore_nans;
mod simd_f64;
mod simd_f64_ignore_nans;
// SIGNED INT
mod simd_i16;
mod simd_i32;
mod simd_i64;
mod simd_i8;
// UNSIGNED INT
mod simd_u16;
mod simd_u32;
mod simd_u64;
mod simd_u8;

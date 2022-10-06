pub mod generic;
mod simd;
mod task;

pub use generic::simple_argminmax;
pub use simd::{simd_f32, simd_f64, simd_i16, simd_i32, simd_i64};

use ndarray::ArrayView1;

pub trait ArgMinMax {
    fn argminmax(self) -> (usize, usize);
}

impl ArgMinMax for ArrayView1<'_, f64> {
    fn argminmax(self) -> (usize, usize) {
        // #[cfg(not(target_feature = "sse"))]
        // return Some(simple_argminmax(self));
        // #[cfg(target_feature = "sse")]
        return simd_f64::argminmax_f64(self);
    }
}

impl ArgMinMax for ArrayView1<'_, i64> {
    fn argminmax(self) -> (usize, usize) {
        // #[cfg(not(target_feature = "sse"))]
        // return Some(simple_argminmax(self));
        // #[cfg(target_feature = "sse")]
        return simd_i64::argminmax_i64(self);
    }
}

impl ArgMinMax for ArrayView1<'_, f32> {
    fn argminmax(self) -> (usize, usize) {
        // #[cfg(not(target_feature = "sse"))]
        // return Some(simple_argminmax(self));
        // #[cfg(target_feature = "sse")]
        return simd_f32::argminmax_f32(self);
    }
}

impl ArgMinMax for ArrayView1<'_, i32> {
    fn argminmax(self) -> (usize, usize) {
        // #[cfg(not(target_feature = "sse"))]
        // return Some(simple_argminmax(self));
        // #[cfg(target_feature = "sse")]
        return simd_i32::argminmax_i32(self);
    }
}

impl ArgMinMax for ArrayView1<'_, i16> {
    fn argminmax(self) -> (usize, usize) {
        // #[cfg(not(target_feature = "sse"))]
        // return Some(simple_argminmax(self));
        // #[cfg(target_feature = "sse")]
        return simd_i16::argminmax_i16(self);
    }
}

// impl ArgMinMax for ArrayView1<'_, i8> {
//     fn argminmax(self) -> Option<(usize, usize)> {
//         #[cfg(not(target_feature = "sse"))]
//         return Some(simple_argminmax(self));
//         #[cfg(target_feature = "sse")]
//         return simd_i8::argminmax_i8(self);
//     }
// }

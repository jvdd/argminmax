#![feature(stdsimd)]

use argminmax::ArgMinMax;
use codspeed_criterion_compat::*;
use dev_utils::{config, utils};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use argminmax::{FloatReturnNaN, SIMDArgMinMax, AVX2, AVX512, SSE};
#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
use argminmax::{FloatReturnNaN, SIMDArgMinMax, NEON};
use argminmax::{ScalarArgMinMax, SCALAR};

// _rn stands for "return nan"

fn argminmax_rn_f32_random_array_long(c: &mut Criterion) {
    let n = config::ARRAY_LENGTH_LONG;
    let data: &[f32] = &utils::get_random_array::<f32>(n, f32::MIN, f32::MAX);
    c.bench_function("scalar_f32_argminmax_rn", |b| {
        b.iter(|| SCALAR::<FloatReturnNaN>::argminmax(black_box(data)))
    });
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("sse4.1") {
        c.bench_function("sse_f32_argminmax_rn", |b| {
            b.iter(|| unsafe { SSE::<FloatReturnNaN>::argminmax(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") {
        c.bench_function("avx2_f32_argminmax_rn", |b| {
            b.iter(|| unsafe { AVX2::<FloatReturnNaN>::argminmax(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx512f") {
        c.bench_function("avx512_f32_argminmax_rn", |b| {
            b.iter(|| unsafe { AVX512::<FloatReturnNaN>::argminmax(black_box(data)) })
        });
    }
    #[cfg(target_arch = "arm")]
    if std::arch::is_arm_feature_detected!("neon") {
        c.bench_function("neon_f32_argminmax_rn", |b| {
            b.iter(|| unsafe { NEON::<FloatReturnNaN>::argminmax(black_box(data)) })
        });
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        c.bench_function("neon_f32_argminmax_rn", |b| {
            b.iter(|| unsafe { NEON::<FloatReturnNaN>::argminmax(black_box(data)) })
        });
    }
    c.bench_function("impl_f32_argminmax_rn", |b| {
        b.iter(|| black_box(data.nanargminmax()))
    });
}

criterion_group!(benches, argminmax_rn_f32_random_array_long,);
criterion_main!(benches);

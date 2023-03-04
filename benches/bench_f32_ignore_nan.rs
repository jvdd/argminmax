#![feature(stdsimd)]

use argminmax::ArgMinMax;
use codspeed_criterion_compat::*;
use dev_utils::{config, utils};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use argminmax::{FloatIgnoreNaN, SIMDArgMinMax, AVX2, AVX512, SSE};
#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
use argminmax::{FloatIgnoreNaN, SIMDArgMinMax, NEON};
use argminmax::{ScalarArgMinMax, SCALAR};

// _in stands for "ignore nan"

fn argminmax_in_f32_random_array_long(c: &mut Criterion) {
    let n = config::ARRAY_LENGTH_LONG;
    let data: &[f32] = &utils::get_random_array::<f32>(n, f32::MIN, f32::MAX);
    c.bench_function("scalar_f32_argminmax_in", |b| {
        b.iter(|| SCALAR::<FloatIgnoreNaN>::argminmax(black_box(data)))
    });
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("sse4.1") {
        c.bench_function("sse_f32_argminmax_in", |b| {
            b.iter(|| unsafe { SSE::<FloatIgnoreNaN>::argminmax(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx") {
        c.bench_function("avx_f32_argminmax_in", |b| {
            b.iter(|| unsafe { AVX2::<FloatIgnoreNaN>::argminmax(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx512f") {
        c.bench_function("avx512_f32_argminmax_in", |b| {
            b.iter(|| unsafe { AVX512::<FloatIgnoreNaN>::argminmax(black_box(data)) })
        });
    }
    #[cfg(target_arch = "arm")]
    if std::arch::is_arm_feature_detected!("neon") {
        c.bench_function("neon_f32_argminmax_in", |b| {
            b.iter(|| unsafe { NEON::<FloatIgnoreNaN>::argminmax(black_box(data)) })
        });
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        c.bench_function("neon_f32_argminmax_in", |b| {
            b.iter(|| unsafe { NEON::<FloatIgnoreNaN>::argminmax(black_box(data)) })
        });
    }
    c.bench_function("impl_f32_argminmax_in", |b| {
        b.iter(|| black_box(data.argminmax()))
    });
}

criterion_group!(benches, argminmax_in_f32_random_array_long,);
criterion_main!(benches);

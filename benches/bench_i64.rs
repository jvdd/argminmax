#![feature(stdsimd)]

use argminmax::ArgMinMax;
use codspeed_criterion_compat::*;
use dev_utils::{config, utils};

use argminmax::scalar::{ScalarArgMinMax, SCALAR};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use argminmax::simd::{SIMDArgMinMax, AVX2, AVX512, SSE};

fn argminmax_i64_random_array_long(c: &mut Criterion) {
    let n = config::ARRAY_LENGTH_LONG;
    let data: &[i64] = &utils::get_random_array::<i64>(n, i64::MIN, i64::MAX);
    c.bench_function("scalar_i64_argminmax", |b| {
        b.iter(|| SCALAR::argminmax(black_box(data)))
    });
    c.bench_function("scalar_i64_argmin", |b| {
        b.iter(|| SCALAR::argmin(black_box(data)))
    });
    c.bench_function("scalar_i64_argmax", |b| {
        b.iter(|| SCALAR::argmax(black_box(data)))
    });
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("sse4.2") {
        c.bench_function("sse_i64_argminmax", |b| {
            b.iter(|| unsafe { SSE::argminmax(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("sse4.2") {
        c.bench_function("sse_i64_argmin", |b| {
            b.iter(|| unsafe { SSE::argmin(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("sse4.2") {
        c.bench_function("sse_i64_argmax", |b| {
            b.iter(|| unsafe { SSE::argmax(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") {
        c.bench_function("avx2_i64_argminmax", |b| {
            b.iter(|| unsafe { AVX2::argminmax(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") {
        c.bench_function("avx2_i64_argmin", |b| {
            b.iter(|| unsafe { AVX2::argmin(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") {
        c.bench_function("avx2_i64_argmax", |b| {
            b.iter(|| unsafe { AVX2::argmax(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx512f") {
        c.bench_function("avx512_i64_argminmax", |b| {
            b.iter(|| unsafe { AVX512::argminmax(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx512f") {
        c.bench_function("avx512_i64_argmin", |b| {
            b.iter(|| unsafe { AVX512::argmin(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx512f") {
        c.bench_function("avx512_i64_argmax", |b| {
            b.iter(|| unsafe { AVX512::argmax(black_box(data)) })
        });
    }
    c.bench_function("impl_i64_argminmax", |b| {
        b.iter(|| black_box(data.argminmax()))
    });
    c.bench_function("impl_i64_argmin", |b| b.iter(|| black_box(data.argmin())));
    c.bench_function("impl_i64_argmax", |b| b.iter(|| black_box(data.argmax())));
}

criterion_group!(benches, argminmax_i64_random_array_long,);
criterion_main!(benches);

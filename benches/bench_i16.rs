#![feature(stdsimd)]

use argminmax::ArgMinMax;
use codspeed_criterion_compat::*;
use dev_utils::{config, utils};

use argminmax::scalar::{ScalarArgMinMax, SCALAR};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use argminmax::simd::{SIMDArgMinMax, AVX2, AVX512, SSE};
#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
use argminmax::simd::{SIMDArgMinMax, NEON};

fn argminmax_i16_random_array_long(c: &mut Criterion) {
    let n = config::ARRAY_LENGTH_LONG;
    let data: &[i16] = &utils::get_random_array::<i16>(n, i16::MIN, i16::MAX);
    c.bench_function("scalar_i16_argminmax", |b| {
        b.iter(|| SCALAR::argminmax(black_box(data)))
    });
    c.bench_function("scalar_i16_argmin", |b| {
        b.iter(|| SCALAR::argmin(black_box(data)))
    });
    c.bench_function("scalar_i16_argmax", |b| {
        b.iter(|| SCALAR::argmax(black_box(data)))
    });
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("sse4.1") {
        c.bench_function("sse_i16_argminmax", |b| {
            b.iter(|| unsafe { SSE::argminmax(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("sse4.1") {
        c.bench_function("sse_i16_argmin", |b| {
            b.iter(|| unsafe { SSE::argmin(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("sse4.1") {
        c.bench_function("sse_i16_argmax", |b| {
            b.iter(|| unsafe { SSE::argmax(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") {
        c.bench_function("avx2_i16_argminmax", |b| {
            b.iter(|| unsafe { AVX2::argminmax(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") {
        c.bench_function("avx2_i16_argmin", |b| {
            b.iter(|| unsafe { AVX2::argmin(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") {
        c.bench_function("avx2_i16_argmax", |b| {
            b.iter(|| unsafe { AVX2::argmax(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx512bw") {
        c.bench_function("avx512_i16_argminmax", |b| {
            b.iter(|| unsafe { AVX512::argminmax(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx512bw") {
        c.bench_function("avx512_i16_argmin", |b| {
            b.iter(|| unsafe { AVX512::argmin(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx512bw") {
        c.bench_function("avx512_i16_argmax", |b| {
            b.iter(|| unsafe { AVX512::argmax(black_box(data)) })
        });
    }
    #[cfg(target_arch = "arm")]
    if std::arch::is_arm_feature_detected!("neon") {
        c.bench_function("neon_i16_argminmax", |b| {
            b.iter(|| unsafe { NEON::argminmax(black_box(data)) })
        });
    }
    #[cfg(target_arch = "arm")]
    if std::arch::is_arm_feature_detected!("neon") {
        c.bench_function("neon_i16_argmin", |b| {
            b.iter(|| unsafe { NEON::argmin(black_box(data)) })
        });
    }
    #[cfg(target_arch = "arm")]
    if std::arch::is_arm_feature_detected!("neon") {
        c.bench_function("neon_i16_argmax", |b| {
            b.iter(|| unsafe { NEON::argmax(black_box(data)) })
        });
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        c.bench_function("neon_i16_argminmax", |b| {
            b.iter(|| unsafe { NEON::argminmax(black_box(data)) })
        });
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        c.bench_function("neon_i16_argmin", |b| {
            b.iter(|| unsafe { NEON::argmin(black_box(data)) })
        });
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        c.bench_function("neon_i16_argmax", |b| {
            b.iter(|| unsafe { NEON::argmax(black_box(data)) })
        });
    }
    c.bench_function("impl_i16_argminmax", |b| {
        b.iter(|| black_box(data.argminmax()))
    });
    c.bench_function("impl_i16_argmin", |b| b.iter(|| black_box(data.argmin())));
    c.bench_function("impl_i16_argmax", |b| b.iter(|| black_box(data.argmax())));
}

criterion_group!(benches, argminmax_i16_random_array_long,);
criterion_main!(benches);

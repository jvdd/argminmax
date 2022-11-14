#[macro_use]
extern crate criterion;
extern crate dev_utils;

use argminmax::ArgMinMax;
use criterion::{black_box, Criterion};
use dev_utils::{config, utils};

use argminmax::{AVX2, AVX512, NEON, SIMD, SSE};

fn minmax_i32_random_array_long(c: &mut Criterion) {
    let n = config::ARRAY_LENGTH_LONG;
    let data = utils::get_random_array::<i32>(n, i32::MIN, i32::MAX);
    c.bench_function("scalar_random_long_i32", |b| {
        b.iter(|| argminmax::scalar_argminmax(black_box(data.view())))
    });
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("sse4.1") {
        c.bench_function("sse_random_long_i32", |b| {
            b.iter(|| unsafe { SSE::argminmax(black_box(data.view())) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") {
        c.bench_function("avx2_random_long_i32", |b| {
            b.iter(|| unsafe { AVX2::argminmax(black_box(data.view())) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx512f") {
        c.bench_function("avx512_random_long_i32", |b| {
            b.iter(|| unsafe { AVX512::argminmax(black_box(data.view())) })
        });
    }
    #[cfg(target_arch = "arm")]
    if std::arch::is_arm_feature_detected!("neon") {
        c.bench_function("neon_random_long_i32", |b| {
            b.iter(|| unsafe { NEON::argminmax(black_box(data.view())) })
        });
    }
    c.bench_function("impl_random_long_i32", |b| {
        b.iter(|| black_box(data.view().argminmax()))
    });
}

fn minmax_i32_random_array_short(c: &mut Criterion) {
    let n = config::ARRAY_LENGTH_SHORT;
    let data = utils::get_random_array::<i32>(n, i32::MIN, i32::MAX);
    c.bench_function("scalar_random_short_i32", |b| {
        b.iter(|| argminmax::scalar_argminmax(black_box(data.view())))
    });
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("sse4.1") {
        c.bench_function("sse_random_short_i32", |b| {
            b.iter(|| unsafe { SSE::argminmax(black_box(data.view())) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") {
        c.bench_function("avx2_random_short_i32", |b| {
            b.iter(|| unsafe { AVX2::argminmax(black_box(data.view())) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx512f") {
        c.bench_function("avx512_random_short_i32", |b| {
            b.iter(|| unsafe { AVX512::argminmax(black_box(data.view())) })
        });
    }
    #[cfg(target_arch = "arm")]
    if std::arch::is_arm_feature_detected!("neon") {
        c.bench_function("neon_random_short_i32", |b| {
            b.iter(|| unsafe { NEON::argminmax(black_box(data.view())) })
        });
    }
    c.bench_function("impl_random_short_i32", |b| {
        b.iter(|| black_box(data.view().argminmax()))
    });
}

fn minmax_i32_worst_case_array_long(c: &mut Criterion) {
    let n = config::ARRAY_LENGTH_LONG;
    let data = utils::get_worst_case_array::<i32>(n, 1);
    c.bench_function("scalar_worst_long_i32", |b| {
        b.iter(|| argminmax::scalar_argminmax(black_box(data.view())))
    });
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("sse4.1") {
        c.bench_function("sse_worst_long_i32", |b| {
            b.iter(|| unsafe { SSE::argminmax(black_box(data.view())) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") {
        c.bench_function("avx2_worst_long_i32", |b| {
            b.iter(|| unsafe { AVX2::argminmax(black_box(data.view())) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx512f") {
        c.bench_function("avx512_worst_long_i32", |b| {
            b.iter(|| unsafe { AVX512::argminmax(black_box(data.view())) })
        });
    }
    #[cfg(target_arch = "arm")]
    if std::arch::is_arm_feature_detected!("neon") {
        c.bench_function("neon_worst_long_i32", |b| {
            b.iter(|| unsafe { NEON::argminmax(black_box(data.view())) })
        });
    }
    c.bench_function("impl_worst_long_i32", |b| {
        b.iter(|| black_box(data.view().argminmax()))
    });
}

fn minmax_i32_worst_case_array_short(c: &mut Criterion) {
    let n = config::ARRAY_LENGTH_SHORT;
    let data = utils::get_worst_case_array::<i32>(n, 1);
    c.bench_function("scalar_worst_short_i32", |b| {
        b.iter(|| argminmax::scalar_argminmax(black_box(data.view())))
    });
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("sse4.1") {
        c.bench_function("sse_worst_short_i32", |b| {
            b.iter(|| unsafe { SSE::argminmax(black_box(data.view())) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") {
        c.bench_function("avx2_worst_short_i32", |b| {
            b.iter(|| unsafe { AVX2::argminmax(black_box(data.view())) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx512f") {
        c.bench_function("avx512_worst_short_i32", |b| {
            b.iter(|| unsafe { AVX512::argminmax(black_box(data.view())) })
        });
    }
    #[cfg(target_arch = "arm")]
    if std::arch::is_arm_feature_detected!("neon") {
        c.bench_function("neon_worst_short_i32", |b| {
            b.iter(|| unsafe { NEON::argminmax(black_box(data.view())) })
        });
    }
    c.bench_function("impl_worst_short_i32", |b| {
        b.iter(|| black_box(data.view().argminmax()))
    });
}

criterion_group!(
    benches,
    minmax_i32_random_array_long,
    minmax_i32_random_array_short,
    minmax_i32_worst_case_array_long,
    minmax_i32_worst_case_array_short
);
criterion_main!(benches);

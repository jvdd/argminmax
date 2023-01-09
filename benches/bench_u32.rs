#![feature(stdsimd)]

#[macro_use]
extern crate criterion;
extern crate dev_utils;

use argminmax::ArgMinMax;
use criterion::{black_box, Criterion};
use dev_utils::{config, utils};

use argminmax::{ScalarArgMinMax, SCALAR};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use argminmax::{AVX2, AVX512, SIMD, SSE};
#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
use argminmax::{NEON, SIMD};

fn minmax_u32_random_array_long(c: &mut Criterion) {
    let n = config::ARRAY_LENGTH_LONG;
    let data: &[u32] = &utils::get_random_array::<u32>(n, u32::MIN, u32::MAX);
    c.bench_function("scalar_random_long_u32", |b| {
        b.iter(|| SCALAR::argminmax(black_box(data)))
    });
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("sse4.1") {
        c.bench_function("sse_random_long_u32", |b| {
            b.iter(|| unsafe { SSE::argminmax(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") {
        c.bench_function("avx2_random_long_u32", |b| {
            b.iter(|| unsafe { AVX2::argminmax(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx512f") {
        c.bench_function("avx512_random_long_u32", |b| {
            b.iter(|| unsafe { AVX512::argminmax(black_box(data)) })
        });
    }
    #[cfg(target_arch = "arm")]
    if std::arch::is_arm_feature_detected!("neon") {
        c.bench_function("neon_random_long_u32", |b| {
            b.iter(|| unsafe { NEON::argminmax(black_box(data)) })
        });
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        c.bench_function("neon_random_long_u32", |b| {
            b.iter(|| unsafe { NEON::argminmax(black_box(data)) })
        });
    }
    c.bench_function("impl_random_long_u32", |b| {
        b.iter(|| black_box(data.argminmax()))
    });
}

fn minmax_u32_random_array_short(c: &mut Criterion) {
    let n = config::ARRAY_LENGTH_SHORT;
    let data: &[u32] = &utils::get_random_array::<u32>(n, u32::MIN, u32::MAX);
    c.bench_function("scalar_random_short_u32", |b| {
        b.iter(|| SCALAR::argminmax(black_box(data)))
    });
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("sse4.1") {
        c.bench_function("sse_random_short_u32", |b| {
            b.iter(|| unsafe { SSE::argminmax(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") {
        c.bench_function("avx2_random_short_u32", |b| {
            b.iter(|| unsafe { AVX2::argminmax(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx512f") {
        c.bench_function("avx512_random_short_u32", |b| {
            b.iter(|| unsafe { AVX512::argminmax(black_box(data)) })
        });
    }
    #[cfg(target_arch = "arm")]
    if std::arch::is_arm_feature_detected!("neon") {
        c.bench_function("neon_random_short_u32", |b| {
            b.iter(|| unsafe { NEON::argminmax(black_box(data)) })
        });
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        c.bench_function("neon_random_short_u32", |b| {
            b.iter(|| unsafe { NEON::argminmax(black_box(data)) })
        });
    }
    c.bench_function("impl_random_short_u32", |b| {
        b.iter(|| black_box(data.argminmax()))
    });
}

fn minmax_u32_worst_case_array_long(c: &mut Criterion) {
    let n = config::ARRAY_LENGTH_LONG;
    let data: &[u32] = &utils::get_worst_case_array::<u32>(n, 1);
    c.bench_function("scalar_worst_long_u32", |b| {
        b.iter(|| SCALAR::argminmax(black_box(data)))
    });
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("sse4.1") {
        c.bench_function("sse_worst_long_u32", |b| {
            b.iter(|| unsafe { SSE::argminmax(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") {
        c.bench_function("avx2_worst_long_u32", |b| {
            b.iter(|| unsafe { AVX2::argminmax(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx512f") {
        c.bench_function("avx512_worst_long_u32", |b| {
            b.iter(|| unsafe { AVX512::argminmax(black_box(data)) })
        });
    }
    #[cfg(target_arch = "arm")]
    if std::arch::is_arm_feature_detected!("neon") {
        c.bench_function("neon_worst_long_u32", |b| {
            b.iter(|| unsafe { NEON::argminmax(black_box(data)) })
        });
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        c.bench_function("neon_worst_long_u32", |b| {
            b.iter(|| unsafe { NEON::argminmax(black_box(data)) })
        });
    }
    c.bench_function("impl_worst_long_u32", |b| {
        b.iter(|| black_box(data.argminmax()))
    });
}

// WORST case is not really worst case -> it is actually best case
fn minmax_u32_worst_case_array_short(c: &mut Criterion) {
    let n = config::ARRAY_LENGTH_SHORT;
    let data: &[u32] = &utils::get_worst_case_array::<u32>(n, 1);
    c.bench_function("scalar_worst_short_u32", |b| {
        b.iter(|| SCALAR::argminmax(black_box(data)))
    });
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("sse4.1") {
        c.bench_function("sse_worst_short_u32", |b| {
            b.iter(|| unsafe { SSE::argminmax(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") {
        c.bench_function("avx2_worst_short_u32", |b| {
            b.iter(|| unsafe { AVX2::argminmax(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx512f") {
        c.bench_function("avx512_worst_short_u32", |b| {
            b.iter(|| unsafe { AVX512::argminmax(black_box(data)) })
        });
    }
    #[cfg(target_arch = "arm")]
    if std::arch::is_arm_feature_detected!("neon") {
        c.bench_function("neon_worst_short_u32", |b| {
            b.iter(|| unsafe { NEON::argminmax(black_box(data)) })
        });
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        c.bench_function("neon_worst_short_u32", |b| {
            b.iter(|| unsafe { NEON::argminmax(black_box(data)) })
        });
    }
    c.bench_function("impl_worst_short_u32", |b| {
        b.iter(|| black_box(data.argminmax()))
    });
}

criterion_group!(
    benches,
    minmax_u32_random_array_long,
    // minmax_u32_random_array_short,
    // minmax_u32_worst_case_array_long,
    // minmax_u32_worst_case_array_short
);
criterion_main!(benches);

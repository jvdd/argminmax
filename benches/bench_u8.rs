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

fn minmax_u8_random_array_long(c: &mut Criterion) {
    let n = config::ARRAY_LENGTH_LONG;
    let data = utils::get_random_array::<u8>(n, u8::MIN, u8::MAX);
    c.bench_function("scalar_random_long_u8", |b| {
        b.iter(|| SCALAR::argminmax(black_box(data.view())))
    });
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("sse4.1") {
        c.bench_function("sse_random_long_u8", |b| {
            b.iter(|| unsafe { SSE::argminmax(black_box(data.view())) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") {
        c.bench_function("avx2_random_long_u8", |b| {
            b.iter(|| unsafe { AVX2::argminmax(black_box(data.view())) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx512bw") {
        c.bench_function("avx512_random_long_u8", |b| {
            b.iter(|| unsafe { AVX512::argminmax(black_box(data.view())) })
        });
    }
    #[cfg(target_arch = "arm")]
    if std::arch::is_arm_feature_detected!("neon") {
        c.bench_function("neon_random_long_u8", |b| {
            b.iter(|| unsafe { NEON::argminmax(black_box(data.view())) })
        });
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        c.bench_function("neon_random_long_u8", |b| {
            b.iter(|| unsafe { NEON::argminmax(black_box(data.view())) })
        });
    }
    c.bench_function("impl_random_long_u8", |b| {
        b.iter(|| black_box(data.view().argminmax()))
    });
}

fn minmax_u8_random_array_short(c: &mut Criterion) {
    let n = config::ARRAY_LENGTH_SHORT;
    let data = utils::get_random_array::<u8>(n, u8::MIN, u8::MAX);
    c.bench_function("scalar_random_short_u8", |b| {
        b.iter(|| SCALAR::argminmax(black_box(data.view())))
    });
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("sse4.1") {
        c.bench_function("sse_random_short_u8", |b| {
            b.iter(|| unsafe { SSE::argminmax(black_box(data.view())) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") {
        c.bench_function("avx2_random_short_u8", |b| {
            b.iter(|| unsafe { AVX2::argminmax(black_box(data.view())) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx512bw") {
        c.bench_function("avx512_random_short_u8", |b| {
            b.iter(|| unsafe { AVX512::argminmax(black_box(data.view())) })
        });
    }
    #[cfg(target_arch = "arm")]
    if std::arch::is_arm_feature_detected!("neon") {
        c.bench_function("neon_random_short_u8", |b| {
            b.iter(|| unsafe { NEON::argminmax(black_box(data.view())) })
        });
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        c.bench_function("neon_random_short_u8", |b| {
            b.iter(|| unsafe { NEON::argminmax(black_box(data.view())) })
        });
    }
    c.bench_function("impl_random_short_u8", |b| {
        b.iter(|| black_box(data.view().argminmax()))
    });
}

fn minmax_u8_worst_case_array_long(c: &mut Criterion) {
    let n = config::ARRAY_LENGTH_LONG;
    let data = utils::get_worst_case_array::<u8>(n, 1);
    c.bench_function("scalar_worst_long_u8", |b| {
        b.iter(|| SCALAR::argminmax(black_box(data.view())))
    });
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("sse4.1") {
        c.bench_function("sse_worst_long_u8", |b| {
            b.iter(|| unsafe { SSE::argminmax(black_box(data.view())) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") {
        c.bench_function("avx2_worst_long_u8", |b| {
            b.iter(|| unsafe { AVX2::argminmax(black_box(data.view())) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx512bw") {
        c.bench_function("avx512_worst_long_u8", |b| {
            b.iter(|| unsafe { AVX512::argminmax(black_box(data.view())) })
        });
    }
    #[cfg(target_arch = "arm")]
    if std::arch::is_arm_feature_detected!("neon") {
        c.bench_function("neon_worst_long_u8", |b| {
            b.iter(|| unsafe { NEON::argminmax(black_box(data.view())) })
        });
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        c.bench_function("neon_worst_long_u8", |b| {
            b.iter(|| unsafe { NEON::argminmax(black_box(data.view())) })
        });
    }
    c.bench_function("impl_worst_long_u8", |b| {
        b.iter(|| black_box(data.view().argminmax()))
    });
}

fn minmax_u8_worst_case_array_short(c: &mut Criterion) {
    let n = config::ARRAY_LENGTH_SHORT;
    let data = utils::get_worst_case_array::<u8>(n, 1);
    c.bench_function("scalar_worst_short_u8", |b| {
        b.iter(|| SCALAR::argminmax(black_box(data.view())))
    });
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("sse4.1") {
        c.bench_function("sse_worst_short_u8", |b| {
            b.iter(|| unsafe { SSE::argminmax(black_box(data.view())) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") {
        c.bench_function("avx2_worst_short_u8", |b| {
            b.iter(|| unsafe { AVX2::argminmax(black_box(data.view())) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx512bw") {
        c.bench_function("avx512_worst_short_u8", |b| {
            b.iter(|| unsafe { AVX512::argminmax(black_box(data.view())) })
        });
    }
    #[cfg(target_arch = "arm")]
    if std::arch::is_arm_feature_detected!("neon") {
        c.bench_function("neon_worst_short_u8", |b| {
            b.iter(|| unsafe { NEON::argminmax(black_box(data.view())) })
        });
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        c.bench_function("neon_worst_short_u8", |b| {
            b.iter(|| unsafe { NEON::argminmax(black_box(data.view())) })
        });
    }
    c.bench_function("impl_worst_short_u8", |b| {
        b.iter(|| black_box(data.view().argminmax()))
    });
}

criterion_group!(
    benches,
    minmax_u8_random_array_long,
    // minmax_u8_random_array_short,
    // minmax_u8_worst_case_array_long,
    // minmax_u8_worst_case_array_short
);
criterion_main!(benches);

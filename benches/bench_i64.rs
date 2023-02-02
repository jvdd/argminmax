#![feature(stdsimd)]

extern crate dev_utils;

#[cfg(feature = "half")]
use argminmax::ArgMinMax;
use codspeed_criterion_compat::*;
use dev_utils::{config, utils};

use argminmax::{ScalarArgMinMax, SCALAR};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use argminmax::{AVX2, AVX512, SIMD, SSE};

fn minmax_i64_random_array_long(c: &mut Criterion) {
    let n = config::ARRAY_LENGTH_LONG;
    let data = utils::get_random_array::<i64>(n, i64::MIN, i64::MAX);
    c.bench_function("scalar_random_long_i64", |b| {
        b.iter(|| SCALAR::argminmax(black_box(data.view())))
    });
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("sse4.2") {
        c.bench_function("sse_random_long_i64", |b| {
            b.iter(|| unsafe { SSE::argminmax(black_box(data.view())) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") {
        c.bench_function("avx2_random_long_i64", |b| {
            b.iter(|| unsafe { AVX2::argminmax(black_box(data.view())) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx512f") {
        c.bench_function("avx512_random_long_i64", |b| {
            b.iter(|| unsafe { AVX512::argminmax(black_box(data.view())) })
        });
    }
    c.bench_function("impl_random_long_i64", |b| {
        b.iter(|| black_box(data.view().argminmax()))
    });
}

fn minmax_i64_random_array_short(c: &mut Criterion) {
    let n = config::ARRAY_LENGTH_SHORT;
    let data = utils::get_random_array::<i64>(n, i64::MIN, i64::MAX);
    c.bench_function("scalar_random_short_i64", |b| {
        b.iter(|| SCALAR::argminmax(black_box(data.view())))
    });
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("sse4.2") {
        c.bench_function("sse_random_short_i64", |b| {
            b.iter(|| unsafe { SSE::argminmax(black_box(data.view())) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") {
        c.bench_function("avx2_random_short_i64", |b| {
            b.iter(|| unsafe { AVX2::argminmax(black_box(data.view())) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx512f") {
        c.bench_function("avx512_random_short_i64", |b| {
            b.iter(|| unsafe { AVX512::argminmax(black_box(data.view())) })
        });
    }
    c.bench_function("impl_random_short_i64", |b| {
        b.iter(|| black_box(data.view().argminmax()))
    });
}

fn minmax_i64_worst_case_array_long(c: &mut Criterion) {
    let n = config::ARRAY_LENGTH_LONG;
    let data = utils::get_worst_case_array::<i64>(n, 1);
    c.bench_function("scalar_worst_long_i64", |b| {
        b.iter(|| SCALAR::argminmax(black_box(data.view())))
    });
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("sse4.2") {
        c.bench_function("sse_worst_long_i64", |b| {
            b.iter(|| unsafe { SSE::argminmax(black_box(data.view())) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") {
        c.bench_function("avx2_worst_long_i64", |b| {
            b.iter(|| unsafe { AVX2::argminmax(black_box(data.view())) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx512f") {
        c.bench_function("avx512_worst_long_i64", |b| {
            b.iter(|| unsafe { AVX512::argminmax(black_box(data.view())) })
        });
    }
    c.bench_function("impl_worst_long_i64", |b| {
        b.iter(|| black_box(data.view().argminmax()))
    });
}

fn minmax_i64_worst_case_array_short(c: &mut Criterion) {
    let n = config::ARRAY_LENGTH_SHORT;
    let data = utils::get_worst_case_array::<i64>(n, 1);
    c.bench_function("scalar_worst_short_i64", |b| {
        b.iter(|| SCALAR::argminmax(black_box(data.view())))
    });
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("sse4.2") {
        c.bench_function("sse_worst_short_i64", |b| {
            b.iter(|| unsafe { SSE::argminmax(black_box(data.view())) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") {
        c.bench_function("avx2_worst_short_i64", |b| {
            b.iter(|| unsafe { AVX2::argminmax(black_box(data.view())) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx512f") {
        c.bench_function("avx512_worst_short_i64", |b| {
            b.iter(|| unsafe { AVX512::argminmax(black_box(data.view())) })
        });
    }
    c.bench_function("impl_worst_short_i64", |b| {
        b.iter(|| black_box(data.view().argminmax()))
    });
}

criterion_group!(
    benches,
    minmax_i64_random_array_long,
    // minmax_i64_random_array_short,
    // minmax_i64_worst_case_array_long,
    // minmax_i64_worst_case_array_short
);
criterion_main!(benches);

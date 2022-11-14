#[macro_use]
extern crate criterion;
extern crate dev_utils;

#[cfg(feature = "half")]
use argminmax::ArgMinMax;
use criterion::{black_box, Criterion};
use dev_utils::{config, utils};

use argminmax::{AVX2, AVX512, NEON, SIMD, SSE};

#[cfg(feature = "half")]
use half::f16;
use ndarray::Array1;

#[cfg(feature = "half")]
fn get_random_f16_array(n: usize) -> Array1<f16> {
    let data = utils::get_random_array::<u16>(n, u16::MIN, u16::MAX);
    let data: Vec<f16> = data.iter().map(|&x| f16::from_bits(x)).collect();
    // Replace NaNs and Infs with 0
    let data: Vec<f16> = data
        .iter()
        .map(|&x| {
            if x.is_nan() || x.is_infinite() {
                f16::from_bits(0)
            } else {
                x
            }
        })
        .collect();
    let arr: Array1<f16> = Array1::from(data);
    arr
}

#[cfg(feature = "half")]
fn minmax_f16_random_array_long(c: &mut Criterion) {
    let n = config::ARRAY_LENGTH_LONG;
    let data = get_random_f16_array(n);
    c.bench_function("scalar_random_long_f16", |b| {
        b.iter(|| argminmax::scalar_argminmax_f16(black_box(data.view())))
    });
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("sse4.1") {
        c.bench_function("sse_random_long_f16", |b| {
            b.iter(|| unsafe { SSE::argminmax(black_box(data.view())) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") {
        c.bench_function("avx2_random_long_f16", |b| {
            b.iter(|| unsafe { AVX2::argminmax(black_box(data.view())) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx512bw") {
        c.bench_function("avx512_random_long_f16", |b| {
            b.iter(|| unsafe { AVX512::argminmax(black_box(data.view())) })
        });
    }
    #[cfg(feature = "arm")]
    if std::arch::is_arm_feature_detected!("neon") {
        c.bench_function("neon_random_long_f16", |b| {
            b.iter(|| unsafe { NEON::argminmax(black_box(data.view())) })
        });
    }
    c.bench_function("impl_random_long_f16", |b| {
        b.iter(|| black_box(data.view().argminmax()))
    });
}

#[cfg(feature = "half")]
fn minmax_f16_random_array_short(c: &mut Criterion) {
    let n = config::ARRAY_LENGTH_SHORT;
    let data = get_random_f16_array(n);
    c.bench_function("scalar_random_short_f16", |b| {
        b.iter(|| argminmax::scalar_argminmax_f16(black_box(data.view())))
    });
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("sse4.1") {
        c.bench_function("sse_random_short_f16", |b| {
            b.iter(|| unsafe { SSE::argminmax(black_box(data.view())) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") {
        c.bench_function("avx2_random_short_f16", |b| {
            b.iter(|| unsafe { AVX2::argminmax(black_box(data.view())) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx512bw") {
        c.bench_function("avx512_random_short_f16", |b| {
            b.iter(|| unsafe { AVX512::argminmax(black_box(data.view())) })
        });
    }
    #[cfg(feature = "arm")]
    if std::arch::is_arm_feature_detected!("neon") {
        c.bench_function("neon_random_short_f16", |b| {
            b.iter(|| unsafe { NEON::argminmax(black_box(data.view())) })
        });
    }
    c.bench_function("impl_random_short_f16", |b| {
        b.iter(|| black_box(data.view().argminmax()))
    });
}

#[cfg(feature = "half")]
fn minmax_f16_worst_case_array_long(c: &mut Criterion) {
    let n = config::ARRAY_LENGTH_LONG;
    let data = utils::get_worst_case_array::<f16>(n, f16::from_f32(1.));
    c.bench_function("scalar_worst_long_f16", |b| {
        b.iter(|| argminmax::scalar_argminmax_f16(black_box(data.view())))
    });
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("sse4.1") {
        c.bench_function("sse_worst_long_f16", |b| {
            b.iter(|| unsafe { SSE::argminmax(black_box(data.view())) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") {
        c.bench_function("avx2_worst_long_f16", |b| {
            b.iter(|| unsafe { AVX2::argminmax(black_box(data.view())) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx512bw") {
        c.bench_function("avx512_worst_long_f16", |b| {
            b.iter(|| unsafe { AVX512::argminmax(black_box(data.view())) })
        });
    }
    #[cfg(feature = "arm")]
    if std::arch::is_arm_feature_detected!("neon") {
        c.bench_function("neon_worst_long_f16", |b| {
            b.iter(|| unsafe { NEON::argminmax(black_box(data.view())) })
        });
    }
    c.bench_function("impl_worst_long_f16", |b| {
        b.iter(|| black_box(data.view().argminmax()))
    });
}

#[cfg(feature = "half")]
fn minmax_f16_worst_case_array_short(c: &mut Criterion) {
    let n = config::ARRAY_LENGTH_SHORT;
    let data = utils::get_worst_case_array::<f16>(n, f16::from_f32(1.));
    c.bench_function("scalar_worst_short_f16", |b| {
        b.iter(|| argminmax::scalar_argminmax_f16(black_box(data.view())))
    });
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("sse4.1") {
        c.bench_function("sse_worst_short_f16", |b| {
            b.iter(|| unsafe { SSE::argminmax(black_box(data.view())) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") {
        c.bench_function("avx2_worst_short_f16", |b| {
            b.iter(|| unsafe { AVX2::argminmax(black_box(data.view())) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx512bw") {
        c.bench_function("avx512_worst_short_f16", |b| {
            b.iter(|| unsafe { AVX512::argminmax(black_box(data.view())) })
        });
    }
    #[cfg(feature = "arm")]
    if std::arch::is_arm_feature_detected!("neon") {
        c.bench_function("neon_worst_short_f16", |b| {
            b.iter(|| unsafe { NEON::argminmax(black_box(data.view())) })
        });
    }
    c.bench_function("impl_worst_short_f16", |b| {
        b.iter(|| black_box(data.view().argminmax()))
    });
}

#[cfg(feature = "half")]
criterion_group!(
    benches,
    minmax_f16_random_array_long,
    minmax_f16_random_array_short,
    minmax_f16_worst_case_array_long,
    minmax_f16_worst_case_array_short
);
#[cfg(feature = "half")]
criterion_main!(benches);

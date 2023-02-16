#![feature(stdsimd)]

extern crate dev_utils;

#[cfg(feature = "half")]
use argminmax::ArgMinMax;
use codspeed_criterion_compat::*;
use dev_utils::{config, utils};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use argminmax::{SIMDArgMinMax, AVX2, AVX512, SSE};
#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
use argminmax::{SIMDArgMinMax, NEON};
use argminmax::{ScalarArgMinMax, SCALAR};

#[cfg(feature = "half")]
use half::f16;

#[cfg(feature = "half")]
fn get_random_f16_array(n: usize) -> Vec<f16> {
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
    data
}

#[cfg(feature = "half")]
fn minmax_f16_random_array_long(c: &mut Criterion) {
    let n = config::ARRAY_LENGTH_LONG;
    let data: &[f16] = &get_random_f16_array(n);
    c.bench_function("scalar_random_long_f16", |b| {
        b.iter(|| SCALAR::argminmax(black_box(data)))
    });
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("sse4.1") {
        c.bench_function("sse_random_long_f16", |b| {
            b.iter(|| unsafe { SSE::argminmax(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") {
        c.bench_function("avx2_random_long_f16", |b| {
            b.iter(|| unsafe { AVX2::argminmax(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx512bw") {
        c.bench_function("avx512_random_long_f16", |b| {
            b.iter(|| unsafe { AVX512::argminmax(black_box(data)) })
        });
    }
    #[cfg(target_arch = "arm")]
    if std::arch::is_arm_feature_detected!("neon") {
        c.bench_function("neon_random_long_f16", |b| {
            b.iter(|| unsafe { NEON::argminmax(black_box(data)) })
        });
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        c.bench_function("neon_random_long_f16", |b| {
            b.iter(|| unsafe { NEON::argminmax(black_box(data)) })
        });
    }
    c.bench_function("impl_random_long_f16", |b| {
        b.iter(|| black_box(data.argminmax()))
    });
}

#[cfg(feature = "half")]
fn minmax_f16_random_array_short(c: &mut Criterion) {
    let n = config::ARRAY_LENGTH_SHORT;
    let data: &[f16] = &get_random_f16_array(n);
    c.bench_function("scalar_random_short_f16", |b| {
        b.iter(|| SCALAR::argminmax(black_box(data)))
    });
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("sse4.1") {
        c.bench_function("sse_random_short_f16", |b| {
            b.iter(|| unsafe { SSE::argminmax(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") {
        c.bench_function("avx2_random_short_f16", |b| {
            b.iter(|| unsafe { AVX2::argminmax(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx512bw") {
        c.bench_function("avx512_random_short_f16", |b| {
            b.iter(|| unsafe { AVX512::argminmax(black_box(data)) })
        });
    }
    #[cfg(target_arch = "arm")]
    if std::arch::is_arm_feature_detected!("neon") {
        c.bench_function("neon_random_short_f16", |b| {
            b.iter(|| unsafe { NEON::argminmax(black_box(data)) })
        });
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        c.bench_function("neon_random_short_f16", |b| {
            b.iter(|| unsafe { NEON::argminmax(black_box(data)) })
        });
    }
    c.bench_function("impl_random_short_f16", |b| {
        b.iter(|| black_box(data.argminmax()))
    });
}

#[cfg(feature = "half")]
fn minmax_f16_worst_case_array_long(c: &mut Criterion) {
    let n = config::ARRAY_LENGTH_LONG;
    let data: &[f16] = &utils::get_worst_case_array::<f16>(n, f16::from_f32(1.));
    c.bench_function("scalar_worst_long_f16", |b| {
        b.iter(|| SCALAR::argminmax(black_box(data)))
    });
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("sse4.1") {
        c.bench_function("sse_worst_long_f16", |b| {
            b.iter(|| unsafe { SSE::argminmax(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") {
        c.bench_function("avx2_worst_long_f16", |b| {
            b.iter(|| unsafe { AVX2::argminmax(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx512bw") {
        c.bench_function("avx512_worst_long_f16", |b| {
            b.iter(|| unsafe { AVX512::argminmax(black_box(data)) })
        });
    }
    #[cfg(target_arch = "arm")]
    if std::arch::is_arm_feature_detected!("neon") {
        c.bench_function("neon_worst_long_f16", |b| {
            b.iter(|| unsafe { NEON::argminmax(black_box(data)) })
        });
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        c.bench_function("neon_worst_long_f16", |b| {
            b.iter(|| unsafe { NEON::argminmax(black_box(data)) })
        });
    }
    c.bench_function("impl_worst_long_f16", |b| {
        b.iter(|| black_box(data.argminmax()))
    });
}

#[cfg(feature = "half")]
fn minmax_f16_worst_case_array_short(c: &mut Criterion) {
    let n = config::ARRAY_LENGTH_SHORT;
    let data: &[f16] = &utils::get_worst_case_array::<f16>(n, f16::from_f32(1.));
    c.bench_function("scalar_worst_short_f16", |b| {
        b.iter(|| SCALAR::argminmax(black_box(data)))
    });
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("sse4.1") {
        c.bench_function("sse_worst_short_f16", |b| {
            b.iter(|| unsafe { SSE::argminmax(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") {
        c.bench_function("avx2_worst_short_f16", |b| {
            b.iter(|| unsafe { AVX2::argminmax(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx512bw") {
        c.bench_function("avx512_worst_short_f16", |b| {
            b.iter(|| unsafe { AVX512::argminmax(black_box(data)) })
        });
    }
    #[cfg(target_arch = "arm")]
    if std::arch::is_arm_feature_detected!("neon") {
        c.bench_function("neon_worst_short_f16", |b| {
            b.iter(|| unsafe { NEON::argminmax(black_box(data)) })
        });
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        c.bench_function("neon_worst_short_f16", |b| {
            b.iter(|| unsafe { NEON::argminmax(black_box(data)) })
        });
    }
    c.bench_function("impl_worst_short_f16", |b| {
        b.iter(|| black_box(data.argminmax()))
    });
}

#[cfg(feature = "half")]
criterion_group!(
    benches,
    minmax_f16_random_array_long,
    // minmax_f16_random_array_short,
    // minmax_f16_worst_case_array_long,
    // minmax_f16_worst_case_array_short
);
#[cfg(feature = "half")]
criterion_main!(benches);

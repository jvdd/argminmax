#![feature(stdsimd)]

use argminmax::NaNArgMinMax;
use codspeed_criterion_compat::*;
use dev_utils::{config, utils};

use argminmax::dtype_strategy::FloatReturnNaN;
use argminmax::scalar::{ScalarArgMinMax, SCALAR};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use argminmax::simd::{SIMDArgMinMax, AVX2, AVX512, SSE};
#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
use argminmax::simd::{SIMDArgMinMax, NEON};

use half::f16;

fn get_random_f16_array(n: usize) -> Vec<f16> {
    let data = utils::get_random_array::<i16>(n, -0x7C00, 0x7C00);
    let data: Vec<f16> = data.iter().map(|&x| f16::from_bits(x as u16)).collect();
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

// _rn stands for "return nan"

fn argminmax_rn_f16_random_array_long(c: &mut Criterion) {
    let n = config::ARRAY_LENGTH_LONG;
    let data: &[f16] = &get_random_f16_array(n);
    c.bench_function("scalar_f16_argminmax_rn", |b| {
        b.iter(|| SCALAR::<FloatReturnNaN>::argminmax(black_box(data)))
    });
    c.bench_function("scalar_f16_argmin_rn", |b| {
        b.iter(|| SCALAR::<FloatReturnNaN>::argmin(black_box(data)))
    });
    c.bench_function("scalar_f16_argmax_rn", |b| {
        b.iter(|| SCALAR::<FloatReturnNaN>::argmax(black_box(data)))
    });
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("sse4.1") {
        c.bench_function("sse_f16_argminmax_rn", |b| {
            b.iter(|| unsafe { SSE::<FloatReturnNaN>::argminmax(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("sse4.1") {
        c.bench_function("sse_f16_argmin_rn", |b| {
            b.iter(|| unsafe { SSE::<FloatReturnNaN>::argmin(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("sse4.1") {
        c.bench_function("sse_f16_argmax_rn", |b| {
            b.iter(|| unsafe { SSE::<FloatReturnNaN>::argmax(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") {
        c.bench_function("avx2_f16_argminmax_rn", |b| {
            b.iter(|| unsafe { AVX2::<FloatReturnNaN>::argminmax(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") {
        c.bench_function("avx2_f16_argmin_rn", |b| {
            b.iter(|| unsafe { AVX2::<FloatReturnNaN>::argmin(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") {
        c.bench_function("avx2_f16_argmax_rn", |b| {
            b.iter(|| unsafe { AVX2::<FloatReturnNaN>::argmax(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx512bw") {
        c.bench_function("avx512_f16_argminmax_rn", |b| {
            b.iter(|| unsafe { AVX512::<FloatReturnNaN>::argminmax(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx512bw") {
        c.bench_function("avx512_f16_argmin_rn", |b| {
            b.iter(|| unsafe { AVX512::<FloatReturnNaN>::argmin(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx512bw") {
        c.bench_function("avx512_f16_argmax_rn", |b| {
            b.iter(|| unsafe { AVX512::<FloatReturnNaN>::argmax(black_box(data)) })
        });
    }
    #[cfg(target_arch = "arm")]
    if std::arch::is_arm_feature_detected!("neon") {
        c.bench_function("neon_f16_argminmax_rn", |b| {
            b.iter(|| unsafe { NEON::<FloatReturnNaN>::argminmax(black_box(data)) })
        });
    }
    #[cfg(target_arch = "arm")]
    if std::arch::is_arm_feature_detected!("neon") {
        c.bench_function("neon_f16_argmin_rn", |b| {
            b.iter(|| unsafe { NEON::<FloatReturnNaN>::argmin(black_box(data)) })
        });
    }
    #[cfg(target_arch = "arm")]
    if std::arch::is_arm_feature_detected!("neon") {
        c.bench_function("neon_f16_argmax_rn", |b| {
            b.iter(|| unsafe { NEON::<FloatReturnNaN>::argmax(black_box(data)) })
        });
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        c.bench_function("neon_f16_argminmax_rn", |b| {
            b.iter(|| unsafe { NEON::<FloatReturnNaN>::argminmax(black_box(data)) })
        });
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        c.bench_function("neon_f16_argmin_rn", |b| {
            b.iter(|| unsafe { NEON::<FloatReturnNaN>::argmin(black_box(data)) })
        });
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        c.bench_function("neon_f16_argmax_rn", |b| {
            b.iter(|| unsafe { NEON::<FloatReturnNaN>::argmax(black_box(data)) })
        });
    }
    c.bench_function("impl_f16_argminmax_rn", |b| {
        b.iter(|| black_box(data.nanargminmax()))
    });
    c.bench_function("impl_f16_argmin_rn", |b| {
        b.iter(|| black_box(data.nanargmin()))
    });
    c.bench_function("impl_f16_argmax_rn", |b| {
        b.iter(|| black_box(data.nanargmax()))
    });
}

criterion_group!(benches, argminmax_rn_f16_random_array_long,);
criterion_main!(benches);

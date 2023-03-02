#![feature(stdsimd)]

extern crate dev_utils;

use argminmax::ArgMinMax;
use codspeed_criterion_compat::*;
use dev_utils::{config, utils};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use argminmax::{AVX2IgnoreNaN, AVX512IgnoreNaN, SIMDArgMinMaxIgnoreNaN, SSEIgnoreNaN};
use argminmax::{SCALARIgnoreNaN, ScalarArgMinMax};

// _in stands for "ignore nan"

fn argminmax_in_f64_random_array_long(c: &mut Criterion) {
    let n = config::ARRAY_LENGTH_LONG;
    let data: &[f64] = &utils::get_random_array::<f64>(n, f64::MIN, f64::MAX);
    c.bench_function("scalar_argminmax_in_f64", |b| {
        b.iter(|| SCALARIgnoreNaN::argminmax(black_box(data)))
    });
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("sse4.1") {
        c.bench_function("sse_argminmax_in_f64", |b| {
            b.iter(|| unsafe { SSEIgnoreNaN::argminmax(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx") {
        c.bench_function("avx_argminmax_in_f64", |b| {
            b.iter(|| unsafe { AVX2IgnoreNaN::argminmax(black_box(data)) })
        });
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx512f") {
        c.bench_function("avx512_argminmax_in_f64", |b| {
            b.iter(|| unsafe { AVX512IgnoreNaN::argminmax(black_box(data)) })
        });
    }
    c.bench_function("impl_argminmax_in_f64", |b| {
        b.iter(|| black_box(data.argminmax()))
    });
}

criterion_group!(benches, argminmax_in_f64_random_array_long,);
criterion_main!(benches);

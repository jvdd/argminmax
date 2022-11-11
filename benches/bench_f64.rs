#[macro_use]
extern crate criterion;
extern crate dev_utils;

use argminmax::ArgMinMax;
use criterion::{black_box, Criterion};
use dev_utils::{config, utils};

use argminmax::{AVX2, AVX512, SSE, SIMD};

fn minmax_f64_random_array_long(c: &mut Criterion) {
    let n = config::ARRAY_LENGTH_LONG;
    let data = utils::get_random_array::<f64>(n, f64::MIN, f64::MAX);
    c.bench_function("simple_random_long_f64", |b| {
        b.iter(|| argminmax::scalar_argminmax(black_box(data.view())))
    });
    c.bench_function("sse_random_long_f64", |b| {
        b.iter(|| black_box(unsafe { SSE::argminmax(black_box(data.view())) }))
    });
    c.bench_function("avx2_random_long_f64", |b| {
        b.iter(|| black_box(unsafe { AVX2::argminmax(black_box(data.view())) }))
    });
    c.bench_function("avx512_random_long_f64", |b| {
        b.iter(|| black_box(unsafe { AVX512::argminmax(black_box(data.view())) }))
    });
    c.bench_function("impl_random_long_f64", |b| {
        b.iter(|| black_box(data.view().argminmax()))
    });
}

fn minmax_f64_random_array_short(c: &mut Criterion) {
    let n = config::ARRAY_LENGTH_SHORT;
    let data = utils::get_random_array::<f64>(n, f64::MIN, f64::MAX);
    c.bench_function("simple_random_short_f64", |b| {
        b.iter(|| argminmax::scalar_argminmax(black_box(data.view())))
    });
    c.bench_function("sse_random_short_f64", |b| {
        b.iter(|| black_box(unsafe { SSE::argminmax(black_box(data.view())) }))
    });
    c.bench_function("avx2_random_short_f64", |b| {
        b.iter(|| black_box(unsafe { AVX2::argminmax(black_box(data.view())) }))
    });
    c.bench_function("avx512_random_short_f64", |b| {
        b.iter(|| black_box(unsafe { AVX512::argminmax(black_box(data.view())) }))
    });
    c.bench_function("impl_random_short_f64", |b| {
        b.iter(|| black_box(data.view().argminmax()))
    });
}

fn minmax_f64_worst_case_array_long(c: &mut Criterion) {
    let n = config::ARRAY_LENGTH_LONG;
    let data = utils::get_worst_case_array::<f64>(n, 1.0);
    c.bench_function("simple_worst_long_f64", |b| {
        b.iter(|| argminmax::scalar_argminmax(black_box(data.view())))
    });
    c.bench_function("sse_worst_long_f64", |b| {
        b.iter(|| black_box(unsafe { SSE::argminmax(black_box(data.view())) }))
    });
    c.bench_function("avx2_worst_long_f64", |b| {
        b.iter(|| black_box(unsafe { AVX2::argminmax(black_box(data.view())) }))
    });
    c.bench_function("avx512_worst_long_f64", |b| {
        b.iter(|| black_box(unsafe { AVX512::argminmax(black_box(data.view())) }))
    });
    c.bench_function("impl_worst_long_f64", |b| {
        b.iter(|| black_box(data.view().argminmax()))
    });
}

fn minmax_f64_worst_case_array_short(c: &mut Criterion) {
    let n = config::ARRAY_LENGTH_SHORT;
    let data = utils::get_worst_case_array::<f64>(n, 1.0);
    c.bench_function("simple_worst_short_f64", |b| {
        b.iter(|| argminmax::scalar_argminmax(black_box(data.view())))
    });
    c.bench_function("sse_worst_short_f64", |b| {
        b.iter(|| black_box(unsafe { SSE::argminmax(black_box(data.view())) }))
    });
    c.bench_function("avx2_worst_short_f64", |b| {
        b.iter(|| black_box(unsafe { AVX2::argminmax(black_box(data.view())) }))
    });
    c.bench_function("avx512_worst_short_f64", |b| {
        b.iter(|| black_box(unsafe { AVX512::argminmax(black_box(data.view())) }))
    });
    c.bench_function("impl_worst_short_f64", |b| {
        b.iter(|| black_box(data.view().argminmax()))
    });
}

criterion_group!(
    benches,
    minmax_f64_random_array_long,
    minmax_f64_random_array_short,
    minmax_f64_worst_case_array_long,
    minmax_f64_worst_case_array_short
);
criterion_main!(benches);

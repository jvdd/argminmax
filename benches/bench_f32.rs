#[macro_use]
extern crate criterion;
extern crate dev_utils;

use argminmax::ArgMinMax;
use criterion::{black_box, Criterion};
use dev_utils::{config, utils};

fn minmax_f32_random_array_long(c: &mut Criterion) {
    let n = config::ARRAY_LENGTH_LONG;
    let data = utils::get_random_array::<f32>(n, f32::MIN, f32::MAX);
    c.bench_function("simple_random_long_f32", |b| {
        b.iter(|| argminmax::generic::simple_argminmax(black_box(data.view())))
    });
    c.bench_function("simd_random_long_f32", |b| {
        b.iter(|| black_box(data.view().argminmax()))
    });
}

fn minmax_f32_random_array_short(c: &mut Criterion) {
    let n = config::ARRAY_LENGTH_SHORT;
    let data = utils::get_random_array::<f32>(n, f32::MIN, f32::MAX);
    c.bench_function("simple_random_short_f32", |b| {
        b.iter(|| argminmax::generic::simple_argminmax(black_box(data.view())))
    });
    c.bench_function("simd_random_short_f32", |b| {
        b.iter(|| black_box(data.view().argminmax()))
    });
}

fn minmax_f32_worst_case_array_long(c: &mut Criterion) {
    let n = config::ARRAY_LENGTH_LONG;
    let data = utils::get_worst_case_array::<f32>(n, 1.0);
    c.bench_function("simple_worst_long_f32", |b| {
        b.iter(|| argminmax::generic::simple_argminmax(black_box(data.view())))
    });
    c.bench_function("simd_worst_long_f32", |b| {
        b.iter(|| black_box(data.view().argminmax()))
    });
}

fn minmax_f32_worst_case_array_short(c: &mut Criterion) {
    let n = config::ARRAY_LENGTH_SHORT;
    let data = utils::get_worst_case_array::<f32>(n, 1.0);
    c.bench_function("simple_worst_short_f32", |b| {
        b.iter(|| argminmax::generic::simple_argminmax(black_box(data.view())))
    });
    c.bench_function("simd_worst_short_f32", |b| {
        b.iter(|| black_box(data.view().argminmax()))
    });
}

criterion_group!(
    benches,
    minmax_f32_random_array_long,
    minmax_f32_random_array_short,
    minmax_f32_worst_case_array_long,
    minmax_f32_worst_case_array_short
);
criterion_main!(benches);

#[macro_use]
extern crate criterion;
extern crate dev_utils;

use argminmax::ArgMinMax;
use criterion::{black_box, Criterion};
use dev_utils::{config, utils};

fn minmax_i32_random_array_long(c: &mut Criterion) {
    let n = config::ARRAY_LENGTH_LONG;
    let data = utils::get_random_array::<i32>(n, i32::MIN, i32::MAX);
    c.bench_function("simple_random_long_i32", |b| {
        b.iter(|| argminmax::scalar_argminmax(black_box(data.view())))
    });
    c.bench_function("simd_random_long_i32", |b| {
        b.iter(|| black_box(data.view().argminmax()))
    });
}

fn minmax_i32_random_array_short(c: &mut Criterion) {
    let n = config::ARRAY_LENGTH_SHORT;
    let data = utils::get_random_array::<i32>(n, i32::MIN, i32::MAX);
    c.bench_function("simple_random_short_i32", |b| {
        b.iter(|| argminmax::scalar_argminmax(black_box(data.view())))
    });
    c.bench_function("simd_random_short_i32", |b| {
        b.iter(|| black_box(data.view().argminmax()))
    });
}

fn minmax_i32_worst_case_array_long(c: &mut Criterion) {
    let n = config::ARRAY_LENGTH_LONG;
    let data = utils::get_worst_case_array::<i32>(n, 1);
    c.bench_function("simple_worst_long_i32", |b| {
        b.iter(|| argminmax::scalar_argminmax(black_box(data.view())))
    });
    c.bench_function("simd_worst_long_i32", |b| {
        b.iter(|| black_box(data.view().argminmax()))
    });
}

fn minmax_i32_worst_case_array_short(c: &mut Criterion) {
    let n = config::ARRAY_LENGTH_SHORT;
    let data = utils::get_worst_case_array::<i32>(n, 1);
    c.bench_function("simple_worst_short_i32", |b| {
        b.iter(|| argminmax::scalar_argminmax(black_box(data.view())))
    });
    c.bench_function("simd_worst_short_i32", |b| {
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

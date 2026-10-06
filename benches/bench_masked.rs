use argminmax::ArgMinMaxMasked;
use codspeed_criterion_compat::*;
use dev_utils::{config, utils};

use argminmax::dtype_strategy::{FloatIgnoreNaN, Int};
use argminmax::scalar::{ScalarArgMinMax, SCALAR};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use argminmax::simd::{SIMDArgMinMax, AVX2, AVX512, SSE};
#[cfg(target_arch = "aarch64")]
use argminmax::simd::{SIMDArgMinMax, NEON};

/// A random validity bitmap with ~10% nulls
fn get_validity(n: usize) -> Vec<u8> {
    let random: Vec<u8> = utils::SampleUniformFullRange::get_random_array(n);
    utils::get_validity(n, 0, |i| random[i] >= 26)
}

/// Macro for benchmarking the masked argminmax of a data type (with its DTypeStrategy and
/// the target features that the SSE, AVX2 and AVX512 implementations require)
macro_rules! bench_masked {
    ($bench:ident, $dtype:ty, $dtype_strategy:ty, $sse:tt, $avx2:tt, $avx512:tt) => {
        fn $bench(c: &mut Criterion) {
            let n = config::ARRAY_LENGTH_LONG;
            let data: &[$dtype] = &utils::SampleUniformFullRange::get_random_array(n);
            let validity: &[u8] = &get_validity(n);
            let name = |implementation: &str| {
                format!("{implementation}_{}_argminmax_masked", stringify!($dtype))
            };
            c.bench_function(&name("scalar"), |b| {
                b.iter(|| SCALAR::<$dtype_strategy>::argminmax_masked(black_box(data), validity, 0))
            });
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if is_x86_feature_detected!($sse) {
                c.bench_function(&name("sse"), |b| {
                    b.iter(|| unsafe {
                        SSE::<$dtype_strategy>::argminmax_masked(black_box(data), validity, 0)
                    })
                });
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if is_x86_feature_detected!($avx2) {
                c.bench_function(&name("avx2"), |b| {
                    b.iter(|| unsafe {
                        AVX2::<$dtype_strategy>::argminmax_masked(black_box(data), validity, 0)
                    })
                });
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if is_x86_feature_detected!($avx512) {
                c.bench_function(&name("avx512"), |b| {
                    b.iter(|| unsafe {
                        AVX512::<$dtype_strategy>::argminmax_masked(black_box(data), validity, 0)
                    })
                });
            }
            #[cfg(target_arch = "aarch64")]
            if std::arch::is_aarch64_feature_detected!("neon") {
                c.bench_function(&name("neon"), |b| {
                    b.iter(|| unsafe {
                        NEON::<$dtype_strategy>::argminmax_masked(black_box(data), validity, 0)
                    })
                });
            }
            c.bench_function(&name("impl"), |b| {
                b.iter(|| black_box(data).argminmax_masked(validity, 0))
            });
        }
    };
}

bench_masked!(masked_u8, u8, Int, "sse4.1", "avx2", "avx512bw");
bench_masked!(masked_i16, i16, Int, "sse4.1", "avx2", "avx512bw");
bench_masked!(masked_i32, i32, Int, "sse4.1", "avx2", "avx512f");
bench_masked!(masked_i64, i64, Int, "sse4.2", "avx2", "avx512f");
bench_masked!(masked_f32, f32, FloatIgnoreNaN, "sse4.1", "avx", "avx512f");
bench_masked!(masked_f64, f64, FloatIgnoreNaN, "sse4.1", "avx", "avx512f");

criterion_group!(benches, masked_u8, masked_i16, masked_i32, masked_i64, masked_f32, masked_f64);
criterion_main!(benches);

# ArgMinMax
> Efficient argmin &amp; argmax (in 1 function) with SIMD (SSE, AVX(2), AVX512, NEON) for `f16`, `f32`, `f64`, `i16`, `i32`, `i64`, `u16`, `u32`, `u64` on `ndarray::ArrayView1`

<!-- This project uses [SIMD](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data) to compute argmin and argmax in a single function.   -->

🚀 The function is generic over the type of the array, so it can be used on an `ndarray::ArrayView1<T>` where `T` can be `f16`*, `f32`, `f64`, `i16`, `i32`, `i64`, `u16`, `u32`, `u64`.

👀 Note that this implementation contains no if checks, ensuring that the runtime of the function is independent of the input data its order (best-case = worst-case = average-case).

*for `f16` you should enable the 'half' feature.

## Installing

Add the following to your `Cargo.toml`:

```toml
[dependencies]
argminmax = "0.1.1"
```

## Example usage

```rust
use argminmax::ArgMinMax;  // extension trait for ndarray::ArrayView1
use ndarray::Array1;

let arr: Vec<i32> = (0..200_000).collect();
let arr: Array1<i32> = Array1::from(arr);

let (min, max) = arr.view().argminmax();  // apply extension

println!("min: {}, max: {}", min, max);
println!("arr[min]: {}, arr[max]: {}", arr[min], arr[max]);
```

## Benchmarks

Benchmarks on my laptop *(AMD Ryzen 7 4800U, 1.8 GHz, 16GB RAM)* using [criterion](https://github.com/bheisler/criterion.rs) show that the function is 3-20x faster than the scalar implementation (depending of data type).

See `/benches/results`.

<!-- *For example, finding the argmin & argmax in an array of 10,000,000  random `f32` elements is 3.5x faster than the scalar implementation (taking 2.4ms vs 8.5ms).* -->

Run the benchmarks yourself with the following command:
```bash
cargo bench --quiet --message-format=short --features half | grep "time:"
```

## Tests

To run the tests use the following command:
```bash
cargo test --message-format=short --features half
```

---

## Acknowledgements

Some parts of this library are inspired by the great work of [minimalrust](https://github.com/minimalrust)'s [argmm](https://github.com/minimalrust/argmm) project.

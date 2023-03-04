# ArgMinMax
> Efficient argmin &amp; argmax (in 1 function) with SIMD (SSE, AVX(2), AVX512, NEON) for `f16`, `f32`, `f64`, `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`.

<!-- This project uses [SIMD](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data) to compute argmin and argmax in a single function.   -->

🚀 The function is generic over the type of the array, so it can be used on `&[T]` or `Vec<T>` where `T` can be `f16`<sup>1</sup>, `f32`, `f64`, `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`.

🤝 The trait is implemented for [`slice`](https://doc.rust-lang.org/std/primitive.slice.html), [`Vec`](https://doc.rust-lang.org/std/vec/struct.Vec.html), 1D [`ndarray::ArrayBase`](https://docs.rs/ndarray/latest/ndarray/struct.ArrayBase.html)<sup>2</sup>, and apache [`arrow::PrimitiveArray`](https://docs.rs/arrow/latest/arrow/array/struct.PrimitiveArray.html)<sup>3</sup>.

⚡ **Runtime CPU feature detection** is used to select the most efficient implementation for the current CPU. This means that the same binary can be used on different CPUs without recompilation. 

👀 The SIMD implementation contains **no if checks**, ensuring that the runtime of the function is independent of the input data its order (best-case = worst-case = average-case).

🪄 **Efficient support for f16 and uints**: through (bijective aka symmetric) bitwise operations, f16 (optional<sup>1</sup>) and uints are converted to ordered integers, allowing to use integer SIMD instructions.


> <i><sup>1</sup> for <code>f16</code> you should enable the `"half"` feature.</i>  
> <i><sup>2</sup> for <code>ndarray::ArrayBase</code> you should enable the `"ndarray"` feature.</i>  
> <i><sup>3</sup> for <code>arrow::PrimitiveArray</code> you should enable the `"arrow"` feature.</i>  

## Installing

Add the following to your `Cargo.toml`:

```toml
[dependencies]
argminmax = "0.4"
```

## Example usage

```rust
use argminmax::ArgMinMax;  // import trait

let arr: Vec<i32> = (0..200_000).collect();  // create a vector

let (min, max) = arr.argminmax();  // apply extension

println!("min: {}, max: {}", min, max);
println!("arr[min]: {}, arr[max]: {}", arr[min], arr[max]);
```

## Features
- **"half"**: support `f16` argminmax (through using the [`half`](https://docs.rs/half/latest/half) crate).
- **"ndarray"**: add `ArgMinMax` trait to [`ndarray`](https://docs.rs/ndarray/latest/ndarray) its `Array1` & `ArrayView1`.
- **"arrow"**: add `ArgMinMax` trait to [`arrow`](https://docs.rs/arrow/latest/arrow) its `PrimitiveArray`.

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
cargo test --message-format=short --all-features
```

## Limitations

The library handles NaNs! 🚀 

<!-- For NaN-handling there are two variants:
- **Ignore NaN**: NaNs are ignored and the index of the highest / lowest non-NaN value is returned.
- **Return NaN**: the first NaN value is returned. -->

❗ Some (minor) caveats for the NaN-handling:
- **Ignore NaN**: in some cases (when the array contains exclusively NaNs and/or +/-Inf) the function will return 0.
- **Return NaN**: the first NaN value is only returned *iff* all NaN values have the same bit representation.   
   => when NaN values have different bit representations then the index of the highest / lowest `ord_transform` is returned.

Tip 💡: if you know that there are no NaNs in your the array, we advise you to use the `IgnoreNaN` variant, as this is 5-30% faster than the `ReturnNaN` variant.


❗ The "half" feature (f16 support) is not yet fully supported:
- [x] `ReturnNaN` variant
- [ ] `IgnoreNaN` variant -> currently defaults to the `ReturnNaN` variant

---

## Acknowledgements

Some parts of this library are inspired by the great work of [minimalrust](https://github.com/minimalrust)'s [argmm](https://github.com/minimalrust/argmm) project.

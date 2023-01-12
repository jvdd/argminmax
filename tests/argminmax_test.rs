use argminmax::ArgMinMax;
use dev_utils::utils;

#[cfg(feature = "ndarray")]
use ndarray::Array1;

const ARRAY_LENGTH: usize = 100_000;

#[test]
fn test_argminmax_slice() {
    // Test slice (aka the base implementation)
    let data: &[f32] = &(0..ARRAY_LENGTH).map(|x| x as f32).collect::<Vec<f32>>();
    let (min, max) = data.argminmax();
    assert_eq!(min, 0);
    assert_eq!(max, ARRAY_LENGTH - 1);
    // Borrowed slice
    let (min, max) = (&data).argminmax();
    assert_eq!(min, 0);
    assert_eq!(max, ARRAY_LENGTH - 1);
}

// TODO: this is currently not supported yet
// #[test]
// fn test_argminmax_array() {
//     // Test array
//     let data: [f32; ARRAY_LENGTH] = (0..ARRAY_LENGTH).map(|x| x as f32).collect::<Vec<f32>>().try_into().unwrap();
//     let (min, max) = data.argminmax();
//     assert_eq!(min, 0);
//     assert_eq!(max, ARRAY_LENGTH - 1);
// }

#[test]
fn test_argminmax_vec() {
    let data: Vec<f32> = (0..ARRAY_LENGTH).map(|x| x as f32).collect();
    // Test owned vec
    let (min, max) = data.argminmax();
    assert_eq!(min, 0);
    assert_eq!(max, ARRAY_LENGTH - 1);
    // Test borrowed vec
    let (min, max) = (&data).argminmax();
    assert_eq!(min, 0);
    assert_eq!(max, ARRAY_LENGTH - 1);
}

#[cfg(feature = "ndarray")]
#[test]
fn test_argminmax_ndarray() {
    let data: Array1<f32> = Array1::from((0..ARRAY_LENGTH).map(|x| x as f32).collect::<Vec<f32>>());
    // --- Array1
    // Test owned Array1
    let (min, max) = data.argminmax();
    assert_eq!(min, 0);
    assert_eq!(max, ARRAY_LENGTH - 1);
    // Test borrowed Array1
    let (min, max) = (&data).argminmax();
    assert_eq!(min, 0);
    assert_eq!(max, ARRAY_LENGTH - 1);
    // --- ArrayView1
    // Test owened ArrayView1
    let (min, max) = data.view().argminmax();
    assert_eq!(min, 0);
    assert_eq!(max, ARRAY_LENGTH - 1);
    // Test borrowed ArrayView1
    let (min, max) = (&data.view()).argminmax();
    assert_eq!(min, 0);
    assert_eq!(max, ARRAY_LENGTH - 1);
}

#[test]
fn test_argminmax_many_random_runs() {
    for _ in 0..500 {
        let data: Vec<f32> = utils::get_random_array::<f32>(5_000, f32::MIN, f32::MAX);
        // Slice
        let slice: &[f32] = &data;
        let (min_slice, max_slice) = slice.argminmax();
        // Vec
        let (min_vec, max_vec) = data.argminmax();
        // Array1
        #[cfg(feature = "ndarray")]
        let array: Array1<f32> = Array1::from(data);
        #[cfg(feature = "ndarray")]
        let (min_array, max_array) = array.argminmax();
        // Assert
        assert_eq!(min_slice, min_vec);
        assert_eq!(max_slice, max_vec);
        #[cfg(feature = "ndarray")]
        assert_eq!(min_slice, min_array);
        #[cfg(feature = "ndarray")]
        assert_eq!(max_slice, max_array);
    }
}

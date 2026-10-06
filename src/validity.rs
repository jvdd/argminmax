//! Skipping the null elements of an array, given its validity bitmap.
//!
//! The validity bitmap follows the [Arrow format](https://arrow.apache.org/docs/format/Columnar.html#validity-bitmaps):
//! element `i` is valid (i.e., not null) when bit `offset + i` of the bitmap is set,
//! where the bits are numbered from the least significant bit of the first byte.

use crate::scalar::ScalarArgMinMax;

/// The index and value of the minimum (or maximum) - `None` when there is no valid element
pub(crate) type Found<T> = Option<(usize, T)>;
/// The index and value of the minimum and of the maximum
pub(crate) type FoundMinMax<T> = (Found<T>, Found<T>);

/// Panics when the validity bitmap has less than `offset + len` bits
pub(crate) fn assert_validity_len(validity: &[u8], offset: usize, len: usize) {
    let nb_bits = offset.checked_add(len);
    assert!(
        nb_bits.is_some_and(|nb_bits| nb_bits.div_ceil(8) <= validity.len()),
        "The validity bitmap is too short for the data"
    );
}

/// Returns the validity bits of the elements `i..i + 64` (bit `j` is the validity of
/// element `i + j`). The bits beyond the end of the bitmap are 0.
#[inline(always)]
pub(crate) fn validity_word(validity: &[u8], offset: usize, i: usize) -> u64 {
    let bit = offset + i;
    let byte = bit / 8;
    // The 64 bits span at most 9 bytes, which a 16 byte load covers
    let bytes: [u8; 16] = match validity.get(byte..byte + 16) {
        Some(bytes) => bytes.try_into().unwrap(),
        None => {
            let mut bytes = [0; 16];
            let tail = validity.get(byte..).unwrap_or_default();
            bytes[..tail.len()].copy_from_slice(tail);
            bytes
        }
    };
    (u128::from_le_bytes(bytes) >> (bit % 8)) as u64
}

/// Returns an iterator over the blocks of (at most) 64 elements: the index of the first
/// element of the block and the validity bits of its elements.
fn blocks(validity: &[u8], offset: usize, len: usize) -> impl Iterator<Item = (usize, u64)> + '_ {
    (0..len).step_by(64).map(move |start| {
        let block_len = 64.min(len - start);
        let bits = validity_word(validity, offset, start) & (u64::MAX >> (64 - block_len));
        (start, bits)
    })
}

/// Returns the first valid element that equals `value`
pub(crate) fn first_valid_eq<T: Copy + PartialEq>(
    arr: &[T],
    validity: &[u8],
    offset: usize,
    value: T,
) -> Found<T> {
    for (start, mut bits) in blocks(validity, offset, arr.len()) {
        while bits != 0 {
            let i = start + bits.trailing_zeros() as usize;
            if arr[i] == value {
                return Some((i, value));
            }
            bits &= bits - 1;
        }
    }
    None
}

/// Combines the minimum of two consecutive parts of an array (`a` comes first).
/// `SCALAR` decides, so the NaN handling and ties are the same as when the parts are
/// one array.
#[inline(always)]
pub(crate) fn merge_min<T: Copy + PartialOrd, SCALAR: ScalarArgMinMax<T>>(
    a: Found<T>,
    b: Found<T>,
) -> Found<T> {
    match (a, b) {
        (Some(a), Some(b)) => Some(if SCALAR::argmin(&[a.1, b.1]) == 0 {
            a
        } else {
            b
        }),
        (a, b) => a.or(b),
    }
}

/// Combines the maximum of two consecutive parts of an array (`a` comes first).
/// `SCALAR` decides, so the NaN handling and ties are the same as when the parts are
/// one array.
#[inline(always)]
pub(crate) fn merge_max<T: Copy + PartialOrd, SCALAR: ScalarArgMinMax<T>>(
    a: Found<T>,
    b: Found<T>,
) -> Found<T> {
    match (a, b) {
        (Some(a), Some(b)) => Some(if SCALAR::argmax(&[a.1, b.1]) == 0 {
            a
        } else {
            b
        }),
        (a, b) => a.or(b),
    }
}

/// The minimum (if `MIN`) and maximum (if `MAX`) of the valid elements, by applying
/// `SCALAR` on each run of fully valid blocks of 64 elements, and on the valid elements
/// of each other block. The latter follow the minimum and maximum so far: these stay the
/// result on ties, and `SCALAR` rarely updates them (as in one pass over the array).
#[inline(always)]
pub(crate) fn scalar_masked<T, SCALAR, const MIN: bool, const MAX: bool>(
    arr: &[T],
    validity: &[u8],
    offset: usize,
) -> FoundMinMax<T>
where
    T: Copy + PartialOrd,
    SCALAR: ScalarArgMinMax<T>,
{
    let (mut min, mut max) = (None, None);
    let Some(&first) = arr.first() else {
        return (min, max);
    };
    // The minimum and maximum so far, followed by the valid elements of a block
    let mut values = [first; 66];
    let mut indices = [0; 66];
    let mut blocks = blocks(validity, offset, arr.len()).peekable();
    while let Some((start, bits)) = blocks.next() {
        if bits == u64::MAX {
            let mut end = start + 64;
            while blocks.next_if(|&(_, bits)| bits == u64::MAX).is_some() {
                end += 64;
            }
            let run = &arr[start..end];
            let (min_index, max_index) = scalar_min_max::<T, SCALAR, MIN, MAX>(run);
            if MIN {
                min = merge_min::<T, SCALAR>(min, Some((start + min_index, run[min_index])));
            }
            if MAX {
                max = merge_max::<T, SCALAR>(max, Some((start + max_index, run[max_index])));
            }
        } else if bits != 0 {
            let mut len = 0;
            for (index, value) in [min, max].into_iter().flatten() {
                (indices[len], values[len]) = (index, value);
                len += 1;
            }
            let mut remaining = bits;
            while remaining != 0 {
                let i = start + remaining.trailing_zeros() as usize;
                (indices[len], values[len]) = (i, arr[i]);
                len += 1;
                remaining &= remaining - 1;
            }
            let (min_index, max_index) = scalar_min_max::<T, SCALAR, MIN, MAX>(&values[..len]);
            if MIN {
                min = Some((indices[min_index], values[min_index]));
            }
            if MAX {
                max = Some((indices[max_index], values[max_index]));
            }
        }
    }
    (min, max)
}

/// The index of the minimum (if `MIN`) and of the maximum (if `MAX`, else 0) by `SCALAR`
#[inline(always)]
fn scalar_min_max<T, SCALAR, const MIN: bool, const MAX: bool>(values: &[T]) -> (usize, usize)
where
    T: Copy + PartialOrd,
    SCALAR: ScalarArgMinMax<T>,
{
    match (MIN, MAX) {
        (true, true) => SCALAR::argminmax(values),
        (true, false) => (SCALAR::argmin(values), 0),
        (false, _) => (0, SCALAR::argmax(values)),
    }
}

// ======================================= TESTS =======================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validity_word() {
        let validity: Vec<u8> = (0..20).collect();
        // Every bit offset, also near the end of the bitmap
        for bit in 0..validity.len() * 8 {
            let expected = (0..64)
                .filter(|j| {
                    let b = bit + j;
                    b / 8 < validity.len() && (validity[b / 8] >> (b % 8)) & 1 == 1
                })
                .fold(0u64, |word, j| word | (1 << j));
            assert_eq!(validity_word(&validity, bit, 0), expected);
            assert_eq!(validity_word(&validity, 0, bit), expected);
        }
    }

    #[test]
    fn test_blocks() {
        // The bits beyond the length are cleared
        let validity = [0xFF; 20];
        let blocks = |offset, len| blocks(&validity, offset, len).collect::<Vec<_>>();
        assert_eq!(blocks(3, 0), vec![]);
        assert_eq!(blocks(3, 5), vec![(0, 0b11111)]);
        assert_eq!(blocks(3, 64), vec![(0, u64::MAX)]);
        assert_eq!(
            blocks(3, 130),
            vec![(0, u64::MAX), (64, u64::MAX), (128, 0b11)]
        );
    }

    #[test]
    #[should_panic(expected = "The validity bitmap is too short")]
    fn test_validity_too_short() {
        assert_validity_len(&[0xFF; 2], 1, 16);
    }
}

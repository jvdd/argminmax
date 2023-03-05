/// The strategy that is used to handle the data type.

pub struct Int; // Signed & Unsigned Integers
#[cfg(any(feature = "float", feature = "half"))]
pub struct FloatIgnoreNaN; // Floats, ignoring NaNs
#[cfg(any(feature = "float", feature = "half"))]
pub struct FloatReturnNaN; // Floats, returning NaNs

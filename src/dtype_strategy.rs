/// The strategy that is used to handle the data type.

pub(crate) struct Int; // Signed & Unsigned Integers
pub(crate) struct FloatIgnoreNaN; // Floats, ignoring NaNs
pub(crate) struct FloatReturnNaN; // Floats, returning NaNs

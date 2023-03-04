mod generic;
pub use generic::{ScalarArgMinMax, SCALAR};
// Data type specific modules
#[cfg(feature = "half")]
mod scalar_f16;

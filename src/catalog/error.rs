//! Error types for catalog operations.

use thiserror::Error;

/// Errors that can occur during catalog/index operations.
#[derive(Error, Debug)]
pub enum CatalogError {
    /// Invalid magic number in index file.
    #[error("invalid index file: bad magic number")]
    InvalidMagic,

    /// Unsupported index version.
    #[error("unsupported index version: {0}")]
    UnsupportedVersion(u16),

    /// Index file is truncated or corrupted.
    #[error("index file is truncated or corrupted")]
    Truncated,

    /// I/O error reading index file.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Star index out of bounds.
    #[error("star index {0} out of bounds (max {1})")]
    StarIndexOutOfBounds(u32, u32),

    /// Pattern index out of bounds.
    #[error("pattern index out of bounds")]
    PatternIndexOutOfBounds,

    /// Hash bin out of bounds.
    #[error("hash bin {0} out of bounds (max {1})")]
    HashBinOutOfBounds(u32, u32),

    /// Index header contains invalid metadata.
    #[error("malformed index header: {0}")]
    MalformedData(String),
}

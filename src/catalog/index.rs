//! Index file format and memory-mapped reader.

use std::path::Path;
use std::fs::File;
use memmap2::Mmap;
use bytemuck::{Pod, Zeroable, from_bytes, cast_slice};

use super::error::CatalogError;
use super::star::{PackedStar, PackedPattern};
use crate::core::types::CatalogStar;

/// Magic number for index files: "CHAM"
pub const INDEX_MAGIC: [u8; 4] = *b"CHAM";

/// Current index format version.
pub const INDEX_VERSION: u16 = 1;

/// Index file header.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct IndexHeader {
    /// Magic number: "CHAM"
    pub magic: [u8; 4],
    /// Format version.
    pub version: u16,
    /// Reserved for flags.
    pub flags: u16,
    /// Number of stars in the catalog.
    pub num_stars: u32,
    /// Number of patterns in the index.
    pub num_patterns: u32,
    /// Number of hash bins.
    pub num_bins: u32,
    /// Minimum FOV this index is designed for (degrees).
    pub fov_min_deg: f32,
    /// Maximum FOV this index is designed for (degrees).
    pub fov_max_deg: f32,
    /// Magnitude limit of included stars.
    pub mag_limit: f32,
    /// Pattern size (number of stars per pattern, typically 4).
    pub pattern_size: u8,
    /// Reserved padding to reach 64 bytes total.
    pub _reserved: [u8; 31],
}

/// Size of the index header in bytes.
pub const HEADER_SIZE: usize = 64; // Must match std::mem::size_of::<IndexHeader>()

impl IndexHeader {
    /// Create a new header with the given parameters.
    pub fn new(
        num_stars: u32,
        num_patterns: u32,
        num_bins: u32,
        fov_min_deg: f32,
        fov_max_deg: f32,
        mag_limit: f32,
    ) -> Self {
        Self {
            magic: INDEX_MAGIC,
            version: INDEX_VERSION,
            flags: 0,
            num_stars,
            num_patterns,
            num_bins,
            fov_min_deg,
            fov_max_deg,
            mag_limit,
            pattern_size: 4,
            _reserved: [0; 31],
        }
    }

    /// Validate the header.
    pub fn validate(&self) -> Result<(), CatalogError> {
        if self.magic != INDEX_MAGIC {
            return Err(CatalogError::InvalidMagic);
        }
        if self.version != INDEX_VERSION {
            return Err(CatalogError::UnsupportedVersion(self.version));
        }
        Ok(())
    }
}

/// Memory-mapped index for efficient star pattern lookups.
pub struct Index {
    /// Memory-mapped file data.
    _mmap: Mmap,
    /// Parsed header.
    header: IndexHeader,
    /// Slice of packed stars.
    stars: &'static [PackedStar],
    /// Slice of bin offsets (num_bins + 1 entries).
    /// bin_offsets[i] is the start of bin i in patterns array.
    /// bin_offsets[i+1] - bin_offsets[i] is the number of patterns in bin i.
    bin_offsets: &'static [u32],
    /// Slice of all patterns.
    patterns: &'static [PackedPattern],
}

impl Index {
    /// Open an index file from disk.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, CatalogError> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        // Parse header
        if mmap.len() < HEADER_SIZE {
            return Err(CatalogError::Truncated);
        }

        let header: IndexHeader = *from_bytes(&mmap[..HEADER_SIZE]);
        header.validate()?;

        // Calculate section offsets
        let stars_offset = HEADER_SIZE;
        let stars_size = header.num_stars as usize * std::mem::size_of::<PackedStar>();
        let bins_offset = stars_offset + stars_size;
        let bins_size = (header.num_bins as usize + 1) * std::mem::size_of::<u32>();
        let patterns_offset = bins_offset + bins_size;
        let patterns_size = header.num_patterns as usize * std::mem::size_of::<PackedPattern>();

        // Verify file size
        if mmap.len() < patterns_offset + patterns_size {
            return Err(CatalogError::Truncated);
        }

        // SAFETY: We're casting aligned, validated data from the mmap.
        // The 'static lifetime is valid because we keep the mmap alive.
        let stars: &'static [PackedStar] = unsafe {
            let ptr = mmap.as_ptr().add(stars_offset) as *const PackedStar;
            std::slice::from_raw_parts(ptr, header.num_stars as usize)
        };

        let bin_offsets: &'static [u32] = unsafe {
            let ptr = mmap.as_ptr().add(bins_offset) as *const u32;
            std::slice::from_raw_parts(ptr, header.num_bins as usize + 1)
        };

        let patterns: &'static [PackedPattern] = unsafe {
            let ptr = mmap.as_ptr().add(patterns_offset) as *const PackedPattern;
            std::slice::from_raw_parts(ptr, header.num_patterns as usize)
        };

        Ok(Self {
            _mmap: mmap,
            header,
            stars,
            bin_offsets,
            patterns,
        })
    }

    /// Get the index header.
    #[inline]
    pub fn header(&self) -> &IndexHeader {
        &self.header
    }

    /// Get the number of stars.
    #[inline]
    pub fn num_stars(&self) -> u32 {
        self.header.num_stars
    }

    /// Get the number of patterns.
    #[inline]
    pub fn num_patterns(&self) -> u32 {
        self.header.num_patterns
    }

    /// Get the number of hash bins.
    #[inline]
    pub fn num_bins(&self) -> u32 {
        self.header.num_bins
    }

    /// Get a star by index.
    #[inline]
    pub fn get_star(&self, idx: u32) -> Result<&PackedStar, CatalogError> {
        self.stars
            .get(idx as usize)
            .ok_or(CatalogError::StarIndexOutOfBounds(idx, self.header.num_stars))
    }

    /// Get a star as CatalogStar.
    #[inline]
    pub fn get_catalog_star(&self, idx: u32) -> Result<CatalogStar, CatalogError> {
        let packed = self.get_star(idx)?;
        Ok(packed.to_catalog_star(idx))
    }

    /// Get all patterns in a hash bin.
    pub fn get_patterns_in_bin(&self, bin: u32) -> Result<&[PackedPattern], CatalogError> {
        if bin >= self.header.num_bins {
            return Err(CatalogError::HashBinOutOfBounds(bin, self.header.num_bins));
        }

        let start = self.bin_offsets[bin as usize] as usize;
        let end = self.bin_offsets[bin as usize + 1] as usize;

        if end > self.patterns.len() {
            return Err(CatalogError::PatternIndexOutOfBounds);
        }

        Ok(&self.patterns[start..end])
    }

    /// Iterate over all stars.
    pub fn stars(&self) -> impl Iterator<Item = (u32, &PackedStar)> {
        self.stars.iter().enumerate().map(|(i, s)| (i as u32, s))
    }

    /// Get the FOV range this index is designed for.
    pub fn fov_range_deg(&self) -> (f32, f32) {
        (self.header.fov_min_deg, self.header.fov_max_deg)
    }

    /// Get the magnitude limit.
    pub fn mag_limit(&self) -> f32 {
        self.header.mag_limit
    }

    /// Get all stars as a slice.
    #[inline]
    pub fn all_stars(&self) -> &[PackedStar] {
        self.stars
    }
}

// Index is Send + Sync because mmap data is immutable
unsafe impl Send for Index {}
unsafe impl Sync for Index {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_size() {
        // Ensure header is exactly 64 bytes for alignment
        assert_eq!(HEADER_SIZE, 64);
    }

    #[test]
    fn test_header_validation() {
        let valid = IndexHeader::new(100, 1000, 10000, 10.0, 30.0, 7.0);
        assert!(valid.validate().is_ok());

        let mut invalid = valid;
        invalid.magic = *b"XXXX";
        assert!(matches!(invalid.validate(), Err(CatalogError::InvalidMagic)));

        let mut invalid = valid;
        invalid.version = 99;
        assert!(matches!(
            invalid.validate(),
            Err(CatalogError::UnsupportedVersion(99))
        ));
    }
}

//! Index file format and memory-mapped reader.

use bytemuck::{from_bytes, Pod, Zeroable};
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

use super::error::CatalogError;
use super::star::{PackedPattern, PackedStar};
use crate::core::types::CatalogStar;

/// Magic number for index files: "CHAM"
pub const INDEX_MAGIC: [u8; 4] = *b"CHAM";

/// Current index format version.
pub const INDEX_VERSION: u16 = 4;

/// Default hash quantization bins per ratio dimension.
pub const DEFAULT_HASH_BINS_PER_DIM: u8 = 100;

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
    /// Number of logarithmic angular scale bands.
    pub num_scale_bands: u8,
    /// Ratio quantization bins per dimension used for hashing.
    pub hash_bins_per_dim: u8,
    /// Reserved for future header flags.
    pub _reserved0: u8,
    /// Reserved padding to reach 64 bytes total.
    pub _reserved: [u8; 28],
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
        num_scale_bands: u8,
        hash_bins_per_dim: u8,
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
            num_scale_bands: num_scale_bands.max(1),
            hash_bins_per_dim: hash_bins_per_dim.max(2),
            _reserved0: 0,
            _reserved: [0; 28],
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
        if self.num_scale_bands == 0 {
            return Err(CatalogError::MalformedData(
                "num_scale_bands must be >= 1".to_string(),
            ));
        }
        if self.hash_bins_per_dim < 2 {
            return Err(CatalogError::MalformedData(
                "hash_bins_per_dim must be >= 2".to_string(),
            ));
        }
        Ok(())
    }
}

/// Memory-mapped index for efficient star pattern lookups.
pub struct Index {
    /// Memory-mapped file data.
    _mmap: Mmap,
    /// Size of the mapped index file in bytes.
    mapped_len_bytes: usize,
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
            mapped_len_bytes: mmap.len(),
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

    /// Get the number of angular scale bands in this index.
    #[inline]
    pub fn num_scale_bands(&self) -> u8 {
        self.header.num_scale_bands.max(1)
    }

    /// Get hash quantization bins per dimension.
    #[inline]
    pub fn hash_bins_per_dim(&self) -> u8 {
        self.header.hash_bins_per_dim.max(2)
    }

    /// Get a star by index.
    #[inline]
    pub fn get_star(&self, idx: u32) -> Result<&PackedStar, CatalogError> {
        self.stars
            .get(idx as usize)
            .ok_or(CatalogError::StarIndexOutOfBounds(
                idx,
                self.header.num_stars,
            ))
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

    /// Get the global hash-bin range [start, end) for a given angular scale band.
    pub fn scale_band_bin_range(&self, scale_band: u8) -> (u32, u32) {
        let num_scale_bands = self.num_scale_bands() as u32;
        let band = (scale_band as u32).min(num_scale_bands.saturating_sub(1));
        let start = band * self.header.num_bins / num_scale_bands;
        let end = ((band + 1) * self.header.num_bins / num_scale_bands).max(start + 1);
        (start, end)
    }

    /// Number of hash bins assigned to a scale band.
    pub fn bins_in_scale_band(&self, scale_band: u8) -> u32 {
        let (start, end) = self.scale_band_bin_range(scale_band);
        (end - start).max(1)
    }

    /// Map a local per-band hash bin into the global hash-bin space.
    pub fn local_to_global_bin(&self, scale_band: u8, local_bin: u32) -> u32 {
        let (start, end) = self.scale_band_bin_range(scale_band);
        let band_bins = (end - start).max(1);
        start + (local_bin % band_bins)
    }

    /// Quantize an angular scale (degrees) into a scale band id.
    pub fn scale_band_for_angle_deg(&self, angle_deg: f64) -> u8 {
        let bands = self.num_scale_bands() as f64;
        if bands <= 1.0 {
            return 0;
        }

        let min_scale = self.header.fov_min_deg.max(1e-3) as f64;
        let max_scale = self.header.fov_max_deg.max(self.header.fov_min_deg + 1e-3) as f64;
        let angle = angle_deg.clamp(min_scale, max_scale);
        let ratio = (angle / min_scale).ln() / (max_scale / min_scale).ln();
        let idx = (ratio.clamp(0.0, 0.999_999) * bands).floor() as u8;
        idx.min(self.num_scale_bands().saturating_sub(1))
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

    /// Get mapped index file size in bytes.
    #[inline]
    pub fn mapped_len_bytes(&self) -> usize {
        self.mapped_len_bytes
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
        let valid = IndexHeader::new(
            100,
            1000,
            10000,
            10.0,
            30.0,
            7.0,
            8,
            DEFAULT_HASH_BINS_PER_DIM,
        );
        assert!(valid.validate().is_ok());

        let mut invalid = valid;
        invalid.magic = *b"XXXX";
        assert!(matches!(
            invalid.validate(),
            Err(CatalogError::InvalidMagic)
        ));

        let mut invalid = valid;
        invalid.version = 99;
        assert!(matches!(
            invalid.validate(),
            Err(CatalogError::UnsupportedVersion(99))
        ));
    }
}

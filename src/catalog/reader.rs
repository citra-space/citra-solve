//! Alternative index readers for different environments.

use std::io::{Read, Seek, SeekFrom, BufReader};
use std::fs::File;
use std::path::Path;

use bytemuck::from_bytes;

use super::error::CatalogError;
use super::index::{IndexHeader, INDEX_MAGIC, INDEX_VERSION, HEADER_SIZE};
use super::star::{PackedStar, PackedPattern};

/// Streaming index reader for systems without memory mapping.
///
/// This reader loads data on-demand from disk, suitable for systems
/// with limited RAM or no MMU support.
pub struct StreamingIndex {
    file: BufReader<File>,
    header: IndexHeader,
    stars_offset: u64,
    bins_offset: u64,
    patterns_offset: u64,
    /// Cached bin offsets (loaded once).
    bin_offsets: Vec<u32>,
}

impl StreamingIndex {
    /// Open a streaming index reader.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, CatalogError> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Read header
        let mut header_bytes = [0u8; HEADER_SIZE];
        reader.read_exact(&mut header_bytes)?;
        let header: IndexHeader = *from_bytes(&header_bytes);
        header.validate()?;

        // Calculate offsets
        let stars_offset = HEADER_SIZE as u64;
        let stars_size = header.num_stars as u64 * std::mem::size_of::<PackedStar>() as u64;
        let bins_offset = stars_offset + stars_size;
        let bins_size = (header.num_bins as u64 + 1) * std::mem::size_of::<u32>() as u64;
        let patterns_offset = bins_offset + bins_size;

        // Load bin offsets (small enough to keep in memory)
        reader.seek(SeekFrom::Start(bins_offset))?;
        let mut bin_offsets = vec![0u32; header.num_bins as usize + 1];
        for offset in &mut bin_offsets {
            let mut bytes = [0u8; 4];
            reader.read_exact(&mut bytes)?;
            *offset = u32::from_le_bytes(bytes);
        }

        Ok(Self {
            file: reader,
            header,
            stars_offset,
            bins_offset,
            patterns_offset,
            bin_offsets,
        })
    }

    /// Get the header.
    pub fn header(&self) -> &IndexHeader {
        &self.header
    }

    /// Read a single star by index.
    pub fn read_star(&mut self, idx: u32) -> Result<PackedStar, CatalogError> {
        if idx >= self.header.num_stars {
            return Err(CatalogError::StarIndexOutOfBounds(idx, self.header.num_stars));
        }

        let offset = self.stars_offset + idx as u64 * std::mem::size_of::<PackedStar>() as u64;
        self.file.seek(SeekFrom::Start(offset))?;

        let mut bytes = [0u8; std::mem::size_of::<PackedStar>()];
        self.file.read_exact(&mut bytes)?;
        Ok(*from_bytes(&bytes))
    }

    /// Read a range of stars.
    pub fn read_stars(&mut self, start: u32, count: u32) -> Result<Vec<PackedStar>, CatalogError> {
        if start + count > self.header.num_stars {
            return Err(CatalogError::StarIndexOutOfBounds(
                start + count - 1,
                self.header.num_stars,
            ));
        }

        let offset = self.stars_offset + start as u64 * std::mem::size_of::<PackedStar>() as u64;
        self.file.seek(SeekFrom::Start(offset))?;

        let mut stars = Vec::with_capacity(count as usize);
        for _ in 0..count {
            let mut bytes = [0u8; std::mem::size_of::<PackedStar>()];
            self.file.read_exact(&mut bytes)?;
            stars.push(*from_bytes(&bytes));
        }
        Ok(stars)
    }

    /// Read all patterns in a hash bin.
    pub fn read_patterns_in_bin(&mut self, bin: u32) -> Result<Vec<PackedPattern>, CatalogError> {
        if bin >= self.header.num_bins {
            return Err(CatalogError::HashBinOutOfBounds(bin, self.header.num_bins));
        }

        let start_pattern = self.bin_offsets[bin as usize];
        let end_pattern = self.bin_offsets[bin as usize + 1];
        let count = end_pattern - start_pattern;

        if count == 0 {
            return Ok(Vec::new());
        }

        let offset =
            self.patterns_offset + start_pattern as u64 * std::mem::size_of::<PackedPattern>() as u64;
        self.file.seek(SeekFrom::Start(offset))?;

        let mut patterns = Vec::with_capacity(count as usize);
        for _ in 0..count {
            let mut bytes = [0u8; std::mem::size_of::<PackedPattern>()];
            self.file.read_exact(&mut bytes)?;
            patterns.push(*from_bytes(&bytes));
        }
        Ok(patterns)
    }

    /// Get the number of patterns in a bin without reading them.
    pub fn patterns_in_bin_count(&self, bin: u32) -> Result<u32, CatalogError> {
        if bin >= self.header.num_bins {
            return Err(CatalogError::HashBinOutOfBounds(bin, self.header.num_bins));
        }
        Ok(self.bin_offsets[bin as usize + 1] - self.bin_offsets[bin as usize])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Integration tests would go here with actual index files
}

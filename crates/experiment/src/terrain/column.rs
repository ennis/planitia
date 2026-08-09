use crate::terrain::{PackedTerrVec, TerrVec};
use bumpalo::Bump;

/*
impl<'a> TerrColumn<'a> {
    pub fn new(slices: &'a [TerrSlice]) -> Self {
        assert!(slices.first().map(|s| s.low) == Some(0));
        assert!(slices.last().map(|s| s.high) == Some(u16::MAX));
        TerrColumn { ranges: slices }
    }
}*/

/// Samples the column at the given height, returning the packed feature vector.
pub fn sample_column(column: &[TerrSlice], height: u16) -> PackedTerrVec {
    for slice in column {
        if height >= slice.low && height < slice.high {
            return slice.value;
        }
    }
    panic!("height out of bounds");
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct TerrSlice {
    /// Lower height bound.
    pub low: u16,
    /// Upper height bound.
    pub high: u16,
    /// Packed feature vector.
    pub value: PackedTerrVec,
}

impl TerrSlice {
    pub const fn height(&self) -> u16 {
        self.high - self.low
    }

    pub fn merged(&self, other: &TerrSlice) -> TerrSlice {
        assert!(self.high == other.low || self.low == other.high, "slices must be adjacent to merge");

        let self_height = self.height() as f32;
        let other_height = other.height() as f32;

        let high = self.high.max(other.high);
        let low = self.low.min(other.low);

        let v1 = TerrVec::unpack(self.value);
        let v2 = TerrVec::unpack(other.value);

        let avg = (self_height * v1 + other_height * v2) / (self_height + other_height);
        let value = avg.pack();
        TerrSlice { low, high, value }
    }
}

impl Default for TerrSlice {
    fn default() -> Self {
        TerrSlice { low: 0, high: u16::MAX, value: PackedTerrVec::default() }
    }
}

pub fn downsample_columns<'a>(
    arena: &'a Bump,
    columns: &[&[TerrSlice]],
    merge_height_threshold: u16,
) -> &'a [TerrSlice] {
    assert!(columns.len() == 4, "expected 4 columns for downsampling");

    // collect all bounds
    let mut bounds = vec![];
    for col in columns {
        for slice in *col {
            bounds.push(slice.low);
            bounds.push(slice.high);
        }
    }

    // remove duplicates
    bounds.sort_unstable();
    bounds.dedup();

    assert!(bounds.first() == Some(&0));
    assert!(bounds.last() == Some(&u16::MAX));

    let mut fused = vec![TerrSlice::default(); bounds.len() - 1];

    for (i, b) in bounds[1..].iter().enumerate() {
        let b = *b - 1;
        let v1 = TerrVec::unpack(sample_column(columns[0], b));
        let v2 = TerrVec::unpack(sample_column(columns[1], b));
        let v3 = TerrVec::unpack(sample_column(columns[2], b));
        let v4 = TerrVec::unpack(sample_column(columns[3], b));
        fused[i].low = bounds[i];
        fused[i].high = bounds[i + 1];
        fused[i].value = ((v1 + v2 + v3 + v4) * 0.25).pack();
    }

    // merge slices smaller than the threshold with the previous or the next (whichever is smaller)

    let mut i = 0;
    while i < fused.len() {
        if fused.len() == 1 {
            break;
        }
        let h = fused[i].high - fused[i].low;
        if h < merge_height_threshold {
            if i == 0 || (i < fused.len() - 1 && fused[i - 1].height() >= fused[i + 1].height()) {
                // merge with next
                fused[i] = fused[i].merged(&fused[i + 1]);
                fused.remove(i + 1);
            } else {
                // merge with previous
                fused[i - 1] = fused[i - 1].merged(&fused[i]);
                fused.remove(i);
            }
        } else {
            i += 1;
        }
    }

    arena.alloc_slice_copy(&fused)
}

#[cfg(test)]
mod tests {
    use crate::terrain::PackedTerrVec;
    use crate::terrain::column::{TerrSlice, downsample_columns};
    use bumpalo::Bump;

    const STONE: PackedTerrVec = [255, 0, 0, 0, 0, 0, 0, 0];
    const DIRT: PackedTerrVec = [0, 255, 0, 0, 0, 0, 0, 0];
    const GRASS: PackedTerrVec = [0, 0, 255, 0, 0, 0, 0, 0];
    const AIR: PackedTerrVec = [0, 0, 0, 0, 255, 0, 0, 0];

    const fn ts(low: u16, high: u16, value: PackedTerrVec) -> TerrSlice {
        TerrSlice { low, high, value }
    }

    #[test]
    fn test_downsample() {
        let c1 = const { &[ts(0, 10, STONE), ts(10, 20, DIRT), ts(20, 30, GRASS), ts(30, u16::MAX, AIR)] };
        let c2 = const { &[ts(0, 5, STONE), ts(5, 15, DIRT), ts(15, 25, GRASS), ts(25, u16::MAX, AIR)] };
        let c3 = const { &[ts(0, 8, STONE), ts(8, 18, DIRT), ts(18, 28, GRASS), ts(28, u16::MAX, AIR)] };
        let c4 = const { &[ts(0, 12, STONE), ts(12, 22, DIRT), ts(22, 32, GRASS), ts(32, u16::MAX, AIR)] };
        //let expected_avg =
        //    TerrColumn::new(const { &[ts(0, 5, STONE), ts(5, 10, DIRT), ts(10, 15, DIRT), ts(15, 20, GRASS), ts(20, 25, GRASS), ts(25, 30, AIR), ts(30, u16::MAX, AIR)] });

        let arena = Bump::new();
        let avg = downsample_columns(&arena, &[c1, c2, c3, c4], 4);
        dbg!(avg.len());
        dbg!(avg);
        //assert_eq!(avg, expected_avg);
    }
}

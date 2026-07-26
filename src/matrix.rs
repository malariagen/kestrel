use crate::buffer::AlignedBuffer;

// This is column-major
pub type Block<T, const L: usize, const R: usize> = [[T; L]; R];

pub struct BlockArray<T: Copy> {
    buf: AlignedBuffer<T>,
    num_rows: usize,
    row_len: usize,
    lanes: usize
}

impl<T: Copy> BlockArray<T> {

    pub fn new(num_rows: usize, row_len: usize, lanes: usize) -> Self {
        let buf = AlignedBuffer::new(num_rows * row_len, lanes);
        BlockArray { buf, num_rows, row_len, lanes }
    }

    pub fn as_blocks<const L: usize, const R: usize>(&self) -> (&[Block<T, L, R>], &[[T; R]]) {
        let block_len = L * R;
        let num_blocks = self.buf.len() / block_len;
        let remainder = self.buf.len() % block_len;

        let remainder_rows = remainder / R;

        let block_ptr = self.buf.as_ptr() as *const Block<T, L, R>;
        let remainder_ptr = unsafe { self.buf.as_ptr().add(num_blocks.unchecked_mul(block_len)) } as *const [T; R];

        let block_slice = unsafe { std::slice::from_raw_parts(block_ptr, num_blocks) };
        let remainder_slice = unsafe { std::slice::from_raw_parts(remainder_ptr, remainder_rows) };

        (block_slice, remainder_slice)
    }

    pub fn as_blocks_mut<const L: usize, const R: usize>(&mut self) -> (&mut [Block<T, L, R>], &mut [[T; R]]) {
        let block_len = L * R;
        let num_blocks = self.buf.len() / block_len;
        let remainder = self.buf.len() % block_len;

        let remainder_rows = remainder / R;

        let block_ptr = self.buf.as_mut_ptr() as *mut Block<T, L, R>;
        let remainder_ptr = unsafe { self.buf.as_mut_ptr().add(num_blocks.unchecked_mul(block_len)) } as *mut [T; R];

        let block_slice = unsafe { std::slice::from_raw_parts_mut(block_ptr, num_blocks) };
        let remainder_slice = unsafe { std::slice::from_raw_parts_mut(remainder_ptr, remainder_rows) };

        (block_slice, remainder_slice)
    }

    pub fn from_rows<const L: usize, const R: usize>(
        mut iter: impl Iterator<Item = [T; R]>,
        num_rows: usize,
    ) -> Self {
        let mut array = BlockArray::new(num_rows, R, L);

        let (blocks, remainder) = array.as_blocks_mut::<L, R>();

        for block in blocks.iter_mut() {
            for i in 0..L {
                let row = iter.next().unwrap();
                for c in 0..R {
                    block[c][i] = row[c];
                }
            }
        }

        for row in remainder.iter_mut() {
            *row = iter.next().unwrap();
        }

        array
    }

}

pub fn test(n: usize) -> BlockArray<f64> {
    let x = vec![[0.0; 9]; n];
    let array = BlockArray::from_rows::<8, 9>(x.iter().copied(), x.len());
    array
}
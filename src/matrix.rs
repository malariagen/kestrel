use crate::buffer::AlignedBuffer;

// This is column-major
pub type Block<T, const L: usize, const R: usize> = [[T; L]; R];

pub struct BlockBuffer<T: Copy, const L: usize, const R: usize> {
    buf: AlignedBuffer<T>,
    num_rows: usize,
}

impl<T: Copy, const L: usize, const R: usize> BlockBuffer<T, L, R> {
    pub fn new(num_rows: usize) -> Self {
        let buf = AlignedBuffer::new(num_rows * R, L);
        BlockBuffer {
            buf,
            num_rows,
        }
    }

    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    pub fn as_blocks(&self) -> (&[Block<T, L, R>], &[[T; R]]) {
        let block_len = L * R;
        let num_blocks = self.buf.len() / block_len;
        let remainder = self.buf.len() % block_len;

        let remainder_rows = remainder / R;

        let block_ptr = self.buf.as_ptr() as *const Block<T, L, R>;
        let remainder_ptr =
            unsafe { self.buf.as_ptr().add(num_blocks.unchecked_mul(block_len)) } as *const [T; R];

        let block_slice = unsafe { std::slice::from_raw_parts(block_ptr, num_blocks) };
        let remainder_slice = unsafe { std::slice::from_raw_parts(remainder_ptr, remainder_rows) };

        (block_slice, remainder_slice)
    }

    pub fn as_blocks_mut(
        &mut self,
    ) -> (&mut [Block<T, L, R>], &mut [[T; R]]) {
        let block_len = L * R;
        let num_blocks = self.buf.len() / block_len;
        let remainder = self.buf.len() % block_len;

        let remainder_rows = remainder / R;

        let block_ptr = self.buf.as_mut_ptr() as *mut Block<T, L, R>;
        let remainder_ptr = unsafe {
            self.buf
                .as_mut_ptr()
                .add(num_blocks.unchecked_mul(block_len))
        } as *mut [T; R];

        let block_slice = unsafe { std::slice::from_raw_parts_mut(block_ptr, num_blocks) };
        let remainder_slice =
            unsafe { std::slice::from_raw_parts_mut(remainder_ptr, remainder_rows) };

        (block_slice, remainder_slice)
    }

    pub fn fill_from_rows(
        &mut self,
        mut iter: impl Iterator<Item = [T; R]>,
    ) {
        let (blocks, remainder) = self.as_blocks_mut();

        for block in blocks.iter_mut() {
            for i in 0..L {
                let row = unsafe { iter.next().unwrap_unchecked() };
                for c in 0..R {
                    block[c][i] = row[c];
                }
            }
        }

        for row in remainder.iter_mut() {
            *row = unsafe { iter.next().unwrap_unchecked() };
        }
    }
}

use std::alloc::Layout;


struct AlignedBuffer {
    ptr: *mut f64,
    len: usize,
    layout: Layout,
}

impl AlignedBuffer {
    pub fn new(len: usize, align: usize) -> Self {

        assert!(len > 0, "Length must be nonzero");

        let size = len.checked_mul(std::mem::size_of::<f64>()).expect("Size overflow");
        let align = align.checked_mul(std::mem::size_of::<f64>()).expect("Align overflow");

        let layout = Layout::from_size_align(size, align).expect("Invalid layout");

        let ptr = unsafe { std::alloc::alloc_zeroed(layout) as *mut f64 };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }

        AlignedBuffer { ptr, len, layout }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn as_slice(&self) -> &[f64] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        unsafe { std::alloc::dealloc(self.ptr as *mut u8, self.layout) }
    }
}
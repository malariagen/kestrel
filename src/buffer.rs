use std::alloc::Layout;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

// See https://doc.rust-lang.org/nomicon/vec/vec.html
pub struct AlignedBuffer<T: Copy> {
    ptr: NonNull<T>,
    len: usize,
    layout: Layout,
}

impl<T: Copy> AlignedBuffer<T> {
    pub fn new(len: usize, multiple: usize) -> Self {
        assert!(std::mem::size_of::<T>() != 0, "Cannot use zero-sized type");
        assert!(len > 0, "Length must be nonzero");

        let size = len
            .checked_mul(std::mem::size_of::<T>())
            .expect("Size overflow");
        let align = multiple
            .checked_mul(std::mem::size_of::<T>())
            .expect("Align overflow");

        let layout = Layout::from_size_align(size, align).expect("Invalid layout");

        let ptr = unsafe { std::alloc::alloc_zeroed(layout) as *mut T };
        let ptr = match NonNull::new(ptr) {
            Some(p) => p,
            None => std::alloc::handle_alloc_error(layout),
        };

        AlignedBuffer { ptr, len, layout }
    }

    pub fn len(&self) -> usize {
        self.len
    }
}

impl<T: Copy> Drop for AlignedBuffer<T> {
    fn drop(&mut self) {
        unsafe { std::alloc::dealloc(self.ptr.as_ptr() as *mut u8, self.layout) }
    }
}

impl<T: Copy> Deref for AlignedBuffer<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl<T: Copy> DerefMut for AlignedBuffer<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

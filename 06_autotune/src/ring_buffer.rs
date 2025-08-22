#![allow(dead_code)]
use core::{
    cell::UnsafeCell,
    sync::atomic::{AtomicU32, Ordering},
};

/// **SPSC** lock-free ring buffer.
/// `N` **must** be a power of two.
pub struct RingBuffer<const N: usize> {
    buf: UnsafeCell<[f32; N]>,
    write: AtomicU32,
    read: AtomicU32,
}

// Safety – single producer / single consumer.
unsafe impl<const N: usize> Sync for RingBuffer<N> {}

impl<const N: usize> RingBuffer<N> {
    pub fn with_offset(offset: u32) -> Self {
        Self {
            buf: UnsafeCell::new([0.0; N]),
            write: AtomicU32::new(offset),
            read: AtomicU32::new(0),
        }
    }

    pub const fn new() -> Self {
        Self {
            buf: UnsafeCell::new([0.0; N]),
            write: AtomicU32::new(0),
            read: AtomicU32::new(0),
        }
    }

    #[inline(always)]
    pub fn push(&self, v: f32) {
        let w = self.write.load(Ordering::Relaxed);
        unsafe { (*self.buf.get())[w as usize & (N - 1)] = v };
        self.write.store(w.wrapping_add(1), Ordering::Release);
    }

    #[inline(always)]
    pub fn pop(&self) -> f32 {
        let r = self.read.load(Ordering::Relaxed);
        let v = unsafe {
            let cell = &mut (*self.buf.get())[r as usize & (N - 1)];
            let old_val = *cell;
            *cell = 0.0; // Clear after reading
            old_val
        };
        self.read.store(r.wrapping_add(1), Ordering::Release);
        v
    }

    #[inline(always)]
    pub fn write_index(&self) -> u32 {
        self.write.load(core::sync::atomic::Ordering::Relaxed)
    }

    #[inline(always)]
    pub fn advance_write(&self, n: u32) {
        use core::sync::atomic::Ordering;
        self.write.fetch_add(n, Ordering::Relaxed);
    }

    pub fn add_at_offset(&self, offset: u32, val: f32) {
        let idx = self.write.load(Ordering::Relaxed).wrapping_add(offset);
        unsafe {
            let cell = &mut (*self.buf.get())[idx as usize & (N - 1)];
            *cell += val;
        }
    }

    /// Copy the last `LEN` samples (oldest first) into `dest`.
    pub fn latest_block<const LEN: usize>(&self, dest: &mut [f32; LEN]) {
        cortex_m::interrupt::free(|_| {
            let w = self.write.load(Ordering::Acquire);
            for i in 0..LEN {
                let idx = w.wrapping_sub(LEN as u32).wrapping_add(i as u32);
                dest[i] = unsafe { (*self.buf.get())[idx as usize & (N - 1)] };
            }
        });
    }

    pub fn block_from<const LEN: usize>(&self, write_idx: u32, dst: &mut [f32; LEN]) {
        for i in 0..LEN {
            let idx = write_idx.wrapping_sub(LEN as u32).wrapping_add(i as u32);
            dst[i] = unsafe { (*self.buf.get())[idx as usize & (N - 1)] };
        }
    }
}

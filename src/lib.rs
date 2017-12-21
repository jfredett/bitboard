#![feature(unique, alloc, heap_api, allocator_api)]

#![recursion_limit="256"]
extern crate typenum;

use std::mem;
//use std::ops::{Deref, DerefMut};
use std::marker::PhantomData;
use std::heap::{Alloc, Layout, Heap};

use typenum::marker_traits::*;


/// A bitboard of size NxN, with alignment A
///
/// This only constructs square bitboards.
///
/// A can be set. Generally, small alignments favor memory savings, large alignments favor faster
/// operations (fewer total operations to address the whole board),.
///
/// For instance, given a 64x64 bitboard, it will take 64 operations to compare to another board
/// with u64 alignment, and 512 operations to compare to another board given u8 alignment.
///
/// The downside is that with odd numbered boards, u64 is going to result in some dead memory being
/// assigned. In practice it's never enough to matter, but it's easy enough to grant the option.
///
/// There are no aliases provided, but I suggest you create one for whatever application you have.
/// Something like:
///
/// `type Bitboard = bitboard::Bitboard<U{YourSize}, u{YourAlignment}>`
///
/// Will save a lot of typing and will also probably prevent screwups.
pub struct Bitboard<N: Unsigned, A : Sized> {
    ptr: *mut u8,
    offset: usize,
    typenum: PhantomData<N>,
    alignment: PhantomData<A>
}

#[derive(PartialEq, Eq, Debug)]
pub enum BitboardError {
    OutOfBounds(usize, usize),
    UnknownError
}

type BitboardResult<N> = Result<N, BitboardError>;

impl<N : Unsigned, A : Sized> Bitboard<N,A> {
    /// Construct a new, blank bitboard of size NxN with alignment A
    pub fn new() -> Bitboard<N, A> {
        let layout = Self::layout();
        let ptr;

        unsafe {
            match Heap.alloc(layout) {
                Ok(p) => ptr = p,
                Err(e) => panic!("Failed to allocate bitboard! {}", e)
            }
        }

        Bitboard {
            ptr: ptr,
            offset: 0,
            typenum: PhantomData,
            alignment: PhantomData
        }
    }


    /// Set the bit at location (x,y)
    ///
    /// Bitboards are 'in the forth quadrant', meaning their origin is in the top-left, starting at
    /// 0, high values of `x` move to the right, high values of `y` move downward.
    pub fn set(&mut self, x: usize, y: usize) -> BitboardResult<()> {
        if !(Self::in_bounds(x) && Self::in_bounds(y)) { return Err(BitboardError::OutOfBounds(x,y)); }

        let offset = x + y * Self::alignment_bits(); 
        let bit_pos = 1 << (offset % Self::alignment_bits());

        unsafe { *self.ptr.offset(offset as isize) |= bit_pos; }

        Ok(())
    }


    fn in_bounds(i: usize) -> bool {
        i <= N::to_usize()
    }

    /// Return the alignment, in bytes, of the Bitboard
    ///
    /// # Examples
    ///
    ///```
    ///    extern crate typenum;
    ///    extern crate bitboard;
    ///
    ///    use typenum::consts::U8;
    ///    use bitboard::Bitboard;
    ///
    ///    fn main() {
    ///        assert_eq!(Bitboard::<U8, u64>::alignment(), 8);
    ///    }
    ///```
    #[inline(always)]
    pub fn alignment() -> usize {
        mem::align_of::<A>()
    }

    /// Return the alignment, in bytes, of the Bitboard
    ///
    /// # Examples
    ///
    ///```
    ///    extern crate typenum;
    ///    extern crate bitboard;
    ///
    ///    use typenum::consts::U8;
    ///    use bitboard::Bitboard;
    ///
    ///
    ///    fn main() {
    ///        assert_eq!(Bitboard::<U8, u64>::alignment_bits(), 64);
    ///    }
    ///```
    #[inline(always)]
    pub fn alignment_bits() -> usize {
        Self::alignment() * 8
    }


    /// Bits used by the bitboard
    #[inline(always)]
    fn bits_used() -> usize {
        N::to_usize().pow(2)
    }

    /// Bits we need to allocate to satisfy the bitboard
    #[inline(always)]
    fn bits_needed() -> usize {
        let bit_remainder = Self::bits_used() % Self::alignment_bits();
        if bit_remainder == 0 {
            Self::bits_used()
        } else {
            Self::bits_used() + Self::alignment_bits() - bit_remainder
        }
    }

    /// Return the size, in bytes, needed to allocate the bitboard.
    ///
    /// # Examples
    ///
    ///```
    ///    extern crate typenum;
    ///    extern crate bitboard;
    ///
    ///    use typenum::consts::{U8,U19};
    ///    use bitboard::Bitboard;
    ///
    ///    fn main() {
    ///        // chess boards
    ///        assert_eq!(Bitboard::<U8, u64>::size(), 1);
    ///        assert_eq!(Bitboard::<U8, u8>::size(), 8);
    ///        // go boards
    ///        assert_eq!(Bitboard::<U19, u8>::size(), 46);
    ///        assert_eq!(Bitboard::<U19, u64>::size(), 6);
    ///    }
    ///```
    #[inline(always)]
    pub fn size() -> usize {
        Self::bits_needed() / Self::alignment_bits()
    }

    /// Calculate the memory layout for the bitboard.
    fn layout() -> Layout {
        Layout::from_size_align(Self::size(), Self::alignment()).unwrap()
    }
}

impl<N : Unsigned, A : Sized> Drop for Bitboard<N,A> {
    fn drop(&mut self) {
        let layout = Self::layout();
        unsafe { Heap.dealloc(self.ptr, layout); }
    }
}

impl<N : Unsigned, A : Sized> Iterator for Bitboard<N,A> {
    type Item = u8;

    /// Iterates over bytes of the bitboard, not individual bits.
    /// making implementing comparisons across the whole bitboard easy.
    fn next(&mut self) -> Option<u8> {
        if self.offset == Self::size() {
            None
        } else {
            unsafe { 
                let ret = *self.ptr.offset(self.offset as isize);
                self.offset += 1;
                Some(ret)
            }
        }
    }
}


#[cfg(test)]
mod test {
    use super::*;

    use typenum::consts::*;

    fn tic_tac_toe_board<A : Sized>() -> Bitboard<U3, A> { Bitboard::new() }
    fn chess_board<A : Sized>() -> Bitboard<U8, A> { Bitboard::new() }
    fn go_board<A : Sized>() -> Bitboard<U19, A> { Bitboard::new() }

    mod type_allocation {
        use super::*;
        #[test]
        fn creates_correctly_sized_board() {
            assert_eq!(Bitboard::<U8,u8>::size(), 8); // allocated 8 bytes, 64 bins total
            assert_eq!(Bitboard::<U19,u8>::size(), 46); // should be 46 bytes, to support 361 (19**2) bits, aligned to 1b, we need 368 total bits, or 46 bytes
        }

        #[test]
        fn statically_assigned_alignment() {
            // NB: The alignment paramater *could* be autocalculated. If memory is a concern, you
            // might choose a smaller alignment, if speed is all you care about, a larger alignment
            // makes sense. Ultimately it should be arbitrary and things should work regardless of
            // specific alignments.
            assert_eq!(Bitboard::<U8,u8>::alignment(), 1); // aligned to the byte
            assert_eq!(Bitboard::<U19,u8>::alignment(), 1);     // ibid
        }
    }

    mod size {
        use super::*;

        #[test]
        fn size_alignment() {
            assert_eq!(Bitboard::<U8, u64>::size(), 1);
            assert_eq!(Bitboard::<U8, u8>::size(), 8);
            // go boards
            assert_eq!(Bitboard::<U19, u8>::size(), 46);
            assert_eq!(Bitboard::<U19, u64>::size(), 6);
        }
    }

    mod iter {
        use super::*;

        #[test]
        fn inits_to_zero_small_alignment() {
            let bb = chess_board::<u8>();
            for byte in bb {
                assert_eq!(byte, 0);
            }
        }

        #[test]
        fn inits_to_zero_large_alignment() {
            let bb = chess_board::<u64>();
            for byte in bb {
                assert_eq!(byte, 0);
            }
        }
    }
}


        //#[test]
        //fn index_in_bounds() {
            //// TODO: Set a bit, then check the index to make sure it's set.
            //let tt = tic_tac_toe_board();
            ////assert_eq!(*tt[(0,0)], 0);
        //}

        //#[test]
        //#[should_panic]
        //fn index_out_of_bounds() {
            //let tt = tic_tac_toe_board();
            ////assert_eq!(*tt[(3,0)], 0);
        //}

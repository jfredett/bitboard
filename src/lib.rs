#![feature(unique, alloc, heap_api, allocator_api)]

//! Bitboard, compile-time-sized, typesafe, low level bitboards for Rust.
//!
//!

#![recursion_limit="256"]
extern crate typenum;

use std::mem;
use std::fmt;
use std::marker::PhantomData;
use std::heap::{Alloc, Layout, Heap};

use typenum::marker_traits::*;


/// A square bitboard of size `NxN`, with alignment `A`
///
/// XXX: Pretty sure this might not be true -- we have byte-sized pointers... Heap::alloc only
/// gives back bytes... hmm
/// {
/// `A` can be set. Generally, small alignments favor memory savings, large alignments favor faster
/// operations (fewer total operations to address the whole board),.
///
/// For instance, given a 64x64 bitboard, it will take 64 operations to compare to another board
/// with u64 alignment, and 512 operations to compare to another board given u8 alignment.
///
/// The downside is that with odd numbered boards, u64 is going to result in some dead memory being
/// assigned. In practice it's never enough to matter, but it's easy enough to grant the option.
/// }
///
/// There are no aliases provided, but I suggest you create one for whatever application you have.
/// Something like:
///
/// ```
/// extern crate typenum;
/// extern crate bitboard;
///
/// use typenum::consts::U8;
/// use bitboard::Bitboard;
///
/// type Chessboard = bitboard::Bitboard<U8, u64>;
///
/// fn main() {
///     let cc : Chessboard = Bitboard::new();
///     // ...
/// }
/// ```
///
/// Will save a lot of typing and will also probably prevent screwups.
///
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
    /// Construct a new, blank bitboard of size `NxN`, with alignment `A`
    ///
    /// # Examples
    /// ```
    ///    extern crate typenum;
    ///    extern crate bitboard;
    ///
    ///    use typenum::consts::U8;
    ///    use bitboard::Bitboard;
    ///
    ///    fn main() {
    ///        let bb = Bitboard::<U8,u64>::new();
    ///        // ...
    ///    }
    /// ```
    pub fn new() -> Bitboard<N, A> {
        let layout = Self::layout();
        let ptr;

        unsafe {
            match Heap.alloc_zeroed(layout) {
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
    /// Origin is in the top-left, starting at 0, high values of `x` move to the right, high values
    /// of `y` move downward.
    ///
    /// # Examples
    /// ```
    /// extern crate typenum;
    /// extern crate bitboard;
    ///
    /// use typenum::consts::U3;
    /// use bitboard::Bitboard;
    ///
    /// type TicTacToe = bitboard::Bitboard<U3, u16>;
    /// fn main() {
    ///   let mut x_positions : TicTacToe = Bitboard::new();
    ///   // X has taken no positions
    ///   // 000
    ///   // 000
    ///   // 000
    ///   assert!(!x_positions.is_set(1,1).ok().unwrap()); // the center square is free!
    ///
    ///   x_positions.set(1,1); // take the center square
    ///   // x_positions now looks like:
    ///   // 000
    ///   // 010
    ///   // 000
    ///   assert!(x_positions.is_set(1,1).ok().unwrap());
    ///   // ... snip
    /// }
    /// ```
    pub fn set(&mut self, x: usize, y: usize) -> BitboardResult<()> {
        if Self::is_out_of_bounds(x,y) { return Err(BitboardError::OutOfBounds(x,y)); }

        let (offset, bit_pos) = Self::coords_to_offset_and_pos(x,y);

        unsafe { *self.ptr.offset(offset) |= bit_pos; }

        Ok(())
    }

    /// Returns true if the bit at the given coords is set, false otherwise. May error out if given
    /// out-of-bounds coordinates. See #set for examples
    pub fn is_set(&self, x: usize, y: usize) -> BitboardResult<bool> {
        if Self::is_out_of_bounds(x,y) { return Err(BitboardError::OutOfBounds(x,y)); }

        let (offset, bit_pos) = Self::coords_to_offset_and_pos(x,y);

        Ok(unsafe { (*self.ptr.offset(offset) & bit_pos) != 0 })
    }

    /// Flip the value of the bit at (x,y)
    ///
    /// # Examples
    /// ```
    /// extern crate typenum;
    /// extern crate bitboard;
    ///
    /// use typenum::consts::U3;
    /// use bitboard::Bitboard;
    ///
    /// type TicTacToe = bitboard::Bitboard<U3, u16>;
    /// fn main() {
    ///   let mut x_positions : TicTacToe = Bitboard::new();
    ///   // X has taken no positions
    ///   // 000
    ///   // 000
    ///   // 000
    ///   assert!(!x_positions.is_set(1,1).ok().unwrap()); // the center square is free!
    ///
    ///   x_positions.flip(1,1); // take the center square
    ///   // x_positions now looks like:
    ///   // 000
    ///   // 010
    ///   // 000
    ///   assert!(x_positions.is_set(1,1).ok().unwrap());
    ///
    ///   x_positions.flip(1,1); // undo that
    ///
    ///   assert!(!x_positions.is_set(1,1).ok().unwrap());
    /// }
    /// ```
    pub fn  flip(&mut self, x: usize, y: usize) -> BitboardResult<()> {
        if Self::is_out_of_bounds(x,y) { return Err(BitboardError::OutOfBounds(x,y)); }

        let (offset, bit_pos) = Self::coords_to_offset_and_pos(x,y);

        unsafe { *self.ptr.offset(offset) ^= bit_pos; }

        Ok(())
    }

    /// Unset the bit at location (x,y)
    pub fn unset(&mut self, x: usize, y: usize) -> BitboardResult<()> {
        if Self::is_out_of_bounds(x,y) { return Err(BitboardError::OutOfBounds(x,y)); }

        let (offset, bit_pos) = Self::coords_to_offset_and_pos(x,y);

        unsafe { *self.ptr.offset(offset) &= !bit_pos; }

        Ok(())
    }

    /// Returns true if the bit at the given coords is set, false otherwise. May error out if given
    /// out-of-bounds coordinates. See #set for examples
    pub fn is_unset(&self, x: usize, y: usize) -> BitboardResult<bool> {
        if Self::is_out_of_bounds(x,y) { return Err(BitboardError::OutOfBounds(x,y)); }

        let (offset, bit_pos) = Self::coords_to_offset_and_pos(x,y);

        Ok(unsafe { (*self.ptr.offset(offset) & bit_pos) == 0 })
    }


    fn coords_to_offset_and_pos(x: usize, y: usize) -> (isize, u8) {
        let offset = x + y * Self::alignment_bits(); 
        let bit_pos = 1 << (offset % Self::alignment_bits());
        (offset as isize, bit_pos)
    }

    fn is_out_of_bounds(x: usize, y: usize) -> bool {
        !(Self::in_bounds(x) && Self::in_bounds(y))
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
        unsafe { Heap.dealloc(self.ptr as *mut _, layout); }
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

impl<N : Unsigned, A : Sized> fmt::Display for Bitboard<N,A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for i in 0..(N::to_usize()) {
            for j in 0..(N::to_usize()) {
                if self.is_set(i,j).ok().unwrap() {
                    let _ = write!(f, "{}", 1);
                } else {
                    let _ = write!(f, "{}", 0);
                }
            }
            let _ = writeln!(f);
        }
        write!(f, "")
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

    mod size_alignment_and_layout {
        use super::*;

        #[test]
        fn alignment_bits() {
            assert_eq!(Bitboard::<U8, u64>::alignment(), 8);
            assert_eq!(Bitboard::<U8, u64>::alignment_bits(), 64);
            assert_eq!(Bitboard::<U8, u8>::alignment(), 1);
            assert_eq!(Bitboard::<U8, u8>::alignment_bits(), 8);
        }

        #[test]
        fn total_bytes_needed() {
            assert_eq!(Bitboard::<U8, u64>::size(), 1);
            assert_eq!(Bitboard::<U8, u8>::size(), 8);
            // go boards
            assert_eq!(Bitboard::<U19, u8>::size(), 46);
            assert_eq!(Bitboard::<U19, u64>::size(), 6);
        }
    }

    mod set_and_is_set {
        use super::*;

        #[test]
        #[allow(unused_must_use)]
        fn set() {
            let mut tt = tic_tac_toe_board::<u16>();

            tt.set(0,0); tt.set(1,1); tt.set(2,2);

            assert!(tt.is_set(0,0).ok().unwrap());
            assert!(tt.is_unset(0,1).ok().unwrap());
        }

        #[test]
        fn set_oob() {
            let mut go = go_board::<u64>();

            let res = go.set(200,100);

            assert_eq!(res, Err(BitboardError::OutOfBounds(200,100)))
        }

    }

    mod unset_and_is_unset {
        use super::*;

        #[test]
        #[allow(unused_must_use)]
        fn unset() {
            let mut tt = tic_tac_toe_board::<u16>();

            tt.set(0,0);

            assert!(tt.is_set(0,0).ok().unwrap());

            tt.unset(0,0);

            assert!(tt.is_unset(0,0).ok().unwrap());
        }

        #[test]
        fn unset_oob() {
            let mut go = go_board::<u64>();

            let res = go.unset(200,100);

            assert_eq!(res, Err(BitboardError::OutOfBounds(200,100)))
        }
    }

    mod flip {
        use super::*;

        #[test]
        #[allow(unused_must_use)]
        fn flip() {
            let mut tt = tic_tac_toe_board::<u16>();

            assert!(tt.is_unset(0,0).ok().unwrap());

            tt.flip(0,0);

            assert!(tt.is_set(0,0).ok().unwrap());

            tt.flip(0,0);

            assert!(tt.is_unset(0,0).ok().unwrap());
        }

        #[test]
        fn flip_oob() {
            let mut go = go_board::<u64>();

            let res = go.flip(200,100);

            assert_eq!(res, Err(BitboardError::OutOfBounds(200,100)))
        }
    }

    mod display {
        use super::*;

        use std::io::{Write};

        #[test]
        #[allow(unused_must_use)]
        fn formats_tic_tac_toe_bitboard() {
            let mut c = tic_tac_toe_board::<u16>();

            for i in 0..3 { c.set(i,i); }

            println!("{}", c);

            let mut b = vec![];
            let expected = vec![
                '1','0','0','\n',
                '0','1','0','\n',
                '0','0','1','\n'];

            write!(&mut b, "{}", c);

            for i in 0..12 { // 12 = 9 squares + 3 newlines
                assert_eq!(b[i], expected[i] as u8);
            }
        }

        #[test]
        #[allow(unused_must_use)]
        fn formats_chess_bitboard() {
            let mut c = chess_board::<u64>();

            for i in 0..8 { c.set(i,i); }

            println!("{}", c);

            let mut b = vec![];
            let expected = vec![
                '1','0','0','0','0','0','0','0','\n',
                '0','1','0','0','0','0','0','0','\n',
                '0','0','1','0','0','0','0','0','\n',
                '0','0','0','1','0','0','0','0','\n',
                '0','0','0','0','1','0','0','0','\n',
                '0','0','0','0','0','1','0','0','\n',
                '0','0','0','0','0','0','1','0','\n',
                '0','0','0','0','0','0','0','1','\n'];

            write!(&mut b, "{}", c);

            for i in 0..72 { // 72 == 64 squares + 8 newlines
                assert_eq!(b[i], expected[i] as u8);
            }
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

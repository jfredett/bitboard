#![feature(unique, alloc, heap_api, allocator_api, test)]

//! Bitboard, compile-time-sized, typesafe, low level bitboards for Rust.
//!
//!

#![recursion_limit="256"]
extern crate typenum;

#[cfg(test)]
extern crate test;

use std::mem;
use std::ops;
use std::cmp;
use std::fmt;
use std::marker::PhantomData;
use std::heap::{Alloc, Layout, Heap};

use typenum::marker_traits::*;


/// A square bitboard of size `NxN`, with alignment `A`
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
/// type Chessboard = bitboard::Bitboard<U8>;
///
/// fn main() {
///     let cc : Chessboard = Bitboard::new();
///     // ...
/// }
/// ```
///
/// Will save a lot of typing and will also probably prevent screwups.
///
pub struct Bitboard<N: Unsigned> {
    ptr: *mut u8,
    typenum: PhantomData<N>,
}

#[derive(PartialEq, Eq, Debug)]
pub enum BitboardError {
    OutOfBounds(usize, usize),
    UnknownError
}

type BitboardResult<N> = Result<N, BitboardError>;

impl<N : Unsigned> Bitboard<N> {
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
    ///        let bb = Bitboard::<U8>::new();
    ///        // ...
    ///    }
    /// ```
    pub fn new() -> Bitboard<N> {
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
            typenum: PhantomData
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
    /// type TicTacToe = bitboard::Bitboard<U3>;
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
    /// type TicTacToe = bitboard::Bitboard<U3>;
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
    pub fn flip(&mut self, x: usize, y: usize) -> BitboardResult<()> {
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
        let pos = x + y * Self::alignment_bits();
        let offset = pos / Self::alignment_bits();
        let bit_pos = 1 << (pos % Self::alignment_bits());
        (offset as isize, bit_pos)
    }

    #[inline(always)]
    fn is_out_of_bounds(x: usize, y: usize) -> bool {
        !(Self::in_bounds(x) && Self::in_bounds(y))
    }

    #[inline(always)]
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
    ///        assert_eq!(Bitboard::<U8>::alignment(), 1);
    ///    }
    ///```
    #[inline(always)]
    pub fn alignment() -> usize {
        mem::align_of::<u8>()
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
    ///        assert_eq!(Bitboard::<U8>::alignment_bits(), 8);
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
    ///        assert_eq!(Bitboard::<U8>::size(), 8);
    ///        // go boards
    ///        assert_eq!(Bitboard::<U19>::size(), 46);
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

impl<N : Unsigned> Drop for Bitboard<N> {
    fn drop(&mut self) {
        let layout = Self::layout();
        unsafe { Heap.dealloc(self.ptr as *mut _, layout); }
    }
}

impl<N : Unsigned> fmt::Debug for Bitboard<N> {
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

impl<N : Unsigned> fmt::Display for Bitboard<N> {
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

impl<N : Unsigned> cmp::PartialEq for Bitboard<N> {
    fn eq(&self, other: &Bitboard<N>) -> bool {
        // we know the sizes are the same because `N` is the same, and `A` is the same
        for amt in 0..Self::size() {
            unsafe {
                if *self.ptr.offset(amt as isize) != *other.ptr.offset(amt as isize) {
                    return false;
                }
            }
        }
        return true;
    }
}

impl<N : Unsigned> cmp::Eq for Bitboard<N> { }

impl<N: Unsigned> Clone for Bitboard<N> {
    fn clone(&self) -> Bitboard<N> {
        let new_bb : Bitboard<N> = Bitboard::new();
        for amt in 0..Self::size() {
            unsafe { *new_bb.ptr.offset(amt as isize) = *self.ptr.offset(amt as isize) }
        }
        return new_bb;
    }
}

impl<N : Unsigned> ops::BitAnd for Bitboard<N> {
    type Output = Bitboard<N>;

    fn bitand(self, other: Bitboard<N>) -> Bitboard<N> {
        let new_bb : Bitboard<N> = Bitboard::new();
        // we know the sizes are the same because `N` is the same, and `A` is the same
        for amt in 0..(Self::size() as isize) {
            unsafe {
                *new_bb.ptr.offset(amt) = (*self.ptr.offset(amt)) & (*other.ptr.offset(amt))
            }
        }
        return new_bb
    }
}

impl<N : Unsigned> ops::BitOr for Bitboard<N> {
    type Output = Bitboard<N>;

    fn bitor(self, other: Bitboard<N>) -> Bitboard<N> {
        let new_bb : Bitboard<N> = Bitboard::new();
        // we know the sizes are the same because `N` is the same, and `A` is the same
        for amt in 0..(Self::size() as isize) {
            unsafe {
                *new_bb.ptr.offset(amt) = (*self.ptr.offset(amt)) | (*other.ptr.offset(amt))
            }
        }
        return new_bb
    }
}

impl<N : Unsigned> ops::BitXor for Bitboard<N> {
    type Output = Bitboard<N>;

    fn bitxor(self, other: Bitboard<N>) -> Bitboard<N> {
        let new_bb : Bitboard<N> = Bitboard::new();
        // we know the sizes are the same because `N` is the same, and `A` is the same
        for amt in 0..(Self::size() as isize) {
            unsafe {
                *new_bb.ptr.offset(amt) = (*self.ptr.offset(amt)) ^ (*other.ptr.offset(amt))
            }
        }
        return new_bb
    }
}

#[cfg(test)]
#[allow(unused_must_use)]
mod tests {
    use super::*;

    use typenum::consts::*;

    fn tic_tac_toe_board() -> Bitboard<U3> { Bitboard::new() }
    fn chess_board() -> Bitboard<U8> { Bitboard::new() }
    fn go_board() -> Bitboard<U19> { Bitboard::new() }

    mod eq {
        use super::*;

        #[test]
        fn eq() {
            let bb1 = go_board();
            let bb2 = go_board();

            // these are separate objects
            assert_ne!(bb1.ptr, bb2.ptr);
            // equality is by value
            assert_eq!(bb1, bb2);
        }

    }

    mod bitxor {
        use super::*;

        #[test]
        fn is_union() {
            // 100
            // 010
            // 001
            let mut bb1 = tic_tac_toe_board();

            // 001
            // 010
            // 100
            let mut bb2 = tic_tac_toe_board();

            // 101
            // 000
            // 101
            let mut expected = tic_tac_toe_board();

            bb1.set(0,0);
            bb1.set(1,1);
            bb1.set(2,2);

            bb2.set(0,2);
            bb2.set(1,1);
            bb2.set(2,0);

            expected.set(0,0);
            expected.set(2,2);
            expected.set(0,2);
            expected.set(2,0);

            assert_eq!(bb1 ^ bb2, expected);
        }

        #[test]
        fn self_inverse() {
            let mut bb1 = go_board();
            let expected = Bitboard::new();

            assert_eq!(bb1.clone() ^ bb1.clone(), expected);

            bb1.set(1,2);

            assert_eq!(bb1.clone() ^ bb1.clone(), expected);

            bb1.set(10,2);

            assert_eq!(bb1.clone() ^ bb1.clone(), expected);
        }
    }

    mod bitor {
        use super::*;

        #[test]
        fn is_union() {
            // 100
            // 010
            // 001
            let mut bb1 = tic_tac_toe_board();

            // 001
            // 010
            // 100
            let mut bb2 = tic_tac_toe_board();

            // 101
            // 010
            // 101
            let mut expected = tic_tac_toe_board();

            bb1.set(0,0);
            bb1.set(1,1);
            bb1.set(2,2);

            bb2.set(0,2);
            bb2.set(1,1);
            bb2.set(2,0);

            expected.set(0,0);
            expected.set(1,1);
            expected.set(2,2);
            expected.set(0,2);
            expected.set(2,0);

            assert_eq!(bb1 | bb2, expected);
        }

        #[test]
        fn has_identity() {
            let mut bb1 = go_board();

            assert_eq!(bb1.clone() | bb1.clone(), bb1.clone());

            bb1.set(1,2);

            assert_eq!(bb1.clone() | bb1.clone(), bb1.clone());

            bb1.set(10,2);

            assert_eq!(bb1.clone() | bb1.clone(), bb1.clone());
        }
    }

    mod bitand {
        use super::*;

        #[test]
        fn is_intersection() {
            // 100
            // 010
            // 001
            let mut bb1 = tic_tac_toe_board();

            // 001
            // 010
            // 100
            let mut bb2 = tic_tac_toe_board();

            // 000
            // 010
            // 000
            let mut expected = tic_tac_toe_board();

            bb1.set(0,0);
            bb1.set(1,1);
            bb1.set(2,2);

            bb2.set(0,2);
            bb2.set(1,1);
            bb2.set(2,0);

            expected.set(1,1);

            assert_eq!(bb1 & bb2, expected);
        }

        #[test]
        fn has_identity() {
            let mut bb1 = go_board();

            assert_eq!(bb1.clone() & bb1.clone(), bb1.clone());

            bb1.set(1,2);

            assert_eq!(bb1.clone() & bb1.clone(), bb1.clone());

            bb1.set(10,2);

            assert_eq!(bb1.clone() & bb1.clone(), bb1.clone());
        }

    }

    mod type_allocation {
        use super::*;
        #[test]
        fn creates_correctly_sized_board() {
            assert_eq!(Bitboard::<U8>::size(), 8); // allocated 8 bytes, 64 bins total
            assert_eq!(Bitboard::<U19>::size(), 46); // should be 46 bytes, to support 361 (19**2) bits, aligned to 1b, we need 368 total bits, or 46 bytes
        }

        #[test]
        fn statically_assigned_alignment() {
            assert_eq!(Bitboard::<U8>::alignment(), 1); // aligned to the byte
            assert_eq!(Bitboard::<U19>::alignment(), 1);     // ibid
        }
    }

    mod size_alignment_and_layout {
        use super::*;

        #[test]
        fn total_bytes_needed() {
            // tic-tac-toe
            assert_eq!(Bitboard::<U3>::size(), 2);
            // chess board
            assert_eq!(Bitboard::<U8>::size(), 8);
            // go boards
            assert_eq!(Bitboard::<U19>::size(), 46);
        }
    }

    mod set_and_is_set {
        use super::*;

        #[test]
        fn set() {
            let mut tt = tic_tac_toe_board();

            tt.set(0,0); tt.set(1,1); tt.set(2,2);

            assert!(tt.is_set(0,0).ok().unwrap());
            assert!(tt.is_unset(0,1).ok().unwrap());
        }

        #[test]
        fn set_oob() {
            let mut go = go_board();

            let res = go.set(200,100);

            assert_eq!(res, Err(BitboardError::OutOfBounds(200,100)))
        }

    }

    mod unset_and_is_unset {
        use super::*;

        #[test]
        fn unset() {
            let mut tt = tic_tac_toe_board();

            tt.set(0,0);

            assert!(tt.is_set(0,0).ok().unwrap());

            tt.unset(0,0);

            assert!(tt.is_unset(0,0).ok().unwrap());
        }

        #[test]
        fn unset_oob() {
            let mut go = go_board();

            let res = go.unset(200,100);

            assert_eq!(res, Err(BitboardError::OutOfBounds(200,100)))
        }
    }

    mod flip {
        use super::*;

        #[test]
        fn flip() {
            let mut tt = tic_tac_toe_board();

            assert!(tt.is_unset(0,0).ok().unwrap());

            tt.flip(0,0);

            assert!(tt.is_set(0,0).ok().unwrap());

            tt.flip(0,0);

            assert!(tt.is_unset(0,0).ok().unwrap());
        }

        #[test]
        fn flip_oob() {
            let mut go = go_board();

            let res = go.flip(200,100);

            assert_eq!(res, Err(BitboardError::OutOfBounds(200,100)))
        }
    }

    mod debug {
        use super::*;

        use std::io::{Write};

        #[test]
        fn formats_tic_tac_toe_bitboard() {
            let mut c = tic_tac_toe_board();

            for i in 0..3 { c.set(i,i); }

            println!("");
            println!("{:?}", c);

            let mut b = vec![];
            let expected = vec![
                '1','0','0','\n',
                '0','1','0','\n',
                '0','0','1','\n'];

            write!(&mut b, "{:?}", c);

            for i in 0..12 { // 12 = 9 squares + 3 newlines
                assert_eq!(b[i], expected[i] as u8);
            }
        }

        #[test]
        fn formats_chess_bitboard() {
            let mut c = chess_board();

            for i in 0..8 { c.set(i,i); }

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

            write!(&mut b, "{:?}", c);

            for i in 0..72 { // 72 == 64 squares + 8 newlines
                assert_eq!(b[i], expected[i] as u8);
            }
        }
    }

    mod display {
        use super::*;

        use std::io::{Write};

        #[test]
        fn formats_tic_tac_toe_bitboard() {
            let mut c = tic_tac_toe_board();

            for i in 0..3 { c.set(i,i); }

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
        fn formats_chess_bitboard() {
            let mut c = chess_board();

            for i in 0..8 { c.set(i,i); }

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
}

#[cfg(test)]
#[allow(unused_must_use)]
mod benches {
    use super::*;

    use typenum::consts::*;
    use test::Bencher;

    fn tic_tac_toe_board() -> Bitboard<U3> { Bitboard::new() }
    fn chess_board() -> Bitboard<U8> { Bitboard::new() }
    fn go_board() -> Bitboard<U19> { Bitboard::new() }


    fn prepped_ttt_board() -> Bitboard<U3> {
        let mut bb1 = tic_tac_toe_board();

        bb1.set(0,0);
        bb1.set(1,1);
        bb1.set(2,2);

        bb1
    }

    fn prepped_chess_board() -> Bitboard<U8> {
        let mut bb1 = chess_board();

        bb1.set(0,6);
        bb1.set(1,0);
        bb1.set(1,3);
        bb1.set(3,5);
        bb1.set(7,1);
        bb1.set(1,2);
        bb1.set(3,0);
        bb1.set(2,3);

        bb1
    }

    fn prepped_go_board() -> Bitboard<U19> {
        let mut bb1 = go_board();

        bb1.set(0,6);
        bb1.set(1,0);
        bb1.set(1,3);
        bb1.set(3,5);
        bb1.set(7,1);
        bb1.set(1,2);
        bb1.set(3,0);
        bb1.set(2,3);

        bb1.set(1,16);
        bb1.set(11,11);
        bb1.set(1,14);
        bb1.set(13,15);
        bb1.set(7,11);
        bb1.set(1,12);
        bb1.set(3,10);
        bb1.set(18,13);

        bb1
    }

    mod eq {
        use super::*;


        #[bench]
        fn large(b: &mut Bencher) {
            let bb1 = &test::black_box(prepped_go_board());
            let bb2 = &test::black_box(prepped_go_board());
            b.iter(|| {
                bb1.to_owned() == bb2.to_owned()
            });
        }

        #[bench]
        fn medium(b: &mut Bencher) {
            let bb1 = &test::black_box(prepped_chess_board());
            let bb2 = &test::black_box(prepped_chess_board());
            b.iter(|| {
                bb1.to_owned() == bb2.to_owned()
            });
        }

        #[bench]
        fn small(b: &mut Bencher) {
            let bb1 = &test::black_box(prepped_ttt_board());
            let bb2 = &test::black_box(prepped_ttt_board());
            b.iter(|| {
                bb1.to_owned() == bb2.to_owned()
            });
        }
    }

    mod bitxor {
        use super::*;

        #[bench]
        fn large(b: &mut Bencher) {
            let bb1 = &test::black_box(prepped_go_board());
            let bb2 = &test::black_box(prepped_go_board());
            b.iter(|| {
                bb1.to_owned() ^ bb2.to_owned()
            });
        }

        #[bench]
        fn medium(b: &mut Bencher) {
            let bb1 = &test::black_box(prepped_chess_board());
            let bb2 = &test::black_box(prepped_chess_board());
            b.iter(|| {
                bb1.to_owned() ^ bb2.to_owned()
            });
        }

        #[bench]
        fn small(b: &mut Bencher) {
            let bb1 = &test::black_box(prepped_ttt_board());
            let bb2 = &test::black_box(prepped_ttt_board());
            b.iter(|| {
                bb1.to_owned() ^ bb2.to_owned()
            });
        }
    }

    mod bitor {
        use super::*;

        #[bench]
        fn large(b: &mut Bencher) {
            let bb1 = &test::black_box(prepped_go_board());
            let bb2 = &test::black_box(prepped_go_board());
            b.iter(|| {
                bb1.to_owned() | bb2.to_owned()
            });
        }

        #[bench]
        fn medium(b: &mut Bencher) {
            let bb1 = &test::black_box(prepped_chess_board());
            let bb2 = &test::black_box(prepped_chess_board());
            b.iter(|| {
                bb1.to_owned() | bb2.to_owned()
            });
        }

        #[bench]
        fn small(b: &mut Bencher) {
            let bb1 = &test::black_box(prepped_ttt_board());
            let bb2 = &test::black_box(prepped_ttt_board());
            b.iter(|| {
                bb1.to_owned() | bb2.to_owned()
            });
        }
    }

    mod bitand {
        use super::*;

        #[bench]
        fn large(b: &mut Bencher) {
            let bb1 = &test::black_box(prepped_go_board());
            let bb2 = &test::black_box(prepped_go_board());
            b.iter(|| {
                bb1.to_owned() & bb2.to_owned()
            });
        }

        #[bench]
        fn medium(b: &mut Bencher) {
            let bb1 = &test::black_box(prepped_chess_board());
            let bb2 = &test::black_box(prepped_chess_board());
            b.iter(|| {
                bb1.to_owned() & bb2.to_owned()
            });
        }

        #[bench]
        fn small(b: &mut Bencher) {
            let bb1 = &test::black_box(prepped_ttt_board());
            let bb2 = &test::black_box(prepped_ttt_board());
            b.iter(|| {
                bb1.to_owned() & bb2.to_owned()
            });
        }
    }

    mod clone {
        use super::*;

        #[bench]
        fn large(b: &mut Bencher) {
            let bb1 = &test::black_box(prepped_go_board());
            b.iter(|| {
                bb1.clone()
            });
        }

        #[bench]
        fn medium(b: &mut Bencher) {
            let bb1 = &test::black_box(prepped_chess_board());
            b.iter(|| {
                bb1.clone()
            });
        }

        #[bench]
        fn small(b: &mut Bencher) {
            let bb1 = &test::black_box(prepped_ttt_board());
            b.iter(|| {
                bb1.clone()
            });
        }
    }

}


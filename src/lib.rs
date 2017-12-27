#![feature(unique, alloc, heap_api, allocator_api, test)]

//! Bitboard, compile-time-sized, typesafe, low level bitboards for Rust.

#![recursion_limit="256"]
extern crate typenum;

#[cfg(test)]
extern crate test;

use std::ptr;
use std::mem;
use std::ops;
use std::hash;
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
    // temporarily while I bughunt
    pub ptr: *mut u64,
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
                Ok(p) => ptr = p as *mut u64,
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

    pub fn any_set(&self) -> bool {
        let s = Self::pointer_size() as isize;
        for amt in 0..s {
            unsafe {
                if *self.ptr.offset(amt) != 0 {
                    if amt + 1 == s { // we're on the last byte, so re-do the check with the mask
                        let mask = Self::last_byte_mask();
                        // if it's the same as the mask, there are no bits set that are relevant, and
                        // this will return true
                        return (*self.ptr.offset(amt) | mask) != mask
                    } else {
                        // inequality on any other byte is 'real' since all bits are relevant.
                        return true;
                    }
                }
            }
        }
        return false;
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


    fn coords_to_offset_and_pos(x: usize, y: usize) -> (isize, u64) {
        let pos = x + y * N::to_usize();
        let byte_offset = pos / Self::alignment_bits();
        let bit_pos = 1 << (pos % Self::alignment_bits());
        (byte_offset as isize, bit_pos)
    }

    #[inline(always)]
    fn is_out_of_bounds(x: usize, y: usize) -> bool {
        !(Self::in_bounds(x) && Self::in_bounds(y))
    }

    #[inline(always)]
    fn in_bounds(i: usize) -> bool {
        i <= N::to_usize()
    }

    /// The last byte will sometimes contain junk, since we allocate more than we need. This
    /// calculates a mask of the relevant bits in the final byte.
    #[inline(always)]
    fn last_byte_mask() -> u64 {
        // bit overage == size*alignment - n^2; should always be <8
        //
        // e.g., for 5x5, n^2 = 25. alignment = 8 bit, size = 4 bytes, so
        //
        // 32 - 25 = 7.
        //
        // for 4x4, n^2 = 16, alignment = 8bit, size = 2 bytes, so
        //
        // 8 * 2 - 16 = 0;
        //
        // no overage.
        //
        // for 6x6
        //
        // 40 - 36 = 4 bits of overage.
        //
        // Overage = number of bits unneeded, they always come from the high side of the byte. So
        // the needed bits are just :
        //
        // (!0) >> overage
        //
        // thus the mask is
        //
        // !((!0) >> overage)

        let size = Self::pointer_size() * Self::alignment_bits();
        let slots = N::to_usize().pow(2);
        let bit_overage = size - slots;
        !(!0 >> bit_overage)
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
    ///        assert_eq!(Bitboard::<U8>::alignment(), 8);
    ///    }
    ///```
    #[inline(always)]
    pub fn alignment() -> usize {
        mem::align_of::<u64>()
    }

    /// Return the alignment, in bits, of the Bitboard
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
    ///        assert_eq!(Bitboard::<U8>::alignment_bits(), 64);
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
    ///        assert_eq!(Bitboard::<U8>::byte_size(), 8);
    ///        // go boards
    ///        assert_eq!(Bitboard::<U19>::byte_size(), 48);
    ///    }
    ///```
    #[inline(always)]
    pub fn byte_size() -> usize {
        Self::bits_needed() / 8
    }

    // the number of pointers we need to iterate through
    fn pointer_size() -> usize {
        Self::bits_needed() / Self::alignment_bits()
    }

    /// Calculate the memory layout for the bitboard.
    fn layout() -> Layout {
        Layout::from_size_align(Self::byte_size(), Self::alignment()).unwrap()
    }
}

impl<N : Unsigned> Drop for Bitboard<N> {
    fn drop(&mut self) {
        let layout = Self::layout();
        unsafe { Heap.dealloc(self.ptr as *mut _, layout); }
    }
}

impl<N : Unsigned> fmt::Debug for Bitboard<N> {
    #[allow(unused_must_use)]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s = Self::pointer_size();
        writeln!(f);
        write!(f, "data: ");

        for i in 0..s {
            unsafe { write!(f, "{:064b} ", *self.ptr.offset(i as isize)); }
        }
        writeln!(f);
        write!(f, "mask: ");
        for _ in 0..(s-1) {
            write!(f, "{:064b} ", 0);
        }
        write!(f, "{:064b}", Self::last_byte_mask())
    }
}

impl<N : Unsigned> fmt::Display for Bitboard<N> {
    #[allow(unused_must_use)]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s = N::to_usize();

        for c in 0..s {
            for r in 0..s {
                if self.is_set(r,c).ok().unwrap() {
                    write!(f, "{}", 1);
                } else {
                    write!(f, "{}", 0);
                }
            }
            writeln!(f);
        }
        write!(f, "")
    }
}

impl<N : Unsigned> cmp::PartialEq for Bitboard<N> {
    fn eq(&self, other: &Bitboard<N>) -> bool {
        let mut acc = true;
        let size = Self::pointer_size() as isize;
        let mask = Self::last_byte_mask();

        // we know the sizes are the same because `N` is the same, and `A` is the same
        for amt in 0..size {
            unsafe {
                let mut s = *self.ptr.offset(amt);
                let mut o = *other.ptr.offset(amt);

                    acc &= s == o
                        || ((amt + 1 == size) && ((s | mask) == (o | mask)));

                if !acc { return acc; }
            }
        }
        return acc;
    }
}

impl<N : Unsigned> cmp::Eq for Bitboard<N> { }


impl<N : Unsigned> hash::Hash for Bitboard<N> {
    fn hash<H : hash::Hasher>(&self, state: &mut H) {
        let s = Self::pointer_size() as isize;
        for amt in 0..s {
            if amt+1 == s as isize { // we're on the last byte, so re-do the check with the mask
                let mask = Self::last_byte_mask();
                // we need to mask off the end bits so we don't get incorrect hashes
                unsafe { (*self.ptr.offset(amt) | mask).hash(state); }
            } else {
                unsafe { (*self.ptr.offset(amt)).hash(state); }
            }
        }
    }
}

impl<N: Unsigned> Clone for Bitboard<N> {
    fn clone(&self) -> Bitboard<N> {
        let new_bb : Bitboard<N> = Bitboard::new();
        unsafe {
            ptr::copy(self.ptr as *const u64, new_bb.ptr, Self::pointer_size());
        }
        return new_bb;
        //let new_bb : Bitboard<N> = Bitboard::new();
        //for amt in 0..Self::pointer_size() {
            //unsafe { *new_bb.ptr.offset(amt as isize) = *self.ptr.offset(amt as isize) }
        //}
        //return new_bb;
    }
}

impl<N : Unsigned> ops::BitAnd for Bitboard<N> {
    type Output = Bitboard<N>;

    fn bitand(self, other: Bitboard<N>) -> Bitboard<N> {
        let new_bb : Bitboard<N> = Bitboard::new();
        for amt in 0..(Self::pointer_size() as isize) {
            unsafe {
                *new_bb.ptr.offset(amt) = (*self.ptr.offset(amt)) & (*other.ptr.offset(amt))
            }
        }
        return new_bb
    }
}

impl<N : Unsigned> ops::BitAndAssign for Bitboard<N> {
    fn bitand_assign(&mut self, other: Bitboard<N>) {
        for amt in 0..(Self::pointer_size() as isize) {
            unsafe {
                *self.ptr.offset(amt) &= *other.ptr.offset(amt)
            }
        }
    }
}

impl<N : Unsigned> ops::BitOr for Bitboard<N> {
    type Output = Bitboard<N>;

    fn bitor(self, other: Bitboard<N>) -> Bitboard<N> {
        let new_bb : Bitboard<N> = Bitboard::new();
        for amt in 0..(Self::pointer_size() as isize) {
            unsafe {
                *new_bb.ptr.offset(amt) = (*self.ptr.offset(amt)) | (*other.ptr.offset(amt))
            }
        }
        return new_bb
    }
}

impl<N : Unsigned> ops::BitOrAssign for Bitboard<N> {
    fn bitor_assign(&mut self, other: Bitboard<N>) {
        for amt in 0..(Self::pointer_size() as isize) {
            unsafe {
                *self.ptr.offset(amt) |= *other.ptr.offset(amt)
            }
        }
    }
}

impl<N : Unsigned> ops::BitXor for Bitboard<N> {
    type Output = Bitboard<N>;

    fn bitxor(self, other: Bitboard<N>) -> Bitboard<N> {
        let new_bb : Bitboard<N> = Bitboard::new();
        for amt in 0..(Self::pointer_size() as isize) {
            unsafe {
                *new_bb.ptr.offset(amt) = (*self.ptr.offset(amt)) ^ (*other.ptr.offset(amt))
            }
        }
        return new_bb
    }
}

impl<N : Unsigned> ops::BitXorAssign for Bitboard<N> {
    fn bitxor_assign(&mut self, other: Bitboard<N>) {
        for amt in 0..(Self::pointer_size() as isize) {
            unsafe {
                *self.ptr.offset(amt) ^= *other.ptr.offset(amt)
            }
        }
    }
}

impl<N : Unsigned> ops::Not for Bitboard<N> {
    type Output = Bitboard<N>;

    fn not(self) -> Bitboard<N> {
        let new_bb : Bitboard<N> = self.clone();

        let s = Self::pointer_size() as isize;
        for amt in 0..s {
            if amt+1 == s {
                let mask = Self::last_byte_mask();
                unsafe {
                    *new_bb.ptr.offset(amt) = !*self.ptr.offset(amt);
                    *new_bb.ptr.offset(amt) |= mask;
                }
            } else {
                unsafe { *new_bb.ptr.offset(amt) = !*self.ptr.offset(amt); }
            }
        }

        return new_bb;
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

            unsafe {
                for i in 0..Bitboard::<U19>::pointer_size() as isize {
                    assert_eq!(*bb1.ptr.offset(i), 0);
                    assert_eq!(*bb2.ptr.offset(i), 0);
                }
            }
            // these are separate objects
            assert_ne!(bb1.ptr, bb2.ptr);
            // equality is by value
            assert_eq!(bb1, bb2);
        }


        #[test]
        fn inverse_eq_identity() {
            let bb1 = go_board();
            let bb2 = go_board();

            // these are separate objects
            assert_ne!(bb1.ptr, bb2.ptr);
            // equality is by value
            assert_eq!(!bb1, !bb2);
        }

        #[test]
        fn all_set_eq_inverse_of_blank() {
            let mut bb1 = tic_tac_toe_board();
            let bb2 = tic_tac_toe_board();

            for i in 0..3 {
                for j in 0..3 {
                    bb1.set(i,j);
                }
            }

            // these are separate objects
            assert_ne!(bb1.ptr, bb2.ptr);
            // equality is by value
            assert_eq!(bb1, !bb2);
        }

        #[test]
        fn errant_eq_repro() {
            // Reproduce a bug from 23-DEC-2017
            let mut qs = Bitboard::<U4>::new();
            for i in 0..4 {
                for j in 0..4 {
                    qs.set(i,j);
                }
            }

            let mut test = Bitboard::<U5>::new();

            qs.unset(1,3);
            let inv = !Bitboard::new();

            test.set(1,3);

            assert_ne!(qs, inv);
        }
    }

    mod alloc {
        use super::*;

        #[test]
        fn allocs_zeroed() {
            let g = go_board();

            unsafe {
                for i in 0..Bitboard::<U19>::pointer_size() as isize {
                    assert_eq!(*g.ptr.offset(i), 0);
                }
            }
        }
    }

    mod bitnot {
        use super::*;

        #[test]
        fn inverse() {
            // 100
            // 010
            // 001
            let mut bb1 = tic_tac_toe_board();

            // 011
            // 101
            // 110
            let mut expected = tic_tac_toe_board();

            bb1.set(0,0);
            bb1.set(1,1);
            bb1.set(2,2);

            expected.set(1,0);
            expected.set(2,0);
            expected.set(0,1);
            expected.set(2,1);
            expected.set(0,2);
            expected.set(1,2);

            assert_eq!(!bb1.clone(), expected.clone());
            assert_eq!(bb1, !expected);
        }

        #[test]
        fn inverse_and_equals_zero() {
            // 100
            // 010
            // 001
            let mut bb1 = tic_tac_toe_board();

            let expected = tic_tac_toe_board();

            bb1.set(0,0);
            bb1.set(1,1);
            bb1.set(2,2);

            let not_bb1 = !(bb1.clone());

            assert_eq!(bb1 & not_bb1, expected);
        }

        #[test]
        fn self_inverse() {
            let mut bb1 = tic_tac_toe_board();
            let mut expected = Bitboard::new();


            assert_eq!(!(!bb1.clone()).clone(), expected);

            bb1.set(1,2);
            expected.set(1,2);

            assert_eq!(!(!bb1.clone()).clone(), expected);
        }

    }

    mod clone {
        use super::*;

        #[test]
        fn clone() {
            let mut bb1 = tic_tac_toe_board();

            bb1.set(0,0);
            bb1.set(1,1);
            bb1.set(2,2);

            assert_eq!(bb1, bb1.clone());
        }

    }

    mod coord_calculation {
        use super::*;

        #[test]
        fn sets_appropriately() {
            let mut b;
            unsafe {
                b = Bitboard::<U5>::new();
                b.set(0,0); assert_eq!(*b.ptr, 1);

                b = Bitboard::<U5>::new();
                b.set(1,0); assert_eq!(*b.ptr, 2);

                b = Bitboard::<U5>::new();
                b.set(2,0); assert_eq!(*b.ptr, 4);

                b = Bitboard::<U5>::new();
                b.set(3,0); assert_eq!(*b.ptr, 8);

                b = Bitboard::<U5>::new();
                b.set(4,0); assert_eq!(*b.ptr, 16);

                b = Bitboard::<U5>::new();
                b.set(0,1); assert_eq!(*b.ptr, 32);

                b = Bitboard::<U5>::new();
                b.set(1,1); assert_eq!(*b.ptr, 64);

                b = Bitboard::<U5>::new();
                b.set(2,1); assert_eq!(*b.ptr, 128);

                b = Bitboard::<U5>::new();
                b.set(3,1); assert_eq!(*b.ptr, 256);

                b = Bitboard::<U5>::new();
                b.set(4,1); assert_eq!(*b.ptr, 512);

                b = Bitboard::<U5>::new();
                b.set(0,2); assert_eq!(*b.ptr, 1024);
            }
        }


        #[test]
        fn maps_to_linear_sequence_small() {
            let mut positions = vec![];
            for j in 0..3 {
                for i in 0..3 {
                    positions.push(Bitboard::<U3>::coords_to_offset_and_pos(i,j));
                }
            }

            let expected_positions = vec![
                (0,1<<0), (0,1<<1), (0,1<<2),
                (0,1<<3), (0,1<<4), (0,1<<5),
                (0,1<<6), (0,1<<7), (0,1<<8)
            ];

            assert_eq!(positions, expected_positions);
        }

        #[test]
        fn maps_to_linear_sequence_4x4() {
            let mut positions = vec![];
            for j in 0..4 {
                for i in 0..4 {
                    positions.push(Bitboard::<U4>::coords_to_offset_and_pos(i,j));
                }
            }

            let expected_positions = vec![
                (0,1<<0),  (0,1<<1),  (0,1<<2),  (0,1<<3),
                (0,1<<4),  (0,1<<5),  (0,1<<6),  (0,1<<7),
                (0,1<<8),  (0,1<<9),  (0,1<<10), (0,1<<11),
                (0,1<<12), (0,1<<13), (0,1<<14), (0,1<<15)
            ];

            assert_eq!(positions, expected_positions);
        }

        #[test]
        fn maps_to_linear_sequence_medium() {
            let mut positions = vec![];
            for j in 0..8 {
                for i in 0..8 {
                    positions.push(Bitboard::<U8>::coords_to_offset_and_pos(i,j));
                }
            }

            let expected_positions = vec![
                (0,1<<0 ), (0,1<<1 ), (0,1<<2 ), (0,1<<3 ), (0,1<<4 ), (0,1<<5 ), (0,1<<6 ), (0,1<<7 ),
                (0,1<<8 ), (0,1<<9 ), (0,1<<10), (0,1<<11), (0,1<<12), (0,1<<13), (0,1<<14), (0,1<<15),
                (0,1<<16), (0,1<<17), (0,1<<18), (0,1<<19), (0,1<<20), (0,1<<21), (0,1<<22), (0,1<<23),
                (0,1<<24), (0,1<<25), (0,1<<26), (0,1<<27), (0,1<<28), (0,1<<29), (0,1<<30), (0,1<<31),
                (0,1<<32), (0,1<<33), (0,1<<34), (0,1<<35), (0,1<<36), (0,1<<37), (0,1<<38), (0,1<<39),
                (0,1<<40), (0,1<<41), (0,1<<42), (0,1<<43), (0,1<<44), (0,1<<45), (0,1<<46), (0,1<<47),
                (0,1<<48), (0,1<<49), (0,1<<50), (0,1<<51), (0,1<<52), (0,1<<53), (0,1<<54), (0,1<<55),
                (0,1<<56), (0,1<<57), (0,1<<58), (0,1<<59), (0,1<<60), (0,1<<61), (0,1<<62), (0,1<<63)
            ];

            assert_eq!(positions, expected_positions);
        }

    }

    mod bitxor {
        use super::*;

        #[test]
        fn disjoint_sum() {
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
        fn disjoint_sum_happens_during_assign() {
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

            bb1 ^= bb2;

            assert_eq!(bb1, expected);
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
        fn assign_performs_union() {
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

            bb1 |= bb2;

            assert_eq!(bb1, expected);
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
        fn assign_does_intersection_in_place() {
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

            bb1 &= bb2;

            assert_eq!(bb1, expected);
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
            assert_eq!(Bitboard::<U8>::byte_size(), 8);
            assert_eq!(Bitboard::<U19>::byte_size(), 48);
        }

        #[test]
        fn statically_assigned_alignment() {
            assert_eq!(Bitboard::<U8>::alignment(), 8);
            assert_eq!(Bitboard::<U19>::alignment(), 8);
        }
    }

    mod size_alignment_and_layout {
        use super::*;

        #[test]
        fn total_bytes_needed() {
            // tic-tac-toe
            assert_eq!(Bitboard::<U3>::byte_size(), 8); // hits a minimum for alignment purposes
            // chess board
            assert_eq!(Bitboard::<U8>::byte_size(), 8);
            // go boards
            assert_eq!(Bitboard::<U19>::byte_size(), 48);
        }

        #[test]
        fn total_pointers_needed() {
            // tic-tac-toe
            assert_eq!(Bitboard::<U3>::pointer_size(), 1);
            // chess board
            assert_eq!(Bitboard::<U8>::pointer_size(), 1);
            // go boards
            assert_eq!(Bitboard::<U19>::pointer_size(), 6);
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
        fn any_set() {
            let mut tt = tic_tac_toe_board();

            assert!(!tt.any_set());

            tt.set(0,0); tt.set(1,1); tt.set(2,2);

            assert!(tt.any_set());
        }

        #[test]
        fn set_oob() {
            let mut go = go_board();

            let res = go.set(200,100);

            assert_eq!(res, Err(BitboardError::OutOfBounds(200,100)))
        }

        #[test]
        fn no_double_sets() {
            let mut g = go_board();

            for i in 0..19 {
                for j in 0..19 {
                    if g.is_set(i,j).ok().unwrap() { panic!("Tried to double-set ({},{})", i, j); }
                    g.set(i,j);
                }
            }
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

    mod last_byte_mask {
        use super::*;

        #[test]
        fn for_3x3() {
            assert_eq!(Bitboard::<U3>::last_byte_mask(), !((1 << 9) - 1));
        }

        #[test]
        fn for_4x4() {
            assert_eq!(Bitboard::<U4>::last_byte_mask(), !((1 << 16) - 1));
        }

        #[test]
        fn for_5x5() {
            assert_eq!(Bitboard::<U5>::last_byte_mask(), !((1 << 25) - 1));
        }

        #[test]
        fn for_6x6() {
            assert_eq!(Bitboard::<U6>::last_byte_mask(), !((1 << 36) - 1));
        }

        #[test]
        fn for_7x7() {
            assert_eq!(Bitboard::<U7>::last_byte_mask(), !((1 << 49) - 1));
        }

        #[test]
        fn for_8x8() {
            assert_eq!(Bitboard::<U8>::last_byte_mask(), 0);
        }

        #[test]
        fn for_10x10() {
            assert_eq!(Bitboard::<U10>::last_byte_mask(), !((1 << 36) - 1));
        }
    }

    mod debug {
        // FIXME: Disabled for now till I can find a better way to notate the output.

        //use super::*;

        //use std::io::{Write};

        //#[test]
        //fn formats_tic_tac_toe_bitboard() {
            //let mut c = tic_tac_toe_board();

            //// 101
            //// 010
            //// 001
            ////
            //// -> 00010101 XXXXXXX1

            //for i in 0..3 { c.set(i,i); }
            //c.set(2,0);

            //let mut b = vec![];
            //let expected = vec!['\n',
                //'d','a','t','a',':',' ','0','0','0','1','0','1','0','1',' ','0','0','0','0','0','0','0','1','\n',
                //'m','a','s','k',':',' ','0','0','0','1','0','1','0','1',' ','1','1','1','1','1','1','1','0','\n'];

            //write!(&mut b, "{:?}", c);

            //for i in 0..12 { // 12 = 9 squares + 3 newlines
                //assert_eq!(b[i], expected[i] as u8);
            //}
        //}
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
    fn giant_board() -> Bitboard<U100> { Bitboard::new() }

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

    mod set {
        use super::*;

        #[bench]
        fn giant(b: &mut Bencher) {
            let bb1 = &test::black_box(giant_board());
            b.iter(|| {
                bb1.to_owned().set(13,14)
            });
        }

        #[bench]
        fn large(b: &mut Bencher) {
            let bb1 = &test::black_box(prepped_go_board());
            b.iter(|| {
                bb1.to_owned().set(13,14)
            });
        }

        #[bench]
        fn medium(b: &mut Bencher) {
            let bb1 = &test::black_box(prepped_chess_board());
            b.iter(|| {
                bb1.to_owned().set(3,6)
            });
        }

        #[bench]
        fn small(b: &mut Bencher) {
            let bb1 = &test::black_box(prepped_ttt_board());
            b.iter(|| {
                bb1.to_owned().set(1,2)
            });
        }
    }

    mod eq {
        use super::*;

        #[bench]
        fn giant(b: &mut Bencher) {
            let bb1 = &test::black_box(giant_board());
            let bb2 = &test::black_box(giant_board());
            b.iter(|| {
                bb1.to_owned() == bb2.to_owned()
            });
        }

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

    mod bitxor_assign {
        use super::*;

        #[bench]
        fn giant(b: &mut Bencher) {
            let bb1 = &test::black_box(giant_board());
            let bb2 = &test::black_box(giant_board());
            b.iter(|| {
                let mut a = bb1.to_owned();
                let b = bb2.to_owned();
                a ^= b
            });
        }

        #[bench]
        fn large(b: &mut Bencher) {
            let bb1 = &test::black_box(prepped_go_board());
            let bb2 = &test::black_box(prepped_go_board());
            b.iter(|| {
                let mut a = bb1.to_owned();
                let b = bb2.to_owned();
                a ^= b
            });
        }

        #[bench]
        fn medium(b: &mut Bencher) {
            let bb1 = &test::black_box(prepped_chess_board());
            let bb2 = &test::black_box(prepped_chess_board());
            b.iter(|| {
                let mut a = bb1.to_owned();
                let b = bb2.to_owned();
                a ^= b
            });
        }

        #[bench]
        fn small(b: &mut Bencher) {
            let bb1 = &test::black_box(prepped_ttt_board());
            let bb2 = &test::black_box(prepped_ttt_board());
            b.iter(|| {
                let mut a = bb1.to_owned();
                let b = bb2.to_owned();
                a ^= b
            });
        }
    }

    mod bitnot {
        use super::*;


        #[bench]
        fn giant(b: &mut Bencher) {
            let bb1 = &test::black_box(giant_board());
            b.iter(|| {
               !bb1.to_owned()
            });
        }

        #[bench]
        fn large(b: &mut Bencher) {
            let bb1 = &test::black_box(prepped_go_board());
            b.iter(|| {
               !bb1.to_owned()
            });
        }

        #[bench]
        fn medium(b: &mut Bencher) {
            let bb1 = &test::black_box(prepped_chess_board());
            b.iter(|| {
                !bb1.to_owned()
            });
        }

        #[bench]
        fn small(b: &mut Bencher) {
            let bb1 = &test::black_box(prepped_ttt_board());
            b.iter(|| {
                !bb1.to_owned()
            });
        }
    }

    mod bitxor {
        use super::*;

        #[bench]
        fn giant(b: &mut Bencher) {
            let bb1 = &test::black_box(giant_board());
            let bb2 = &test::black_box(giant_board());
            b.iter(|| {
                bb1.to_owned() ^ bb2.to_owned()
            });
        }

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

    mod bitor_assign {
        use super::*;

        #[bench]
        fn giant(b: &mut Bencher) {
            let bb1 = &test::black_box(giant_board());
            let bb2 = &test::black_box(giant_board());
            b.iter(|| {
                let mut a = bb1.to_owned();
                let b = bb2.to_owned();
                a |= b
            });
        }

        #[bench]
        fn large(b: &mut Bencher) {
            let bb1 = &test::black_box(prepped_go_board());
            let bb2 = &test::black_box(prepped_go_board());
            b.iter(|| {
                let mut a = bb1.to_owned();
                let b = bb2.to_owned();
                a |= b
            });
        }

        #[bench]
        fn medium(b: &mut Bencher) {
            let bb1 = &test::black_box(prepped_chess_board());
            let bb2 = &test::black_box(prepped_chess_board());
            b.iter(|| {
                let mut a = bb1.to_owned();
                let b = bb2.to_owned();
                a |= b
            });
        }

        #[bench]
        fn small(b: &mut Bencher) {
            let bb1 = &test::black_box(prepped_ttt_board());
            let bb2 = &test::black_box(prepped_ttt_board());
            b.iter(|| {
                let mut a = bb1.to_owned();
                let b = bb2.to_owned();
                a |= b
            });
        }
    }

    mod bitor {
        use super::*;

        #[bench]
        fn giant(b: &mut Bencher) {
            let bb1 = &test::black_box(giant_board());
            let bb2 = &test::black_box(giant_board());
            b.iter(|| {
                bb1.to_owned() | bb2.to_owned()
            });
        }

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

    mod bitand_assign {
        use super::*;

        #[bench]
        fn giant(b: &mut Bencher) {
            let bb1 = &test::black_box(giant_board());
            let bb2 = &test::black_box(giant_board());
            b.iter(|| {
                let mut a = bb1.to_owned();
                let b = bb2.to_owned();
                a &= b
            });
        }

        #[bench]
        fn large(b: &mut Bencher) {
            let bb1 = &test::black_box(prepped_go_board());
            let bb2 = &test::black_box(prepped_go_board());
            b.iter(|| {
                let mut a = bb1.to_owned();
                let b = bb2.to_owned();
                a &= b
            });
        }

        #[bench]
        fn medium(b: &mut Bencher) {
            let bb1 = &test::black_box(prepped_chess_board());
            let bb2 = &test::black_box(prepped_chess_board());
            b.iter(|| {
                let mut a = bb1.to_owned();
                let b = bb2.to_owned();
                a &= b
            });
        }

        #[bench]
        fn small(b: &mut Bencher) {
            let bb1 = &test::black_box(prepped_ttt_board());
            let bb2 = &test::black_box(prepped_ttt_board());
            b.iter(|| {
                let mut a = bb1.to_owned();
                let b = bb2.to_owned();
                a &= b
            });
        }
    }

    mod bitand {
        use super::*;

        #[bench]
        fn giant(b: &mut Bencher) {
            let bb1 = &test::black_box(giant_board());
            let bb2 = &test::black_box(giant_board());
            b.iter(|| {
                bb1.to_owned() & bb2.to_owned()
            });
        }

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
        fn giant(b: &mut Bencher) {
            let bb1 = &test::black_box(giant_board());
            b.iter(|| {
                bb1.clone()
            });
        }

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


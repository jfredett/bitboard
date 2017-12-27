#![feature(alloc_system, global_allocator, allocator_api, test)]
extern crate bitboard;
extern crate typenum;

#[cfg(test)]
extern crate test;

// SWITCH TO GLOBAL ALLOCATOR FOR VALGRIND TO WORK
extern crate alloc_system;

use alloc_system::System;

#[global_allocator]
static A: System = System;
// END SWITCH

use std::hash::Hash;

use bitboard::Bitboard;
use typenum::consts::*;
use typenum::marker_traits::Unsigned;

use std::fmt;


#[derive(PartialEq, Eq, Clone, Debug, Hash)]
struct Queen<N : Unsigned> {
    pos: Bitboard<N>,
    effect: Bitboard<N>
}


fn main() {
    queen_solver::<U30>().unwrap().position_board();
}

impl<N : Unsigned> Queen<N> {
    // I have no idea why the coords need reversing here...
    #[allow(unused_must_use)]
    pub fn new(y: usize, x: usize) -> Queen<N> {
        let mut pos = Bitboard::new();
        pos.set(x, y);

        let mut effect = pos.clone();
        let s = N::to_usize();
        for i in 0..s {
            effect.set(i, y);
            effect.set(x, i);

            if (x + i) <= s && (y + i) <= s {
                effect.set(x + i, y + i);
                if x >= i { effect.set(x - i, y + i); }
                if y >= i { effect.set(x + i, y - i); }
                if x >= i && y >= i { effect.set(x - i, y - i); }
            }
        }

        Queen {
            pos: pos,
            effect: effect
        }
    }

    pub fn captures(&self, q: Queen<N>) -> bool {
        let clone = self.effect.clone();
        (clone & q.pos).any_set()
    }
}

#[derive(PartialEq, Eq, Clone, Hash)]
struct QueenSet<N : Unsigned> {
    queens: Vec<Queen<N>>,
    effect: Bitboard<N>
}

impl<N : Unsigned + Clone + PartialEq> QueenSet<N> {
    pub fn new() -> QueenSet<N> {
        QueenSet {
            queens: vec![],
            effect: Bitboard::new()
        }
    }

    /// only inserts if the queen isn't captured
    pub fn insert(&mut self, q: Queen<N>) {
        if self.captures(&q) { return; }
        self.effect |= q.effect.clone();
        self.queens.push(q);
    }

    pub fn captures(&self, q: &Queen<N>) -> bool {
        (self.effect.clone() & q.pos.clone()).any_set()
    }

    pub fn remove(&mut self, q: Queen<N>) {
        let mut qs = QueenSet::new();

        let local_q = q.clone();
        for queen in &self.queens {
            if *queen != local_q {
                qs.insert(queen.clone());
            }
        }
        self.queens = qs.queens;
        self.effect = qs.effect;
    }

    pub fn pop(&mut self) {
        let q = self.queens.last().unwrap().clone();
        self.remove(q);
    }

    pub fn out_of_room(&self) -> bool {
        let full_bb = !Bitboard::new();
        let result = self.effect == full_bb;
        result
    }

    pub fn solved(&self) -> bool {
        self.queens.len() == N::to_usize()
    }

    pub fn stuck(&self) -> bool {
        self.out_of_room() && !self.solved()
    }

    pub fn position_board(&self) -> Bitboard<N> {
        let mut bb = Bitboard::new();
        for queen in &self.queens {
            bb |= queen.pos.clone();
        }
        return bb;
    }
}

impl<N : Unsigned + Clone + PartialEq> fmt::Debug for QueenSet<N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "\n\n{}\n", self.position_board())
    }
}

/// produces a solution to the n-queens problem for an NxN board
///
/// Notably, it does this column-major, which is 'slower' in some sense since boards are stored
/// row-major
fn queen_solver<N : Unsigned + Clone + Eq + Hash>() -> Option<QueenSet<N>> {
    // column-major assortment of possible queens
    let mut possible_queens : Vec<Vec<Queen<N>>> = vec![];
    let s = N::to_usize();

    for i in 0..N::to_usize() {
        let mut v = vec![];
        for j in 0..N::to_usize() {
            v.push(Queen::new(j,i));
        }
        possible_queens.push(v);
    }

    let mut row_positions : Vec<usize> = vec![];
    let mut col = 0;

    let mut qs = QueenSet::new();
    let mut iters = 0;

    // start looking at the first square
    row_positions.push(0);
    loop {
        iters += 1;

        let mut row : usize = match row_positions.pop() {
            Some(v) => v,
            None if col == 0 => 0,
            _ => panic!("No solution found")
        };

        loop {
            if qs.solved() {
                return Some(qs);
            }

            if qs.stuck() || row == s {
                // backtrack column
                if col > 0 {
                    col -= 1;
                } else {
                    // bail out, no solutions!
                    return None;
                }
                // remove that column's queen
                qs.pop();

                break;
            }

            let q = &possible_queens[col][row];

            if !qs.captures(q) {
                // add the queen
                qs.insert(q.clone());
                // remember where we left off, the square after this one
                row_positions.push(row+1);
                // start at 0 on the next row.
                row_positions.push(0);
                // advance
                col += 1;
                // move on to next row
                break;
            }

            //otherwise
            row += 1;
        }
    }
}





// testing the tests!
mod tests {
    use super::*;

    mod queen {
        use super::*;
        #[test]
        fn captures() {
            let q : Queen<U8> = Queen::new(4,4);
            let p : Queen<U8> = Queen::new(4,6);

            assert!(q.clone().captures(p));

            let r : Queen<U8> = Queen::new(3,0);

            assert!(!q.clone().captures(r));
        }
    }

    mod queenset {
        use super::*;

        #[test]
        fn captures() {
            let mut qs = QueenSet::<U8>::new();
            qs.insert(Queen::new(4,4));

            assert!(qs.captures(&Queen::new(4,6)));
            assert!(!qs.captures(&Queen::new(3,0)));

            qs.insert(Queen::new(3,0));

            assert!(qs.captures(&Queen::new(3,0)));
        }

        #[test]
        fn queens_problem_4x4() {
            let mut qs = QueenSet::<U4>::new();
            // 0Q00
            // 000Q
            // Q000
            // 00Q0

            qs.insert(Queen::new(1,0));
            qs.insert(Queen::new(3,1));
            qs.insert(Queen::new(0,2));
            qs.insert(Queen::new(2,3));

            assert!(qs.out_of_room());
            assert!(!qs.stuck());
            assert!(qs.solved());
        }

        #[test]
        fn stuck_works() {
            let mut qs = QueenSet::<U4>::new();
            // Q...
            // ....
            // .Q..
            // ....

            qs.insert(Queen::new(0,0));
            qs.insert(Queen::new(1,2));

            assert_ne!(qs.effect, !Bitboard::new());
            assert!(!qs.out_of_room());
            assert!(!qs.stuck());
            assert!(!qs.solved());
        }

    }

    mod queen_solver {
        use super::*;

        #[test]
        fn for_8x8() {
          assert!(queen_solver::<U8>().is_some());
        }


        //#[test]
        // FIXME: This SIGSEGV's with an invalid mem ref. Not sure where
        fn for_16x16() {
          assert!(queen_solver::<U16>().is_some());
        }

        #[test]
        fn for_4x4() {
          assert!(queen_solver::<U4>().is_some());
        }

        #[test]
        fn for_3x3() {
          assert!(queen_solver::<U3>().is_none());
        }

    }
}


#[cfg(test)]
#[allow(unused_must_use)]
mod nqueens_problem {
    use super::*;

    use typenum::consts::*;
    use test::Bencher;

    #[bench]
    fn solve_12x12(b: &mut Bencher) {
        b.iter(|| { queen_solver::<U12>() });
    }


    #[bench]
    fn solve_8x8(b: &mut Bencher) {
        b.iter(|| { queen_solver::<U8>() });
    }

    #[bench]
    fn solve_4x4(b: &mut Bencher) {
        b.iter(|| { queen_solver::<U4>() });
    }
}

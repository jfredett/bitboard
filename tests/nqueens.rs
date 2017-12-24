//#![feature(alloc_system, global_allocator, allocator_api)]
extern crate bitboard;
extern crate typenum;

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
        println!("DEBUG: q.effect:\n{}\n q.position:\n{}\n", q.effect.clone(), q.pos.clone());
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
fn queen_solver<N : Unsigned + Clone + Eq + Hash>() {
    // column-major assortment of possible queens
    let mut possible_queens : Vec<Vec<Queen<N>>> = vec![];
    let s = N::to_usize();

    println!("Building bitboards");

    for i in 0..N::to_usize() {
        let mut v = vec![];
        for j in 0..N::to_usize() {
            v.push(Queen::new(i,j));
        }
        possible_queens.push(v);
    }

    let mut row_positions : Vec<usize> = vec![];
    let mut col = 0;

    let mut qs = QueenSet::new();

    // start looking at the first square
    row_positions.push(0);
    loop {
        let row_start : usize = row_positions.pop().unwrap();
        for row in row_start..s {
            println!("DEBUG: positions:\n{}\neffect:\n{}", qs.position_board(), qs.effect.clone());
            let q = &possible_queens[col][row];

            if !qs.captures(q) {
                // add the queen
                qs.insert(q.clone());
                // remember where we left off, the square after this one
                row_positions.push(row);
                // start at 0 on the next row.
                row_positions.push(0);
                // advance
                col += 1;
                // move on to next row
                break;
            }

            if qs.solved() { break; }

            if qs.stuck() || row + 1 == s {
                    // backtrack column
                    col -= 1;
                    // start where we left off on previous row
                    row_positions.pop();
                    // remove that column's queen
                    qs.pop();

                    break;
            }
        }

        if qs.solved() { break; }
    }

    println!("Solution to {}x{} board:\n{}", s, s, qs.position_board());

    assert!(false);
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

            unsafe {
                for i in 0..2 {
                    println!("effect-ptr + {} = {:08b}", i, *qs.effect.ptr.offset(i));
                }
            }

            println!("pos_board: {}", qs.position_board());
            println!("effect_board: {}", qs.effect.clone());
            println!("empty inverse: {}", !Bitboard::<U4>::new());

            assert_ne!(qs.effect, !Bitboard::new());
            assert!(!qs.out_of_room());
            assert!(!qs.stuck());
            assert!(!qs.solved());
        }

    }

    mod test {
        use super::*;

        #[test]
        fn queens() {
          //queen_solver::<U4>();
        }

    }
}

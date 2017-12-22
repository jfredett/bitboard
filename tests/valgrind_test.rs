#![feature(alloc_system, global_allocator, allocator_api)]
extern crate bitboard;
extern crate typenum;

// SWITCH TO GLOBAL ALLOCATOR FOR VALGRIND TO WORK
extern crate alloc_system;

use alloc_system::System;

#[global_allocator]
static A: System = System;
// END SWITCH

use bitboard::Bitboard;
use typenum::consts::*;

mod memory_allocation {
    use super::*;

    #[test]
    fn alloc() {
        let bb = Bitboard::<U20>::new();
        for byte in bb {
            assert_eq!(byte, 0);
        }
    }
}

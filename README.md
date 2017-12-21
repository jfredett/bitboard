# Bitboard

Bitboards in Rust.

## Summary

_This document is mostly lies at the moment._


[Bitboards](https://en.wikipedia.org/wiki/Bitboard) are a datastructure used in various sorts of
boardgame AI to represent boardgame state. Their primary appeal is that they are cache conscious and
primarily interacted with via simple binary logic operations (AND, OR, NOT, etc), and thus most
operations translate to very few instructions, and usually just the same instruction executed
repeatedly.  This means they're very very fast.

The Bitboards in this project are implemented in Rust as contiguous chunks of memory. Exact
alignment is configurable by the user, and size is statically determined via the type system (using
the [typenum](https://crates.io/crates/typenum) library.

What this means practically is two things.

1. You will get every bit of that delicious cache consciousness.
2. You will have compile-time guarantee's that when combining two bitboards, the combination is
   valid inasmuch as bitboard sizes are concerned.

## What's in this Repo

* Bitboard implementation with a reasonably nice API for creating, combining, and otherwise using
  them.
* Lots of tests.
* Benchmarks
* Some lovely documentation.


## What's not in this repo

* Examples of how to use bitboards (forthcoming in another crate, probably)
* The secret to life.
* My [recipe for baguette](https://www.sharelatex.com/read/kmcwvwhwgkjg), based on Julia Child's
  recipe.

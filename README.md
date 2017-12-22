# Bitboard

Bitboards in Rust.

[![Build Status](https://travis-ci.org/jfredett/bitboard.svg?branch=master)](https://travis-ci.org/jfredett/bitboard)
[![Coverage Status](https://coveralls.io/repos/github/jfredett/bitboard/badge.svg?branch=master)](https://coveralls.io/github/jfredett/bitboard?branch=master)


## Summary

_This document is mostly lies at the moment._


[Bitboards](https://en.wikipedia.org/wiki/Bitboard) are a datastructure used in various sorts of
boardgame AI to represent boardgame state. Their primary appeal is that they are cache conscious and
primarily interacted with via simple binary logic operations (AND, OR, NOT, etc), and thus most
operations translate to very few instructions, and usually just the same instruction executed
repeatedly.  This means they're very very fast.

The Bitboards in this project are implemented in Rust as contiguous chunks of
memory.  Size is statically determined via the type system (using the
[typenum](https://crates.io/crates/typenum) library.

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
* Any kind of advanced bitboards (no rotated / magic bitboards), or
  functionality to support them directly, yet.
    - It is in the cards for the future to support being able to do at least
      90/270 rotations. I don't know about 45s yet.

## What I think would be cool in this repo

* [SIMD is pretty neat.](https://doc.rust-lang.org/1.0.0/std/simd/struct.u64x2.html)
* Maybe some tooling for dealing with sets of bitboards / doing those queries
  efficiently / generally managing a set of bitboards.
    - The purpose of this repo is to support a future project of mine, so I
      suspect features I need will filter backward.

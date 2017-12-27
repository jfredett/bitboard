# Bitboard

Bitboards in Rust.

[![Build Status](https://travis-ci.org/jfredett/bitboard.svg?branch=master)](https://travis-ci.org/jfredett/bitboard)
[![Coverage Status](https://coveralls.io/repos/github/jfredett/bitboard/badge.svg?branch=master)](https://coveralls.io/github/jfredett/bitboard?branch=master)
[![crates.io](https://img.shields.io/crates/v/bitboard.svg)](https://img.shields.io/crates/v/bitboard.svg)
[![Maintenance Status](https://img.shields.io/badge/maintenance-actively--developed-brightgreen.svg)](https://img.shields.io/badge/maintenance-actively--developed-brightgreen.svg)


## Summary

[Bitboards](https://en.wikipedia.org/wiki/Bitboard) are a datastructure used in
various sorts of boardgame AI to represent boardgame state. Their primary appeal
is that they are cache conscious and primarily interacted with via simple binary
logic operations (AND, OR, NOT, etc), and thus most operations translate to very
few instructions, and usually just the same instruction executed repeatedly.
This means they're very very fast.

The Bitboards in this project are implemented in Rust as contiguous chunks of
memory.  Size is statically determined via the type system (using the
[typenum](https://crates.io/crates/typenum) library.

What this means practically is two things.

1. You will get every bit of that delicious cache consciousness.
2. You will have compile-time guarantee's that when combining two bitboards, the
   combination is valid inasmuch as bitboard sizes are concerned.

## What's in this Repo

* Bitboard implementation with a reasonably nice API for creating, combining,
  and otherwise using them.
* Lots of tests.
* Benchmarks
* An N-Queens Solver example of how to use bitboards.
* Some lovely documentation.

## What's not in this repo

* The secret to life.
* My [recipe for baguette](https://www.sharelatex.com/read/kmcwvwhwgkjg), based
  on Julia Child's recipe.
* Any kind of advanced bitboards (no rotated / magic bitboards), or
  functionality to support them directly, yet.
  - These will probably come in some day soonish, as I need them for other
    projects consuming this crate

## What I think would be cool in this repo

* Maybe some tooling for dealing with sets of bitboards / doing those queries
  efficiently / generally managing a set of bitboards.
    - The purpose of this repo is to support a future project of mine, so I
      suspect features I need will filter backward.

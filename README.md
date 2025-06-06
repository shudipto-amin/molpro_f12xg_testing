# README

This repo is for developing test routines for [F12-XG implementation](https://github.com/ak-ustutt/molpro/issues/1) in Molpro

## Scripts
`gen_input.py` : Generates molpro input files for various ansatzes and codebases (writes to `outputs/`)

`analyze_outputs.py` : Analyzes outputs generated from generated input files above.

## Reference files (!!Need to add!!)

`neon_df-hf_df-mp2-f12_aug-cc-pVDZ.*.{xml, out}`

These files are from older dates (before F12-devel was implemented) and can be used for reference.

## ITF files read by different ansatzes
| Ansatz         | ITF                       |
|----------------|---------------------------|
| 3C-FIX-HY1-NOZ | DfMp2-f12.itfca           |
|                | DfMp2-f12_cfixhy1ca.itfca |
| 3-C-FIX-NOZ    | DfMp2-f12.itfca           |
|                | DfMp2-f12_scfixca.itfca   |
| 3C-FIX-NOZ     | DfMp2-f12.itfca           |
|                | DfMp2-f12_cfixca.itfca    |

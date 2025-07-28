# README

This repo is for developing test routines for [F12-XG implementation](https://github.com/ak-ustutt/molpro/issues/1) in Molpro

## Current Development goals:

- [ ] Determine **optimal** gammas of an array individual elements. Folder: `/optimize_gamma_by_element`
   - [ ] Workflow for generating different data for single elements
   - [ ] Make tables of optimal gamma by element and basis set; expand `/optimize_gamma_by_element/plot.ipynb`
- [ ] Run XG on a system containing two such different gammas
   - [ ] Run variations on NaCl, or NH3, or ZnH2, or any closed shell from the paper on core/valence specific gammas.
- [ ] Get accurate reference energy of the dimer systems
   - [ ] Option A: large basis set, or
   - [ ] Option B: extrapolate from several basis set sizes.
- [ ] Compare XG with reference energy to see whether or not different gamma implementation (by atom) has made any improvements toward the "accurate" energy. 

## Practical sub-goals
- [ ] Analyze sizes of F12 tensors from output file and tabulate
- [ ] Analyze largest contribution from log file  by sorting

## Scripts
`test_molpro_dev/gen_input.py` : Generates molpro input files for various ansatzes and codebases (writes to `outputs/`)

`test_molpro_dev/analyze_outputs.py` : Analyzes outputs generated from generated input files above.

`test_molpro_dev/optimize_gamma_by_element/gen_input.py` : Generates input file content based on atom name, gammas, etc

`test_molpro_dev/optimize_gamma_by_element/plot.ipynb` : Plots E vs gamma

`test_molpro_dev/optimize_gamma_by_element/multi_inputs.sh` : Generates (using `gen_input.py` in same folder) many input files based on atom, basis, into the `outputs/` folder

`test_molpro_dev/f12xg_inputs/generate_gauss_from_gamma.ipynb` : Write Expfile.

## Reference files (!!Need to add!!)

`neon_df-hf_df-mp2-f12_aug-cc-pVDZ.*.{xml, out}`

These files are from older dates (before F12-devel was implemented) and can be used for reference.

## ITF files read by different ansatzes
| Ansatz         | ITF                       |
|----------------|---------------------------|
| 3C-FIX-HY1-NOZ | `DfMp2-f12.itfca`           |
|                | `DfMp2-f12_cfixhy1ca.itfca` |
| 3-C-FIX-NOZ    | `DfMp2-f12.itfca`           |
|                | `DfMp2-f12_scfixca.itfca`   |
| 3C-FIX-NOZ     | `DfMp2-f12.itfca`           |
|                | `DfMp2-f12_cfixca.itfca`    |

# Setting up interaction energies

* Create new folder, such as `cu_nh3` for each dimer
   * Create sub-folder such as `cu_nh3/{standard,xg}/`
* Create a `distances.txt`
* Inside each subfolder `standard/` and `xg/` create template input file like `standard_gamma_1.9.inp`
   * Make sure it has a `distance` variable somewhere in the geometry
* Run `python gen_distance_inputs.py` with appropriate arguments (`-h` for help) to generate inputs.



# Getting basis data

* First ensure that `gprint,basis` is enabled in input script, 
* Minimal calculation (hf) is sufficient
* Run and on output file use for example:

sed -n '/BASIS DATA/,/^ NUMBER OF VALENCE ORBITALS:/'p <outfile> > <datfile>


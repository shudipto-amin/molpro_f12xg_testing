#!/bin/bash

bases=("cc-pvtz" "aug-cc-pvtz" "awcvtz")
atoms=("n" "o" "f" "ne" "na" "mg" "s" "cl" "ar" "cu" "zn")

for atom in ${atoms[@]}; do
    for basis in ${bases[@]}; do
        inpfile=outputs/${atom}_${basis}.inp
        python gen_input.py -a $atom -b $basis -g 0.5 3 0.1 > $inpfile
    done
done


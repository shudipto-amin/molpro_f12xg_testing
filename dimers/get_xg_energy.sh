#!/bin/bash

sed -n '/Printing Energies step by step/,/F12-XG CALCULATIONS END/p' $1

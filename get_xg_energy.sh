#!/bin/bash

set -e

sed -n '/Printing Energies step by step/,/F12-XG CALCULATIONS END/p' $1

tail -n 1 $1

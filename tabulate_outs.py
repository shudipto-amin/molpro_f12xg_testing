import numpy as np
import matplotlib.pyplot as pp
import argparse as ap
import subprocess as sp
import sys
import pandas as pd
import os

parser = ap.ArgumentParser(
    description="""
    This script takes in multiple output files and prints out a
    table (pandas.DataFrame) of energies per output file. 

    RESTRICTIONS:
    * Output files MUST only be for a single point energy.
    * They should work with the `get_*_energy.sh`
    """,
        )

parser.add_argument('--outs', '-o', nargs='+',
                    help='default or standard molpro output files')
parser.add_argument('--xgouts', '-x', nargs='+',
                    help='XG molpro output files')

def ener_not_found_error(outfile):
    print(f"No energy found! Output file: {outfile}")

def get_ener(outfile, method='MP2-F12', out_type="std"):
    if out_type == "std":
        out = sp.check_output(f"./get_std_energy.sh {outfile}".split())        
    if out_type == "xg":
        out = sp.check_output(f"./get_xg_energy.sh {outfile}".split())

    out = out.decode('UTF-8')
    if 'Molpro calculation terminated' not in out:
        ener_not_found_error(outfile)
        return None

    lines = out.split('\n')
    for line in lines[-1::-1]:
        if f'!{method}' in line and 'total energy' in line:
            energy = float(line.split()[-1])
            return energy
    ener_not_found_error(outfile)
    return None
    
if __name__ == "__main__":
    args = parser.parse_args()
    #print(args)
    #print(sys.argv)
    data = dict(
        labels = [],
        energies = [],
    )
    def update(outfile, energy):
        data['energies'].append(energy)
        data['labels'].append(
                os.path.basename(outfile).rstrip('.out')
                )
    if len(sys.argv) <= 1:
        print("At least one output file must be provided\n")
        parser.print_help()
        sys.exit(1)
    if args.outs:
        for outfile in args.outs:
            energy = get_ener(outfile)
            update(outfile, energy)
    if args.xgouts:
        for outfile in args.xgouts:
            energy = get_ener(outfile, out_type="xg")
            update(outfile, energy)
    df = pd.DataFrame(data)
    print(df.to_string())

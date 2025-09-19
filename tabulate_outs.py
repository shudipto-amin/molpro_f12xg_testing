import numpy as np
import matplotlib.pyplot as pp
import argparse as ap
import subprocess as sp
import sys
import pandas as pd
import os
import xml_output_parser as xop 

parser = ap.ArgumentParser(
    description="""
    This script takes in multiple output files and prints out a
    table (pandas.DataFrame) of energies per output file. 

    RESTRICTIONS:
    * Output files MUST only be for a single point energy.
    * They should work with the `get_*_energy.sh`
    * For standard and default, must be XML output.
    """,
        )

parser.add_argument('--outs', '-o', nargs='+',
                    help='default or standard molpro XML output files')
parser.add_argument('--xgouts', '-x', nargs='+',
                    help='XG molpro output files')

def ener_not_found_error(outfile):
    print(f"No energy found! Output file: {outfile}")

def get_xg_energy_lines(outfile):
    start_marker = "Printing Energies step by step"
    end_marker = "F12-XG CALCULATIONS END"
    printing = False
    last_line = None
    output_lines = []
    with open(outfile, "r") as f:
        for line in f:
            last_line = line
            if start_marker in line:
                printing = True
            if printing:
                output_lines.append(line)
            if printing and end_marker in line:
                printing = False
    output_lines.append(last_line)
    return "".join(output_lines)


def get_ener(outfile, name='total energy', method='MP2-F12', out_type="std"):
    if out_type == "std":
        ener = xop.get_xmlener(outfile, name=name) 
        return ener            
    if out_type == "xg":
        out = get_xg_energy_lines(outfile)
        if 'Molpro calculation terminated' not in out:
            ener_not_found_error(outfile)
            return None

        lines = out.split('\n')
        for line in lines[-1::-1]:
            if f'{method}' in line and name in line:
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
        total_energies = [],
        correlation_energies = []
    )
    def update(outfile, total_energy, correlation_energy):
        data['total_energies'].append(total_energy)
        data['correlation_energies'].append(correlation_energy)
        data['labels'].append(
                os.path.basename(outfile).rstrip('.out')
                )
        
    if len(sys.argv) <= 1:
        print("At least one output file must be provided\n")
        parser.print_help()
        sys.exit(1)
    if args.outs:
        for outfile in args.outs:
            total_energy = get_ener(outfile, name='total energy')
            correlation_energy = get_ener(outfile, name='correlation energy')
            update(outfile, total_energy, correlation_energy)
    if args.xgouts:
        for outfile in args.xgouts:
            total_energy = get_ener(outfile, name='total energy', out_type='xg')
            correlation_energy = get_ener(
                outfile, name='correlation energy', out_type='xg')
            update(outfile, total_energy, correlation_energy)
    df = pd.DataFrame(data)
    print(df.to_string())

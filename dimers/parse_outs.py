import numpy as np
import matplotlib.pyplot as pp
import argparse as ap
import subprocess as sp

parser = ap.ArgumentParser()

parser.add_argument('--outs', '-o', nargs='+',
                    help='standard molpro output files')
parser.add_argument('--xgouts', '-x', nargs='+',
                    help='XG molpro output files')

def get_ener(outfile, out_type="std"):
    if out_type == "std":
        out = sp.check_output(f"./get_std_energy.sh {outfile}".split())        
    if out_type == "xg":
        out = sp.check_output(f"./get_xg_energy.sh {outfile}".split())
    print(out)
    return energies

if __name__ == "__main__":
    args = parser.parse_args()
    print(args.outs)
    print(args.xgouts)

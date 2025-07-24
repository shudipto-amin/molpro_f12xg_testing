import numpy as np
import matplotlib.pyplot as pp
import argparse as ap
import os
import sys

parser = ap.ArgumentParser()

parser.add_argument('outs', nargs='+',
                    help='molpro output files')

def get_ener(outfile):
    command = f"./get_energy.sh {outfile}"
                


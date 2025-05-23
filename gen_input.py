import os
import sys
import argparse

template = dict()

template['common'] = """
memory,200,m
gprint,orbitals

geometry={Ne}
basis={
  default,aug-cc-pVDZ
  set,mygem
     s, ne, 0.19532, 0.81920, 2.85917, 9.50073,35.69989,197.79328   
     c,1.6, 0.27070, 0.30552, 0.18297, 0.10986, 0.06810,  0.04224
}                             

"""

template['original'] = """
ANSATZ={ansatz}

{{df-hf}}
{{df-mp2}}
{{df-mp2-f12,gem_basis=mygem}}
"""

template['cpproute'] = """
ANSATZ={ansatz}

{{df-hf}}
{{df-mp2}}
{{df-mp2-f12,gem_basis=mygem,cpp_prog='DF-MP2-F12'}}
"""

template['xg'] = """

{df-hf}
{df-mp2}
{df-mp2-f12,gem_basis=mygem,cpp_prog='DF-MP2-XG'}
"""

outdir = "outputs/"

def generate_fname(temp_key, ansatz):
    if temp_key != 'xg':
        fname = f"test_{temp_key}_{ansatz}.inp"
    else:
        fname = f"test_xg.inp"

    for symb in ('(',')', ',', '*'):
        fname = fname.replace(symb,'-')
    fname = f"{outdir}/{fname}"
    return fname
    
def generate_input(temp_key, ansatz):
    return template['common'] + template[temp_key].format(ansatz=ansatz)         
    
ansatzes = [
    "3C(FIX)", "3*C(FIX)", "3C(FIX,HY1)"
]

if __name__ == "__main__":

    if not os.path.isdir(outdir):
        os.mkdir(outdir)
        
    for ansatz in ansatzes:
        for key in template:
            if key in ('common', 'xg'): continue
            print("="*50)
            fname = generate_fname(key, ansatz)
            input_script = generate_input(key, ansatz)

            with open(fname, 'w') as out:
                out.write(input_script)


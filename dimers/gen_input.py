import os
import sys
import argparse as ap

parser = ap.ArgumentParser(
        description = """
        This script takes an input template and generates xg, standard, and default scripts 
        from it.
        """,
        formatter_class=ap.ArgumentDefaultsHelpFormatter
        )

parser.add_argument(
        '-i', '--input', default='template.tinp',
        help = 'template input file',
        )
parser.add_argument(
        '-a', '--ansatz', default='3C(FIX,HY1,NOZ)',
        help = 'ansatz for both std and default'
        )
parser.add_argument(
        '-e', '--expfile', default='/home/linux3_i1/amin/f12xg_inputs/expfile_all_same.txt',
        help = 'expfile for xg'
        )
parser.add_argument(
        '-n', '--namexg', default = '',
        help = 'extra identifier for xg input files'
        )

parser.add_argument(
        '-o', '--outdir', default='outputs/',
        help='folder where generated input files will be saved.'
        )


def get_from_template(inpfile):

    template = dict()

    with open(inpfile, 'r') as inp:
        template['common'] = inp.read()

    template['default'] = """

    {{df-hf}}
    {{df-mp2}}
    {{df-mp2-f12,gem_basis=mygem,ANSATZ={ansatz}}}
    """

    template['standard'] = """

    {{df-hf}}
    {{df-mp2}}
    {{df-mp2-f12,gem_basis=mygem,cpp_prog='DF-MP2-F12',ANSATZ={ansatz}}}
    """

    template['xg'] = """

    {{df-hf}}
    {{df-mp2}}
    {{df-mp2-f12,gem_basis=mygem,cpp_prog='DF-MP2-XG',ANSATZ={expfile}}}
    """
    return template

def generate_fname(temp_key, args):
    inpfname = args.input.rstrip('.tinp')
    if temp_key != 'xg':
        fname = f"{inpfname}_{temp_key}_{args.ansatz}.inp"
    else:
        fname = f"{inpfname}_xg{args.namexg}.inp"

    for symb in ('(',')', ',', '*'):
        fname = fname.replace(symb,'-')
    fname = f"{args.outdir}/{fname}"
    return fname
    
def generate_input(template, temp_key, args):
    input_script = template['common'] + \
    template[temp_key].format(**args.__dict__)         
    if args.ansatz == 'default':
        input_script.replace(",ANSATZ=default", "")

    return input_script
        
if __name__ == "__main__":
    args = parser.parse_args() 
    print(args.__dict__)
    template = get_from_template(args.input)

        
    for key in template:
        if key in ('common'): continue
        print("="*50)
        fname = generate_fname(key, args)
        print(fname)
        input_script = generate_input(template, key, args)
        
        print(input_script)

    #sys.exit()
        with open(fname, 'w') as out:
            out.write(input_script)


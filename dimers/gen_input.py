import os
import sys
import argparse as ap
import textwrap

def existing_file(path):
    """Check that a path is an existing file, and return absolute path of it."""
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f"File not found: {path}")
    return os.path.abspath(path)  # also convert to absolute

def gamma_type(value, template):
    try:
        fval = float(value)
        return f"[{fval}]"
    except ValueError:
        pass

    if value in template["common"]:
        return value
    else:
        raise argparse.ArgumentTypeError(
            f"Invalid gamma '{value}'. Must be a float or one of: {template['common']}"
        )

parser = ap.ArgumentParser(
        description = """
        This script takes an input template and generates xg, standard, and default scripts 
        from it.
        """,
        formatter_class=ap.ArgumentDefaultsHelpFormatter
        )
parser.add_argument(
        '-i', '--input', 
        required=True,
        help = 'template input file',
        )

parser.add_argument(
        '-a', '--ansatz', default='3C(FIX,HY1,NOZ)',
        help = 'ansatz for both std and default'
        )
parser.add_argument(
        '-e', '--expfile', 
        default='/home/linux3_i1/amin/f12xg_inputs/expfile_all_same.txt',
        type=existing_file,
        help = 'expfile for xg'
        )
parser.add_argument(
        '-n', '--namexg', default = '',
        help = 'extra identifier for xg input files'
        )

parser.add_argument(
    "-g", "--gamma",
    default=1.3,   # will be passed to gamma_type too
    help="gamma value: [], or string that must exist in template['common']"
    )

parser.add_argument(
        '-o', '--outdir', default='outputs/',
        help='folder where generated input files will be saved.'
        )

parser.add_argument(
        '--test', 
        action="store_true",
        help="test the script and print out only"
        )

def get_from_template(inpfile):        
    template = dict()                             
    
    with open(inpfile, 'r') as inp:               
        template['common'] = inp.read()           
    
    template['default'] = textwrap.dedent("""\
        {{df-hf}}
        {{df-mp2}}
        {{df-mp2-f12,gem_beta={gamma},ANSATZ={ansatz}}}
    """)

    template['standard'] = textwrap.dedent("""\
        {{df-hf}}
        {{df-mp2}}
        {{df-mp2-f12,gem_beta={gamma},cpp_prog='DF-MP2-F12'}}
    """)

    template['xg'] = textwrap.dedent("""\
        {{df-hf}}
        {{df-mp2}}
        {{df-mp2-f12,cpp_prog='DF-MP2-XG',ANSATZ={expfile}}}
    """)
    
    return template

def gen_header(std_mem, temp_key):
    '''Generate header for input file using standard memory'''
    if temp_key == 'xg':
        mem = 10 * std_mem
    else:
        mem = std_mem
    return f"memory,{mem},m\n"

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
    input_script = gen_header(500, temp_key) + \
    template['common'] + \
    template[temp_key].format(**args.__dict__)         
    if args.ansatz == 'default':
        input_script.replace(",ANSATZ=default", "")

    return input_script
        

if __name__ == "__main__":
    
    args = parser.parse_args()
    template = get_from_template(args.input)
    
    args.gamma = gamma_type(args.gamma, template)


    for key in template:
        if key in ('common'): continue
        fname = generate_fname(key, args)
        input_script = generate_input(template, key, args)
        if args.test: 
            print("="*50)
            print(fname)
            print(input_script)
        else:    
            with open(fname, 'w') as out:
                out.write(input_script)


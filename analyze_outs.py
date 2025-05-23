import os
import sys
import argparse
import pandas
import xml.etree.ElementTree as ET
import gen_input as gi

import re

parser = argparse.ArgumentParser()
parser.parse_args()

def get_files_matching(rexpr):

    matches = []
    regex = re.compile(rexpr)

    for root, dirs, files in os.walk(gi.outdir):
        for file in files:
            #print(file)
            fname = f'{gi.outdir}/{file}'
            if regex.match(fname):
                matches.append(fname)
    return matches

def inspect_children(elem):
    for child in elem:
        print(child.tag, child.attrib)

def pretty_print(elem, indent_lvl=0):
    pre = indent_lvl * "   "
    print(pre, "="*50)
    print(pre, f"=={elem.tag.upper()} DETAILS==")

    for key, val in elem.attrib.items():
        print(pre, f"{key.upper():20s}: {val:20s}")
        
outfiles = []

for ansatz in gi.ansatzes:
    for key in ('original', 'cpproute'):
        infile = gi.generate_fname(key, ansatz)
        job_name = infile.rstrip('inp')
        rexpr = f'{job_name}*.xml'
        #print(rexpr)
        matching_files = sorted(get_files_matching(rexpr))
        #print( matching_files)
        outfiles.append(matching_files[0])

for out in outfiles:
    print(out)


tree = ET.parse(out)
root = tree.getroot()

# clean namespaces
for elem in tree.iter():
    elem.tag = elem.tag.split("}")[1]
    

print('--')
# There is some redundant nesting in our case, so just deal with 'job'
print(len(root))
job = root[0]

# Get to the elements with appropriate data
for jobstep in job.findall('jobstep'):
    if jobstep.get('command') != 'DF-MP2-F12': continue
    pretty_print(jobstep)
    for prop in jobstep.findall('property'):
        if 'energy' in prop.get('name'):
            pretty_print(prop, 1)
        

import xml.etree.ElementTree as ET
import re

def get_clean_tree(xmlfile):
    """
    Strips all tags of the url. 
    
    The molpro generated xml files have tag names with url appended to them,
    which makes it cumbersome to parse. This code simply strips them out
    using '}' as a delimiter.
    """
    try:
        tree = ET.parse(xmlfile)
    except ET.ParseError:
        raise ValueError("Must provide a valid XML file")
    except Exception as e:
        # Handles cases like file not found, permission error, etc.
        raise ValueError("Must provide a valid XML file") from e
    
    for elem in tree.iter():
        elem.tag = elem.tag.split("}")[1]
    return tree

def find_by_attrib(nodes, key, value):
    """
    Find all nodes that have key:value pair in their attrib

    Arguments:
        nodes : an iterable of xml elements
        key : dictionary key
        value : corresponding dict value, as a regex
    """
    results = []
    for node in nodes:
        condition = re.search(value, node.attrib[key])
        if condition:
            results.append(node)
        
    return results

def get_xmlener(xmlfile, name='total energy', verbose=False):
    """
    Get energy from a given xml file
    Can only have a single total energy, won't work on xml
    files containing energies from multiple runs (or jobs).
    """
    tree = get_clean_tree(xmlfile)
    root = tree.getroot()
    energy_elems = root.findall(f".//property/[@name='{name}']")
    try:
        mp2_node, = find_by_attrib(energy_elems, 'method', '^MP2-F12')
    except ValueError as e:
        print("ERROR READING ENERGY =============================")
        print(f"{xmlfile} does not have any '^MP2-F12' method")
        print("==================================================")
        raise(e)
    ener = mp2_node.attrib['value']
    method = mp2_node.attrib['method']
    name = mp2_node.attrib['name']
    if verbose:
        print(f"{method} {name}: {ener}")

    return float(ener)


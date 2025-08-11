import xml.etree.ElementTree as ET
import re

def get_clean_tree(xmlfile):
    """
    Strips all tags of the url. 
    
    The molpro generated xml files have tag names with url appended to them,
    which makes it cumbersome to parse. This code simply strips them out
    using '}' as a delimiter.
    """
    tree = ET.parse(xmlfile)
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

def get_xmlener(xmlfile, verbose=False):
    """
    Get energy from a given xml file
    """
    tree = get_clean_tree(xmlfile)
    root = tree.getroot()
    energy_elems = root.findall(".//property/[@name='total energy']")
    mp2_node, = find_by_attrib(energy_elems, 'method', '^MP2-F12')
    ener = mp2_node.attrib['value']
    method = mp2_node.attrib['method']
    name = mp2_node.attrib['name']
    if verbose:
        print(f"{method} {name}: {ener}")

    return float(ener)
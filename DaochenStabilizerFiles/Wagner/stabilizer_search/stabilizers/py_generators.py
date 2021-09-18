"""Module which finds unique generating sets for pauli Stabilier groups using
python code and the bitarray module."""


from itertools import combinations
from random import shuffle
from .utils import *

import numpy as np
import copy

import os.path as path
import yaml

import pickle


__all__ = ['get_positive_stabilizer_groups']

APP_DIR = path.abspath(__file__)
BIT_STRING = '{}_bitstrings.pkl'
INDEX_STRING = '{}_index.pkl'
#BIT_STRING = '{}_bitstrings.yml'
SUB_STRING = '{}_subspaces.pkl'
GEN_STRING = '{}_generators.pkl'
LOG_STRING = '{:d}_temp.yml'


def xnor(a,b):
    return (a&b)^(~a&~b)


def xor(a,b):
      return (a|b)&~(a&b)


class BinarySubspace(object):
    """Set-like class for bitarray objects to generate a closed subspace."""
    def __init__(self, *data):
        self.order = 0
        self._items = []
        self.generators = []
        for val in data:
            # if not isinstance(val, bitarray):
            if not isinstance(val, np.ndarray):
                raise ValueError('This class works for numpy arrays only!')
                # raise ValueError('This class works for bitarrays only!')
            self.add(val)
    
    def __contains__(self, it):
        for _el in self._items:
            # if all(xnor(_el, it)):
            if np.array_equal(_el, it):
                return True
        return False

    def __iter__(self):
        for item in self._items:
            yield item

    def _generate(self, obj):
        for item in self._items:
            # new = item^obj
            new = xor(item, obj)
            if new in self:
                continue
            else:
                self.order +=1
                self._items.append(new)
                self._generate(new)
        return

    def __eq__(self, other):
        return all([_el in other for _el in self._items])

    def add(self, obj):
        for _el in self._items:
            if all(xnor(obj, _el)):
                return self
        self.order +=1
        self.generators.append(obj)
        self._items.append(obj)
        self._generate(obj)
        return self


def symplectic_inner_product(n, a, b):
    x_a, z_a = a[:n], a[n:]
    x_b, z_b = b[:n], b[n:]
    # count = (x_a&z_b).count() + (x_b&z_a).count()
    count = np.sum((x_a&z_b)) + np.sum((x_b&z_a))
    return count%2


def test_commutivity(n, bits1, bits2):
    return symplectic_inner_product(n, bits1, bits2) == 0 #1 if they anticommute, 0 if they commute


def gen_bitstrings(n):
    bitstrings = []
    for i in range(1, pow(2,2*n)): #We ignore the all 0 string as it corresponds to I^{n}
        bin_string = bin(i)[2:] #strip the 0b from the string
        bin_string = '0'*(2*n - len(bin_string)) + bin_string
        a = np.array([b == '1' for b in bin_string])
        bitstrings.append(a)
    return bitstrings

#def gen_combinations_and_save(bstr, r, format_string):
#    combolist=[]
#    combolist = combinations(bstr, r)
#    with open(path.join('./', format_string.format(r)), 'wb') as f:
#        pickle.dump(items, f)
#    return combolist

def try_load_pickle(format_string, n):
    f_string = format_string.format(n)
    package_path = path.join(APP_DIR, 'data', f_string)
    rel_path = path.join('./', f_string)
    if path.exists(package_path):
        with open(package_path, 'r+b') as _f:
            items = pickle.load(_f)
            _f.close()
    elif path.exists(rel_path):
        with open(rel_path, 'r+b') as _f:
            items = pickle.load(_f)
            _f.close()
    else:
        items = None
    #if items is not None and n_states != n_stabilizer_states(n_qubits):
        #return sample(items, n_states)
    return items

def save_to_pickle(data, format_string, num):
    f_string = format_string.format(num)
    with open(path.join('./', f_string), 'w+b') as f:
        pickle.dump(data, f)
        f.close()
    return

def try_load_config(format_string, num):
    f_string = format_string.format(num)
    package_path = path.join(APP_DIR, 'data', f_string)
    rel_path = path.join('./', f_string)
    if path.exists(package_path):
        with open(package_path, 'r+') as _f:
            items = list(yaml.safe_load_all(_f))
    elif path.exists(rel_path):
        with open(rel_path, 'r+') as _f:
            items = list(yaml.safe_load_all(_f))
    else:
        items = None
    #if items is not None and n_states != n_stabilizer_states(n_qubits):
        #return sample(items, n_states)
    return items

def save_to_yaml(data, format_string, num):
    f_string = format_string.format(num)
    with open(path.join('./', f_string), 'w+') as f:
        yaml.dump_all(data, f)
    return

def get_positive_stabilizer_groups(n_qubits, n_states, j):
    if n_states == n_stabilizer_states(n_qubits): 
        # If generating all states, we want to focus on only the all
        # positive signed operators
        target = n_states/pow(2, n_qubits)
    else:
        #If generating less than all, we'll add signs in randomly to compenstate
        target = n_states
    
    bitstrings = try_load_pickle(BIT_STRING, n_qubits)
    if bitstrings is not None:
        print("Bitstrings loaded! len(bitstrings) = ", len(bitstrings))
    else:
        bitstrings = gen_bitstrings(n_qubits)
        shuffle(bitstrings)
        save_to_pickle(tuple(bitstrings), BIT_STRING, n_qubits)
    
    index = try_load_pickle(INDEX_STRING, n_qubits)
    if index is not None:
        print("Index loaded! index = ", index)
    else:
        index=0
    
    subspaces = try_load_pickle(SUB_STRING, n_qubits)
    if subspaces is not None:
        print("Subspaces loaded! len(subspaces) = ", len(subspaces))
    else:
        subspaces=[]
    
    generators = try_load_pickle(GEN_STRING, n_qubits)
    if generators is not None:
        print("Generators loaded! len(generators) = ", len(generators))
    else:
        generators=[]
    
    #itemdict={}
    #combolist=[]
               
    for group in combinations(bitstrings, n_qubits):
        if j < index:
            j+=1
            continue
        if((j+1) % 1000 == 0):
            print("combos considered = ", j+1) 
        groups = sorted(group, key=bool_to_int)
        if len(group) == 2:
            if not test_commutivity(n_qubits, group[0], group[1]):
                j+=1
                continue
        if len(group) > 2:
            if not all([test_commutivity(n_qubits, pair[0], pair[1]) 
                        for pair in combinations(group, 2)]):
                j+=1
                continue
        candidate = BinarySubspace(*group)
        if len(candidate.generators) < n_qubits:
            j = j+1
            continue
        if len(candidate._items) < pow(2,n_qubits):
            j = j+1
            continue
        res = tuple(i for i in sorted(candidate._items, key=bool_to_int))
        if np.any([np.all([np.allclose(_el1, _el2) for _el1, _el2 in zip(res, space)]) 
                   for space in subspaces]):
            j = j+1
            continue
        index = j
        subspaces.append(res)
        generators.append(tuple(candidate.generators))
        #itemdict["bitstrings"] = tuple(bitstrings)
        #itemdict["subspaces"] = tuple(subspaces)
        #itemdict["generators"] = tuple(generators)
        #generators = list(set(generators))
        save_to_pickle(generators, GEN_STRING, n_qubits)
        save_to_pickle(subspaces, SUB_STRING, n_qubits)
        save_to_pickle(index, INDEX_STRING, n_qubits)
        print(index)
        print("gen. found= ", len(generators))
        #save_to_yaml(itemdict, LOG_STRING, n_qubits)
        if len(generators) == target:
            break
    return generators

def get_stabilizer_groups(n_qubits, n_states):
    positive_groups = get_positive_stabilizer_groups(n_qubits, n_states, 0)
    extend = False
    if n_states == n_stabilizer_states(n_qubits):
        extend = True
        print("Found {} positive groups".format(len(positive_groups)))
    groups = [list(map(array_to_pauli, group)) for group in positive_groups]
    sign_strings = get_sign_strings(n_qubits, n_states)
    return add_sign_to_groups(groups, sign_strings, extend)
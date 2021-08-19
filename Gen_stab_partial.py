#!/usr/bin/env python
# coding: utf-8

from random import sample

from stabilizer_search.stabilizers.eigenstates import py_find_eigenstates
from stabilizer_search.stabilizers.py_generators import get_stabilizer_groups as py_get_groups
from stabilizer_search.stabilizers.utils import *

import os.path as path
import pickle


STATE_STRING = '{}_stabs_partial.pkl'
GROUP_STRING = '{}_groups_partial.pkl'
GEN_STRING = '{}_generators.pkl'


def try_load(format_string, n_qubits):
    f_string = format_string.format(n_qubits)
    rel_path = path.join('./', f_string)
    if path.exists(rel_path):
        with open(rel_path, 'rb') as _f:
            items = pickle.load(_f)
            _f.close()
    else:
        items = None

    return items


def save_to_pickle(items, format_string, n_qubits):
    with open(path.join('./', format_string.format(n_qubits)), 'wb') as f:
        pickle.dump(items, f)
        f.close()
    return


def get_stabilizer_groups(n_qubits):
    positive_groups = try_load(GEN_STRING, n_qubits) 
    #positive_groups = get_positive_stabilizer_groups(n_qubits, n_states, 0)
    #extend = False
    #if n_states == n_stabilizer_states(n_qubits):
    n_states = n_stabilizer_states(n_qubits)
    extend = True
    print("Found {} positive groups".format(len(positive_groups)))
    groups = [list(map(array_to_pauli, group)) for group in positive_groups]
    sign_strings = get_sign_strings(n_qubits, n_states)
    return add_sign_to_groups(groups, sign_strings, extend)


def get_stabilizer_states(n_qubits, real_only, n_states=None):
    """Method for returning a set of stabilizer states. It takes the following 
    arguments:
    Positional:
      n_qubits: The number of qubits our stabilizer states will be built out of.
      n_states (Optional): Number of stabilizer states we require. If not 
      specified, defaults to all Stabilier states.
    Keyword:
      use_cached: Boolean, defaults to True and looks in the package or working 
      dir for serialised states or generators.
      generator_backend: Function which searches for the stabilizer generators
      eigenstate_backend: Function which takes sets of stabilizer generators and
      builds the corresponding eigenstates.
      real_only: Return only real-valued stabilizer states
    """
    use_cached = True
    #generator_func = kwargs.pop('generator_backend', py_get_groups)
    #eigenstate_func = kwargs.pop('eigenstate_backend', py_find_eigenstates)
    #real_only = False
    stabilizer_states = None
    get_all = (n_states == n_stabilizer_states(n_qubits))
    if n_states is None:
        get_all = True
        n_states = n_stabilizer_states(n_qubits)
    if use_cached:
        stabilizer_states = try_load(STATE_STRING, n_qubits)
        if stabilizer_states is None:
            groups = try_load(GROUP_STRING, n_qubits)
            if groups is not None:
                if get_all:
                    save_to_pickle(groups, GROUP_STRING, n_qubits)
                stabilizer_states = py_find_eigenstates(groups, n_states, real_only)
    if stabilizer_states is None:
        #generators = generator_func(n_qubits, n_states)
        #generators = try_load(GEN_STRING, n_qubits)
        generators = get_stabilizer_groups(n_qubits) 
        stabilizer_states = py_find_eigenstates(generators, real_only)
        if use_cached and get_all:
            save_to_pickle(generators, GROUP_STRING, n_qubits)
            save_to_pickle(stabilizer_states, STATE_STRING, n_qubits)
    return stabilizer_states


#!/usr/bin/env python
# coding: utf-8

import numpy as np
import copy
from math import sqrt
from numpy.random import choice
import yaml
import time
import math

import os.path as path
import pickle


from stabilizer_search.search.brute_force import *
from stabilizer_search.mat import X, T
from stabilizer_search.mat import tensor


n_qubits = 4


H = [[np.cos(np.pi/8)],[np.sin(np.pi/8)]]
target_state = tensor(*([H]*n_qubits))
real = np.allclose(np.imag(target_state), 0.)


stabilizers = get_stabilizer_states(n_qubits, real_only=real)
shuffle(stabilizers)
n_stabilizers = len(stabilizers)


# In[ ]:





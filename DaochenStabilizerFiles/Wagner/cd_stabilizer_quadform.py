import random
import keras
import time
import numpy as np
import multiprocessing as mp
from copy import deepcopy
# import numba

#TODOs:
# 1. get numba to work
# 2. allow a zero component in the decomposition
# 3. understand what class of methods this method is in

from keras.optimizers import SGD, Adam
from keras.models import Sequential
from keras.layers import Dense

import utils_quadform as utils
from utils_quadform import bits_to_stab

# environment parameters
n_qubits = 4
chi = 4
H = [[np.cos(np.pi/8)],[np.sin(np.pi/8)]]
target = utils.tensor([H]*n_qubits)

# epsilon = 1
tol = 1.0e-7
real = True
n_cycles = 20

if real:
    m = 3/2*np.power(n_qubits,2)+3/2*n_qubits
else:
    m = 3/2*np.power(n_qubits,2)+5/2*n_qubits
assert m%1 == 0

n_bits = int(m*chi)
init_bits = np.zeros(n_bits,dtype=int)
# init_bits = np.array([np.random.randint(2) for i in range(n_bits)])
# print('\n n_qubits='+str(n_qubits)+'\n chi='+str(chi)+'\n n_bits='+str(n_bits))

def scorebits(bits):
    basis = bits_to_stab(bits,n_qubits,chi,real)
    projector = utils.orthogonal_projector(basis)
    score = np.linalg.norm(projector*target)
    return score

def worker(worker_index):
    print('worker', worker_index, 'is active \n') 

    bits = deepcopy(init_bits)
    print('initial score = ', scorebits(init_bits))

    for cycle in range(n_cycles):
        print('*' * 20)
        print('current cycle = ', cycle)
        print('*' * 20)
        counter = 0 
        for i in range(n_bits):
            bits_old = deepcopy(bits)
            score_old = scorebits(bits_old)
            bits_new = deepcopy(bits)
            bits_new[i] = int(1-bits_new[i])
            score_new = scorebits(bits_new)
            if score_new > score_old:
                bits = bits_new
                print('worker', str(worker_index), 'flipped on coordinate', int(i), 'new score', score_new)
                counter += 1
            # else:
                # print('worker', str(worker_index), 'did not flip on coordinate', int(i), 'score remains:', score_old)
        
        print('at cycle', cycle, 'score = ', scorebits(bits), '\n bits = ', repr(bits))
        if counter == 0:
            print('no more gains to be had from cd after cycle', cycle)
            break

if __name__ == '__main__':    
    pool = mp.Pool(mp.cpu_count())    
    pool.map(worker, range(1));
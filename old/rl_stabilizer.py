import networkx as nx #for various graph parameters, such as eigenvalues, macthing number, etc
import random
import numpy as np
import copy
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam
from keras.models import load_model
from statistics import mean
from math import sqrt
from numpy.random import choice
import pickle
import time
import math
import sympy
import matplotlib.pyplot as plt
import multiprocessing as mp
import utils as utils
from stabilizer_search.search.brute_force import *
from stabilizer_search.mat import X, Z, T
from stabilizer_search.mat import tensor
import randstab as rs

n_qubits = 6
chi = 3
MYN = int(chi)

H = [[np.cos(np.pi/8)],[np.sin(np.pi/8)]]
T_state = np.array([[1],[np.exp(1j*np.pi/4)]])/np.sqrt(2)
T_perp_state = Z*T_state
# target_state = tensor(*([H]*n_qubits))
target_state = (tensor(*([T_state]*n_qubits)) + tensor(*([T_perp_state]*n_qubits)))/np.sqrt(2)

is_target_state_real = all(np.isreal(target_state))
if is_target_state_real:
    print('Target state is real')
else:
    print('Target state is not real')

n_stabilizers_target = 10000
n_generations = 500

# Daochen: why should len_game have anything to do with MYN
len_game = MYN
observation_space = 2*MYN 
INF = 1000000
n_sessions = 500 #number of new sessions per iteration
# default 93, 94 respectively
percentile = 93 #top 100-X percentiled we are learning from
super_percentile = 98 #top 100-X percentile that survives to next iteration

def calcScore(state,stabilizers):
    """
    Calculates the reward for a given word. 
    This function is very slow, it can be massively sped up with numba -- but numba doesn't support networkx yet, which is very convenient to use here
    :param state: the first MYN letters of this param are the word that the neural network has constructed.

    :returns: the reward (a real number). Higher is better, the network will try to maximize this.
    """

    f = state[:MYN];
    candidate_stabilizer_basis = [np.array([stabilizers[f[i],:]]).transpose() for i in range(MYN)]
#     print(candidate_stabilizer_basis)
    projector = ortho_projector(candidate_stabilizer_basis)
    projection = np.linalg.norm(projector*target_state, 2)
#     projection = 1
    
    score = projection
    target = score
    
    if np.allclose(score, 1):
        print('You found a stabilizer decomposition with (n_qubits,chi) = ', [n_qubits,chi])
        print('The set of stabilizers is: ', f)
        return -1, -1
    return target, score

####No need to change anything below here. 
# Daochen: the agent argument will be the "model"
def generate_session(agent, n_sessions, stabilizers, verbose = 1):
    """
    Play n_session games using agent neural network.
    Terminate when games finish 

    Code inspired by https://github.com/yandexdataschool/Practical_RL/blob/master/week01_intro/deep_crossentropy_method.ipynb
    """
    n_stabilizers = len(stabilizers)
    states =  np.zeros([n_sessions, observation_space, len_game], dtype=int)
    actions = np.zeros([n_sessions, len_game], dtype = int)
    state_next = np.zeros([n_sessions,observation_space], dtype = int)
    prob = np.zeros(n_sessions)
    states[:,MYN,0] = 1
    step = 0
    total_target = np.zeros([n_sessions])
#     total_target = np.zeros([n_sessions], dtype=complex)
    total_score = np.zeros([n_sessions])
    recordsess_time = 0
    play_time = 0
    scorecalc_time = 0
    pred_time = 0
    while (True):
        step += 1
        tic = time.time()
        prob = agent.predict(states[:,:,step-1], batch_size = n_sessions) 
        pred_time += time.time()-tic

        for i in range(n_sessions):
            action = choice(n_stabilizers, p=prob[i])
            actions[i][step-1] = action
            tic = time.time()
            state_next[i] = states[i,:,step-1]
            play_time += time.time()-tic
            if (action > 0):
                state_next[i][step-1] = action
            state_next[i][MYN + step-1] = 0
            if (step < MYN):
                state_next[i][MYN + step] = 1
#                 Daochen: terminal equals whether step equals MYN: I suppose meaning that an entire state has been generated
            terminal = step == MYN
            tic = time.time()
            if terminal:
#                 print('state_next[i]', state_next[i])
                total_target[i], total_score[i] = calcScore(state_next[i],stabilizers)
                if total_target[i] == -1:
                    return -1
#                 print("total_score", total_score[i])
            scorecalc_time += time.time()-tic
            tic = time.time()
            if not terminal:
                states[i,:,step] = state_next[i]
            recordsess_time += time.time()-tic
        if terminal:
            break
    #If you want, print out how much time each step has taken. This is useful to find the bottleneck in the program.		
    if (verbose):
        print("Predict: "+str(pred_time)+", play: " + str(play_time) +", scorecalc: " + str(scorecalc_time) +", recordsess: " + str(recordsess_time))
    return states, actions, total_score, total_target

def select_elites(states_batch, actions_batch, rewards_batch, percentile=50):
    """
    Select states and actions from games that have rewards >= percentile
    :param states_batch: list of lists of states, states_batch[session_i][t]
    :param actions_batch: list of lists of actions, actions_batch[session_i][t]
    :param rewards_batch: list of rewards, rewards_batch[session_i]

    :returns: elite_states,elite_actions, both 1D lists of states and respective actions from elite sessions

    This function was mostly taken from https://github.com/yandexdataschool/Practical_RL/blob/master/week01_intro/deep_crossentropy_method.ipynb
    If this function is the bottleneck, it can easily be sped up using numba
    """
    counter = n_sessions * (100.0 - percentile) / 100.0
    reward_threshold = np.percentile(rewards_batch,percentile)

    elite_states = []
    elite_actions = []
    elite_rewards = []
    for i in range(len(states_batch)):
        if rewards_batch[i] >= reward_threshold-0.0000001:
            if (counter > 0) or (rewards_batch[i] >= reward_threshold+0.0000001):
                for item in states_batch[i]:
                    elite_states.append(item.tolist())
                for item in actions_batch[i]:
                    elite_actions.append(item)
            counter -= 1
    elite_states = np.array(elite_states, dtype = int)
    elite_actions = np.array(elite_actions, dtype = int)
    return elite_states, elite_actions

def select_super_sessions(states_batch, actions_batch, rewards_batch, targets_batch, percentile=90):
    """
    Select all the sessions that will survive to the next generation
    Similar to select_elites function
    If this function is the bottleneck, it can easily be sped up using numba
    """
    counter = n_sessions * (100.0 - percentile) / 100.0
    reward_threshold = np.percentile(rewards_batch,percentile)

    super_states = []
    super_actions = []
    super_rewards = []
    super_targets = []
    for i in range(len(states_batch)):
        if rewards_batch[i] >= reward_threshold-0.0000001:
            if (counter > 0) or (rewards_batch[i] >= reward_threshold+0.0000001):
                super_states.append(states_batch[i])
                super_actions.append(actions_batch[i])
                super_rewards.append(rewards_batch[i])
                super_targets.append(targets_batch[i])
                counter -= 1
    super_states = np.array(super_states, dtype = int)
    super_actions = np.array(super_actions, dtype = int)
    super_rewards = np.array(super_rewards)
    super_targets = np.array(super_targets)
    return super_states, super_actions, super_rewards, super_targets

def create_model(n_stabilizers):
    # LEARNING_RATE = 0.0001 #Increase this to make convergence faster, decrease if the algorithm gets stuck in local optima too often.
    LEARNING_RATE = 0.00001

    # These are hyperparameters
    # FIRST_LAYER_NEURONS = 128 #Number of neurons in the hidden layers.
    # SECOND_LAYER_NEURONS = 64
    # THIRD_LAYER_NEURONS = 32

    FIRST_LAYER_NEURONS = 16 #Number of neurons in the hidden layers.
    SECOND_LAYER_NEURONS = 8
    THIRD_LAYER_NEURONS = 4

    model = Sequential()
    model.add(Dense(FIRST_LAYER_NEURONS,  activation="relu"))
    model.add(Dense(SECOND_LAYER_NEURONS, activation="relu"))
    model.add(Dense(THIRD_LAYER_NEURONS, activation="relu"))
    model.add(Dense(n_stabilizers, activation="softmax"))
    model.build((None, observation_space))
    model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(learning_rate = LEARNING_RATE))

    print(model.summary())
    
    return model
    
def worker(worker_index):
    
    print('Worker:', worker_index, 'is active') 
    
    stabilizers = [rs.random_stabilizer_state(n_qubits) for i in range(n_stabilizers_target)]
    L = {array.tobytes(): array for array in stabilizers}
    unique_stabilizers = list(L.values()) # [array([1, 3, 2, 4]), array([1, 2, 3, 4])]
    if is_target_state_real:
        unique_real_stabilizers = list(filter(lambda x: all(np.isreal(x)), unique_stabilizers))
        stabilizers = np.array(unique_real_stabilizers)
    stabilizers = np.array(unique_stabilizers)
    n_stabilizers = len(stabilizers)
    print('number of considered stabilizer states =', n_stabilizers)
    
    model = create_model(n_stabilizers)
    
    super_states =  np.empty((0,len_game,observation_space), dtype = int)
    super_actions = np.array([], dtype = int)
    super_rewards = np.array([])
    super_targets= np.array([])
    sessgen_time = 0
    fit_time = 0
    score_time = 0

    myRand = random.randint(0,1000) #used in the filename
    
    for i in range(n_generations): #1000000 generations should be plenty
        #generate new sessions
        #performance can be improved with joblib
        tic = time.time()
    #     sessions = states, actions, total_score, total_target
        sessions = generate_session(model,n_sessions,stabilizers,0) #change 0 to 1 to print out how much time each step in generate_session takes 
        if sessions == -1:
            break
        sessgen_time = time.time()-tic
        tic = time.time()

        states_batch = np.array(sessions[0], dtype = int)
        actions_batch = np.array(sessions[1], dtype = int)
        rewards_batch = np.array(sessions[2])
        targets_batch = np.array(sessions[3])

        states_batch = np.transpose(states_batch,axes=[0,2,1])
        states_batch = np.append(states_batch,super_states,axis=0)

        if i>0:
            actions_batch = np.append(actions_batch,np.array(super_actions),axis=0)	

        rewards_batch = np.append(rewards_batch,super_rewards)
        targets_batch = np.append(targets_batch,super_targets)

        randomcomp_time = time.time()-tic 
        tic = time.time()

        elite_states, elite_actions = select_elites(states_batch, actions_batch, rewards_batch, percentile=percentile) #pick the sessions to learn from
        select1_time = time.time()-tic

        tic = time.time()
        super_sessions = select_super_sessions(states_batch, actions_batch, rewards_batch, targets_batch, percentile=super_percentile) #pick the sessions to survive
        select2_time = time.time()-tic

        tic = time.time()
        super_sessions = [(super_sessions[0][i], super_sessions[1][i], super_sessions[2][i], super_sessions[3][i]) for i in range(len(super_sessions[2]))]
        super_sessions.sort(key=lambda super_sessions: super_sessions[2],reverse=True)
        select3_time = time.time()-tic

        tic = time.time()
        model.fit(elite_states, elite_actions, verbose=0) #learn from the elite sessions
        fit_time = time.time()-tic

        tic = time.time()

        super_states = [super_sessions[i][0] for i in range(len(super_sessions))]
        super_actions = [super_sessions[i][1] for i in range(len(super_sessions))]
        super_rewards = [super_sessions[i][2] for i in range(len(super_sessions))]
        super_targets = [super_sessions[i][3] for i in range(len(super_sessions))]

    #     print(super_states)

        rewards_batch.sort()
    #     Daochen: why is it -100?
        mean_all_reward = np.mean(rewards_batch[-100:])
        mean_best_reward = np.mean(super_rewards)

        score_time = time.time()-tic

        print("\n" + str(i) +  ". Best individuals (reward): " + str(np.flip(np.sort(super_rewards))))

if __name__ == '__main__':
    print('number of all stabilizer states =', sum(rs.number_of_states(n_qubits)))
    print('number of target stabilizer states =', n_stabilizers_target)
    
    pool = mp.Pool(mp.cpu_count())    
    pool.map(worker, range(24));
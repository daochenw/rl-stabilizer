import random
import keras
import time
import numpy as np
import multiprocessing as mp
# import numba

#TODOs:
# 1. get numba to work
# 2. allow a zero component in the decomposition
# 3. understand what class of methods this method is in
# 4. see what the output is

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
epsilon = 0.1
# epsilon = 1
tol = 1.0e-7
real = True

# base_state = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0],dtype=int)

print('\n n_qubits='+str(n_qubits)+'\n chi='+str(chi)+'\n epsilon='+str(epsilon))

if real:
    m = 3/2*np.power(n_qubits,2)+3/2*n_qubits
else:
    m = 3/2*np.power(n_qubits,2)+5/2*n_qubits
assert m%1 == 0

m = int(m)
MYN = int(m*chi)
observation_space = 2*MYN 
len_game = MYN

# learning parameters
n_sessions = 1000
percentile = 93 #top 100-X percentiled we are learning from
super_percentile = 94 #top 100-X percentile that survives to next iteration
n_generations = 100000
INF = 100000

# neural network parameters 
learning_rate = 0.0001
# first_layer_neurons = int(MYN/2) #Number of neurons in the hidden layers.
# second_layer_neurons = int(MYN/4)
# third_layer_neurons = int(MYN/8)

first_layer_neurons = 128 #Number of neurons in the hidden layers.
second_layer_neurons = 64
third_layer_neurons = 4

def create_model():
    model = Sequential()
    model.add(Dense(first_layer_neurons,  activation="relu"))
    model.add(Dense(second_layer_neurons, activation="relu"))
    model.add(Dense(third_layer_neurons, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.build((None, observation_space))
    model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate = learning_rate)) #Adam optimizer also works well, with lower learning rate

    print(model.summary())
    
    return model

def calc_score(state):
    """
    Calculates the reward for a given word. 
    This function is very slow, it can be massively sped up with numba -- but numba doesn't support networkx yet, which is very convenient to use here
    :param state: the first MYN letters of this param are the word that the neural network has constructed.

    :returns: the reward (a real number). Higher is better, the network will try to maximize this.
    """
    x = state[:MYN]
    # x = np.mod(base_state + x,2)
    basis = bits_to_stab(x,n_qubits,chi,real)
    # print('basis=', basis)
    
    projector = utils.orthogonal_projector(basis)
    score = np.linalg.norm(projector*target) # note that this * is okay as working with matrix objects

    if np.allclose(score, 1, atol=tol):
        print('nice, worker has found a stabilizer decomposition with (n_qubits,chi) = ', [n_qubits,chi], 'and score =', score, '\n')
        return score, score
    
    return score, score

def generate_session(agent, n_sessions, verbose = 1):
    """
    Play n_session games using agent neural network.
    Terminate when games finish 

    Code inspired by https://github.com/yandexdataschool/Practical_RL/blob/master/week01_intro/deep_crossentropy_method.ipynb
    """
    states =  np.zeros([n_sessions, observation_space, len_game], dtype=int)
    actions = np.zeros([n_sessions, len_game], dtype = int)
    state_next = np.zeros([n_sessions,observation_space], dtype = int)
    prob = np.zeros(n_sessions)
    states[:,MYN,0] = 1
    step = 0
    total_target = np.zeros([n_sessions])
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
            # choose action 1 with probability prob[i]
            if np.random.rand() < 1-epsilon:
                if np.random.rand() < prob[i]:
                    action = 1
                else:
                    action = 0
            else:
                action = random.randint(0,1)

            # action = 1

            actions[i][step-1] = action
            tic = time.time()
            state_next[i] = states[i,:,step-1]
            play_time += time.time()-tic
            if (action > 0):
                state_next[i][step-1] = action
            state_next[i][MYN + step-1] = 0
            if (step < MYN):
                state_next[i][MYN + step] = 1
            terminal = step == MYN
            tic = time.time()
            if terminal:
                total_target[i], total_score[i] = calc_score(state_next[i])
                if np.isclose(total_score[i], 1):
                    return -1
            scorecalc_time += time.time()-tic
            tic = time.time()
            if not terminal:
                states[i,:,step] = state_next[i]
            recordsess_time += time.time()-tic
        if terminal:
            break
    #If you want, print out how much time each step has taken. This is useful to find the bottleneck in the program.		
    if (verbose):
        print("predict: "+str(pred_time)+", play: " + str(play_time) +", scorecalc: " + str(scorecalc_time) +", recordsess: " + str(recordsess_time))
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

def worker(worker_index):
    print('worker', worker_index, 'is active \n') 

    model = create_model()
    
    super_states =  np.empty((0,len_game,observation_space), dtype = int)
    super_actions = np.array([], dtype = int)
    super_rewards = np.array([])
    super_targets= np.array([])
    sessgen_time = 0
    fit_time = 0
    score_time = 0

    for i in range(n_generations):
        #generate new sessions
        #performance can be improved with joblib
        tic = time.time()
        sessions = generate_session(model, n_sessions, 0) #change 0 to 1 to print out how much time each step in generate_session takes 
        if sessions == -1:
            print('worker', worker_index, 'has found a stabilizer decomposition and is terminating \n') 
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

        rewards_batch.sort()
        score_time = time.time()-tic

        print(str(i) +  '. best individuals (reward) of worker', str(worker_index),': ' + str(np.flip(np.sort(super_rewards)))+'\n')

if __name__ == '__main__':    
    pool = mp.Pool(mp.cpu_count())    
    pool.map(worker, range(24));
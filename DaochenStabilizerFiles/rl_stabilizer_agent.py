import numpy as np
import kron_vec_product as kvp
import utils as utils
from utils import I, X, Y, Z, pauli, coefficient 
import random as rand
from collections import namedtuple
import multiprocessing as mp

# problem specification and initial conditions
n = 3 # n = number of qubits
chi = 3 # chi = number of stabilizers

H = [[np.cos(np.pi/8)],[np.sin(np.pi/8)]] 
target = utils.tensor(*([H]*n)) # target = magic state to decompose

Z0 = [1,0]
start_state = (utils.tensor(*([Z0]*n)),)*chi # start_state = state at which the computation starts making Pauli moves

tol = 1.0e-10 # tol = how close to the score of 1 is deemed a success

dflt_reward = -1
parameters = {'start_state': start_state,
              'dflt_reward': dflt_reward}

action_id = namedtuple('action_id', 'k p') # k is in range (chi) denoting the index of the state tuple to change, p is pauli update on that index
no_action = action_id(0, [0]*(n+1))

# basic functions
def update(state, action, real = True):
    if real and action.p[1:].count(2) % 2 != 0: # p[1:] corresponds to the number of Y Paulis
        return 0
    new_state = list(state)
    pauli_string = [pauli[i] for i in action.p[1:]]
    pauli_coefficient = coefficient[action.p[0]]
    new_stabilizer = state[action.k] + kvp.kron_vec_prod(pauli_string, state[action.k])
    if np.linalg.norm(new_stabilizer) == 0:
        return 0
    assert np.linalg.norm(new_stabilizer) != 0
    new_stabilizer = new_stabilizer/np.linalg.norm(new_stabilizer)
    new_state[action.k] = new_stabilizer
    return tuple(new_state)

def score(state):
    projector = utils.orthogonal_projector(state)
    score = np.linalg.norm(projector*target)
    return score

# define stabilizer class
class Stabilizer:
    """Stabilizer domain for RL.
    
    """

    def __init__(self, parameters):
        for key in parameters:
            setattr(self, key, parameters[key])
        self.record_list = []
        self.position = [self.start_state]
        self.log_dict = {}
        self.reward_sum = 0 # should reward_sum be in the environment or in the agent, or both?

    def newstate(self, state, action):
        """Computes the newstate.

        Takes a state and an action from the agent and computes its next position.

        Args:
            state: a tuple of chi stabilizer vectors representing the current state.
            action: an action_id tuple.

        Returns:
            newstate: a tuple of chi stabilizer vectors representing the new state

        """
        
        newstate = update(state, action)
        
        if newstate == 0:
            self.position.append(state)
            return state
        
        self.position.append(newstate)

        return newstate

    def reward(self, state):
        """Computes the reward signal for a given state and updates total reward.

        Args:
            state: a tuple of chi stabilizer np.arrays representing the current state.

        Returns:
            reward: a scalar value. -100 for a cliff state, -1 otherwise.

        """

        reward = self.dflt_reward
        self.reward_sum += reward

        return reward

    def is_terminal(self, state):
        """Checks if state is terminal, i.e., state gives a stabilizer decomposition of the target

        Args:
            state: a list of chi stabilizer np.arrays representing the current state.

        Returns:
            True if state is terminal, False otherwise.

        """
        
        if abs(score(state) - 1) < tol:
#             the information can be retrieved by domain.position[-1]
            print('\n nice, agent found decomposition of target state on', n, 'qubits using', chi, 'stabilizers with score =', score(state))
            return True
        else:
            return False

# define agent classes
class RandomAgent:
    """RandomAgent chooses actions at random and does not learn.

    We write this mainly for testing and understanding the interface between
    agent and domain.
    
    """

    def __init__(self):
        self.reward_sum = 0

    def act(self, state):
        """Take state and return action taken at that state."""
        action = action_id(rand.randrange(chi), [rand.randrange(4) for i in range(n+1)])
        return action

    def learn(self, state, action, newstate, reward):
        """RandomAgent does not learn."""
        self.reward_sum += reward
        pass

class RandomWalkAgent:
    """RandomWalkAgent chooses actions following [BSS15 (arxiv.org/abs/1506.01396), Appendix B]
    
    We write this to serve as a baseline and to reproduce existing results.
    """
    # self, worker_index = 1, betai = 1, betaf = 4000, M = 1000, max_annealing = 100
    def __init__(self, worker_index = 1, betai = 1, betaf = 6000, M = 1000, max_annealing = 150):
        self.worker_index = worker_index
        self.betai = betai
        self.betaf = betaf
        self.M = M
        self.max_annealing = max_annealing
        self.beta = betai
        self.geometric_factor = np.power((betaf/betai),(1/max_annealing))
        self.reward_sum = 0
        self.counter = 0
        self.annealing_counter = 0

    def act(self, state):
        """Take state and return action taken at that state."""
        
        self.counter += 1
        
        if self.counter % self.M == 0 :
            print('\033[1m') # turn on bold
            print('status of worker:', self.worker_index)
            print('\033[0m') # turn off bold
            print('- annealing step =', self.annealing_counter)
            print('- current beta =', self.beta)
            print('- current score =', score(state))
            self.beta = self.beta * self.geometric_factor
            self.annealing_counter += 1
            if self.beta > self.betaf:
                return 0
        
        proposed_action = action_id(rand.randrange(chi), [rand.randrange(4) for i in range(n+1)])
        proposed_state = update(state,proposed_action)
        
        if proposed_state == 0:
            return no_action
        
        current_state_score = score(state)
        proposed_state_score = score(proposed_state)
        acceptance_probability = np.exp(-self.beta*(current_state_score-proposed_state_score))
        
        if rand.random() < acceptance_probability:
            return proposed_action
        else:
            return no_action

    def learn(self, state, action, newstate, reward):
        """RandomWalkAgent does not learn."""
        self.reward_sum += reward
        pass

# run agent on domain
def run_episode(domain, agent):
    state = domain.start_state
    while not domain.is_terminal(state):
        action = agent.act(state)
        if action == 0:        
            print('\n sorry, agent terminated without finding a decomposition')
            break
        newstate = domain.newstate(state, action)   #: take the action and compute the changed state.
        reward = domain.reward(newstate)            #: compute reward (note this is a function of s')
        agent.learn(state, action, newstate, reward)#: learn.
        state = newstate                            #: newstate becomes the current state for next iteration.

def worker(worker_index):
    random_agent = RandomWalkAgent(worker_index = worker_index)    #: instantiate an agent object
    domain = Stabilizer(parameters) #: instantiate the stabilizer world
    run_episode(domain, random_agent)

# run in parallel
if __name__ == '__main__':
    pool = mp.Pool(mp.cpu_count()) 
    print('parallelizing job over', mp.cpu_count(), 'CPUs')
    pool.map(worker, range(40));
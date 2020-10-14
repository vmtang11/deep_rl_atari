import numpy as np

class ReplayBuffer(object):
    '''
        Codes the concept of Experience Replay from the paper. Stores state, new state, action and reward for previous n steps. 32 experiences randomly sampled and the behavioral Q-network
        learns via SGD. The experiences are picked according to uniform probability and are shuffled to eliminate correlation.
    '''
    def __init__(self, max_size, input_shape, n_actions):
        '''
        INPUT: max_size: maximum memory size
               input_shape: size of input
               n_actions: number of actions to take
               
        INITIALIZES: mem_cntr: counts the number of memories stored
                     state_memory: memory for current state
                     new_state_memory: memory for next state
                     action_memory: memory of actions
                     reward_memory: memory for rewards received
                     terminal_memory: memory for terminal states
        '''
        self.mem_size = max_size  # user specified - paper uses a million or something but this require TB of RAM. 50,000 ~ 17 GB
        self.mem_cntr = 0  # memory counter is added everytime a memory is stored
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype = np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype = np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype = np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype = np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype = np.bool)
        
    def store_transition(self, state, action, reward, state_, done):
        '''
        Stores memories for each state, action, reward, new state, and terminal state. This ensures memories are not being overwritten and keeps track of how many memories are being stored.
        
        INPUT: state: current state
               action: action taken
               reward: reward received
               state_: next state
               done: done flag for episode
        '''
        # store memories in position of first unoccupied memory
        # find location of unoccupied memory
        index = self.mem_cntr % self.mem_size
        # replace all memory values at index
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done
        # increment memory counter by one because we've just stored a new memory
        self.mem_cntr += 1
        
    def sample_buffer(self, batch_size):
        '''
        Uniformly samples memories of size max_mem from all state, action, reward, new state, and terminal memories without replacement. These are then used in the DQNAgent for learning.

        INPUT: batch_size: size of batch for GD
        
        OUTPUT: states: batch of current state memories for learning in DQNAgent
                actions: batch of actions memories for learning in DQNAgent
                rewards: batch of rewards memories for learning in DQNAgent
                states_: batch of new state memories for learning in DQNAgent
                dones: batch of terminal memories for learning in DQNAgent
        '''
        # position of last stored memory
        max_mem = min(self.mem_cntr, self.mem_size)
        # uniformly sample memory, without replacement
        batch = np.random.choice(max_mem, batch_size, replace = False)
        
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]
        
        return states, actions, rewards, states_, dones

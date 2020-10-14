import numpy as np

class ReplayBuffer(object):
    '''
        Codes the concept of Experience Replay from the paper. Stores state, new state, action and reward for previous n steps. 32 experiences randomly sampled and the behavioral Q-network
        learns via SGD. The experiences are picked according to uniform probability and are shuffled to eliminate correlation.
    '''
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size  # user specified - paper uses a million or something but this require TB of RAM. 50,000 ~ 17 GB
        self.mem_cntr = 0  # memory counter is added everytime a memory is stored
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype = np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype = np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype = np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype = np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype = np.bool)
        
    def store_transition(self, state, action, reward, state_, done):
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

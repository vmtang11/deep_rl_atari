import numpy as np

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype = np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype = np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype = np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype = np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype = np.uint8)
        
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
        # uniformly sample memory, without repeats
        batch = np.random.choise(max_mem, batch_size, replace = False)
        
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]
        
        return states, actions, rewards, states_, dones
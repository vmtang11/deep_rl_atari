import numpy as np
import torch as T
from deep_q_network import DeepQNetwork
from replay_memory import ReplayBuffer

class DQNAgent(object):
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims, mem_size, batch_size, 
                 eps_min = .01, eps_dec = 5e-7, replace = 1000, algo = None, 
                 env_name = None, chkpt_dir = 'tmp/dqn'):  # replace refers to the number of steps after which the target network weights are flipped with that of the behavioral network
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0  # number of times we've called the learn function below. This will be used to update the target Q with the behavioral Q
        
        # Agent's memory
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)
        
        # q_eval: current state and action values
        self.q_eval = DeepQNetwork(self.lr, self.n_actions, input_dims = self.input_dims, 
                                   name = self.env_name + '_' + self.algo + '_q_eval', 
                                   chkpt_dir = self.chkpt_dir)
        
        # q_next: state action values for resulting states. We don't do backpropagation or SGD with the next network. Does not learn
        self.q_next = DeepQNetwork(self.lr, self.n_actions, input_dims = self.input_dims, 
                                   name = self.env_name + '_' + self.algo + '_q_next', 
                                   chkpt_dir = self.chkpt_dir)
        
    def choose_action(self, observation):
        '''
            epsilon greedy
        '''
        if np.random.random() > self.epsilon:
            state = T.tensor([observation], dtype = T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        
        return action
    
    def store_transition(self, state, action, reward, state_, done):
        '''
            Calls the store_transition() from replay_memory.py
        '''
        self.memory.store_transition(state, action, reward, state_, done)
        
    def sample_memory(self):
        '''
            Sample from history
        '''
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        
        states = T.tensor(state).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        
        return states, actions, rewards, states_, dones
    
    def replace_target_network(self):
        '''
            Swap weight of the target network with the behavioral network (set them as equal)
        '''
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())
            
    def decrement_epsilon(self):
        '''
            Linear epsilon decrement
        '''
        if self.epsilon > self.eps_min:
            self.epsilon = self.epsilon - self.eps_dec 
        else:
            self.eps_min
            
    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()
        
    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()
        
    def learn(self):
        '''
            We only wait for a single batch of 32 to fill up to start learning. We could wait for the entire replay_memory to
            fill uop before we start learning but this can be time consuming
        '''
        if self.memory.mem_cntr < self.batch_size:  # check if batch is full
            return
        
        self.q_eval.optimizer.zero_grad()  # zero gradients on the optimizer
        
        self.replace_target_network()  # check to see if target network needs weight replacing before learning. This is where we enforce Q* = Q every 1000 steps
        
        states, actions, rewards, states_, dones = self.sample_memory()

        indices = np.arange(self.batch_size)
        q_pred = self.q_eval.forward(states)[indices, actions]  # get action the agent actually took
        
        q_next = self.q_next.forward(states_).max(dim = 1)[0]  # want to move towards max of next state
        
        q_next[dones] = 0.0  # set q_next to zero when in terminal state.

        q_target = rewards + self.gamma * q_next  # if not terminal, then calculate th immediate reward plus the discounted max q-values according to the greedy target q network
        
        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)  # MSE loss function as defined in the paper.
        loss.backward()  # back propagate
        self.q_eval.optimizer.step()  # optimizer
        self.learn_step_counter += 1
        
        self.decrement_epsilon()  # decrement epsilon

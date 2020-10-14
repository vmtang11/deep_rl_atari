import numpy as np
import torch as T
from deep_q_network import DeepQNetwork
from replay_memory import ReplayBuffer

class DQNAgent(object):
    '''
    Creates the Deep Q Network Agent.
    '''
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims, mem_size, batch_size, 
                 eps_min = .01, eps_dec = 5e-7, replace = 1000, algo = None, 
                 env_name = None, chkpt_dir = 'tmp/dqn'):  
        '''
        INPUT: gamma: discount factor for the action value function 
               epsilon: probability for exploitation
               lr: learning rate of optimizer
               n_actions: number of actions possible
               input_dims: size of input
               mem_size: size of memory
               batch_size: size of batch for GD
               eps_min: minimum possible value for epsilon
               eps_dec: amount to decrement epsilon by
               replace: number of steps after which the target network weights are flipped with that of the behavioral network
               algo: name of algorithm
               env_name: name of environment
               chkpt_dir: name of directory to store checkpoints
        '''
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
        Selects actions based on epsilon greedy method.
        
        INPUT: observation: current state 
        OUTPUT: action: action to take (max action if exploit, random action if explore)
        '''
        if np.random.random() > self.epsilon:
            # exploit
            state = T.tensor([observation], dtype = T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            # explore
            action = np.random.choice(self.action_space)
        
        return action
    
    def store_transition(self, state, action, reward, state_, done):
        '''
        Stores memories for current state, action taken, reward received, next state, and done flag. (replay_memory.py).
        
        INPUT: state: current state
               action: action taken
               reward: reward received
               state_: next state
               done: done flag
        '''
        self.memory.store_transition(state, action, reward, state_, done)
        
    def sample_memory(self):
        '''
        Samples memories from stored memories.
        
        OUTPUT: states: memory of states
                actions: memory of actions taken
                rewards: memory of rewards received
                states_: memory of next states
                dones: memory of done flags
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
        Swap weight of the target network with the behavioral network (set them as equal).
        '''
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())
            
    def decrement_epsilon(self):
        '''
        Linearly decrement epsilon to eps_min.
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
        Formulates the DQN Agent's learning process. First, it only learns once the batch size is full (has 32 stored memories) to help save time rather than waiting for the entire 
        ReplayBuffer to fill. Once the batch size is full:
            1. Zero optimizer gradients and replace target network (if necessary) 
            2. Sample memories for states, actions, rewards, next states, and done flags
            3. Forward propagate states (q_pred) to get actions taken
            4. Forward propagate next states and find maximal action (q_next)
            5. Reset done flags in q_next (next states) to 0 when in terminal state
            6. Calculate target value (maximum possible action that could have been taken)
            7. Calculate MSE loss: difference between possible max action and action taken
            8. Backpropagate loss, step optimizer, increment step counter, and decrement epsilon
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

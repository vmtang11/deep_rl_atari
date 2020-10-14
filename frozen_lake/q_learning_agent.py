import numpy as np

class Agent():
    def __init__(self, lr, gamma, n_actions, n_states, eps_start, eps_end, eps_dec):
        self.lr = lr
        self.gamma = gamma
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = eps_start
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        
        self.Q = {}
        
        self.init_Q()
        
    def init_Q(self):
        for state in range(self.n_states):
            for action in range(self.n_actions):
                self.Q[(state, action)] = 0.0
                
    def choose_action(self, state):
        # epsilon greedy: if num is less than epsilon then randomly choose action
        if np.random.random() < self.epsilon:
            # randomly choose an action
            action = np.random.choice([i for i in range(self.n_actions)])
            
        else:
            # list of elements corresponding to action values for given state by looking in Q table
            actions = np.array([self.Q[(state, a)] for a in range(self.n_actions)])
            # index of maximal action
            action = np.argmax(actions)
            
        return action
    
    def decrement_epsilon(self):
        # linear
        if self.epsilon > self.eps_min:
            self.epsilon = self.epsilon * self.eps_dec 
        else:
            self.eps_min
            
    def learn(self, state, action, reward, state_):
        actions = np.array([self.Q[(state_, a)] for a in range(self.n_actions)])
        a_max = np.argmax(actions)
        # underscores denote next state/action
        
        self.Q[(state, action)] += self.lr * (reward + self.gamma * self.Q[(state_, a_max)] - self.Q[(state, action)])
        
        self.decrement_epsilon()
    
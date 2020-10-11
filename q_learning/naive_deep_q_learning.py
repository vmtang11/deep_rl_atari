import gym
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T
from util import plot_learning_curve

class LinearDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, input_dims):
        super(LinearDeepQNetwork, self).__init__()
        
        self.fc1 = nn.Linear(*input_dims, 128)
        self.fc2 = nn.Linear(128, n_actions)
        # calc estimate for Q (state action value func)
        # pass in state of env, get out value for each action of that state
        
        self.optimizer = optim.Adam(self.parameters(), lr = lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, state):
        layer1 = F.relu(self.fc1(state))
        actions = self.fc2(layer1)
        
        return actions
    
class Agent():
    def __init__(self, input_dims, n_actions, lr, gamma = 0.99, epsilon = 1.0, eps_dec = 1e-5, eps_min = 0.01):
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.action_space = [i for i in range(self.n_actions)]
        # makes it easier to choose action
        
        self.Q = LinearDeepQNetwork(self.lr, self.n_actions, self.input_dims)
        
    def choose_action(self, observation):
        # epsilon greedy
        if np.random.random() > self.epsilon:
            state = T.tensor(observation, dtype = T.float).to(self.Q.device)
            actions = self.Q.forward(state)       # actions for a given state
            action = T.argmax(actions).item()     # maximum of actions, need .item to get numpy array
        else:
            action = np.random.choice(self.action_space)
            
        return action
    
    def decrement_epsilon(self):
        if self.epsilon > self.eps_min:
            self.epsilon = self.epsilon - self.eps_dec
        else:
            self.eps_min
            
    def learn(self, state, action, reward, state_):
        self.Q.optimizer.zero_grad()
        
        # numpy arrays to pytorch tensors
        states = T.tensor(state, dtype = T.float).to(self.Q.device)
        actions = T.tensor(action).to(self.Q.device)
        rewards = T.tensor(reward).to(self.Q.device)
        states_ = T.tensor(state_, dtype = T.float).to(self.Q.device)
        
        # feed forward for update equation for Q estimate for current state
        # predicted q values for current state of env, get actions from that 
        q_pred = self.Q.forward(states)[actions]
        
        # value of maximal action in next state
        q_next = self.Q.forward(states_).max()
        
        # direction we want to move in
        q_target = reward + self.gamma * q_next
        
        # loss: difference between action taken and possible maximal action 
        loss = self.Q.loss(q_target, q_pred).to(self.Q.device)
        
        # backprop, step optimizer, decrement epsilon
        loss.backward()
        self.Q.optimizer.step()
        self.decrement_epsilon()
        
if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    n_games = 10000
    scores = []
    eps_history = []
    
    agent = Agent(lr = .0001, input_dims = env.observation_space.shape, n_actions = env.action_space.n)
    
    for i in range(n_games):
        score = 0
        done = False
        obs = env.reset()
        
        while not done:
            action = agent.choose_action(obs)                 # choose action from current state
            obs_, reward, done, info = env.step(action)       # new state, reward, done, debug info from env after action 
            score += reward                                   # increment score by reward
            agent.learn(obs, action, reward, obs_)            # learn from state, action, reward, new state
            obs = obs_                                        # set old state to new state, choose next action based on new state
        scores.append(score)
        eps_history.append(agent.epsilon)
        
        if i % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print('episode ', i, 'score %.1f avg score %.1f epsilon %.2f' % (score, avg_score, agent.epsilon))
            
    filename = 'naive_deep_q_learning.png'
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, scores, eps_history, filename)
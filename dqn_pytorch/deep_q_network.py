import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepQNetwork(nn.Module):
    '''
    Creates the deep q network and saves/loads checkpoints.
    '''
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        '''
        Initializes the Deep Q Network with 3 convolutional layers and 2 fully connected layers.
        
        INPUTS: lr: learning rate
                n_actions: number of actions that can be taken
                name: name for checkpoint file
                input_dims: size of input
                chkpt_dir: directory to put checkpoints
        '''
        super(DeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        
        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride = 4)  # 32 filters, 8X8 kernel and stride = 4 pixels
                                                                  # input_dims[0] = 4
        self.conv2 = nn.Conv2d(32, 64, 4, stride = 2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride = 1)
        
        fc_input_dims = self.calculate_conv_output_dims(input_dims)  # function to calculate input_dims to fc layer
        
        self.fc1 = nn.Linear(fc_input_dims, 512)
        self.fc2 = nn.Linear(512, n_actions)
        
        self.optimizer = optim.RMSprop(self.parameters(), lr = lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def calculate_conv_output_dims(self, input_dims):
        '''
        Calculates output dimensions from convolutional layers so that any input size can be used.
        
        INPUT: input_dims: size of input
        OUTPUT: dimensions of output from convolutional layers
        '''
        state = T.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))  # returns dim of output conv layer. This is to generalise to any input feature size and not hard code
    
    def forward(self, state):
        '''
        Forward propagates current state through DQN. 
        
        INPUT: state: current state
        OUTPUT: actions: actions resulting from current state 
        '''
        conv1 = F.relu(self.conv1(state))  # pass state through first conv layer and activate using ReLU
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        # conv3 shape is BS x n_filters x H x W (height and width of final convolved img)
        # need to reshape for fc layers to batch size and flatten
        conv_state = conv3.view(conv3.size()[0], -1)  # equivalent to np.reshape(). The -1 tells the func to flatten all other dim
        flat1 = F.relu(self.fc1(conv_state))
        actions = self.fc2(flat1)
        
        return actions
    
    def save_checkpoint(self):
        '''
        Saves checkpoint.
        '''
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)
        
    def load_checkpoint(self):
        '''
        Loads checkpoint.
        '''
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

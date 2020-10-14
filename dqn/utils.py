import collections
import cv2
import matplotlib.pyplot as plt
import numpy as np
import gym

def plot_learning_curve(x, scores, epsilons, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-100):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    plt.savefig(filename)
    
class RepeatActionAndMaxFrame(gym.Wrapper):
    '''
        This class repeats actions and returns the max of the latest 2 frames.
        Action repeating is don as it is cheaper to repeat actions and continue playing the game rather than have the agent choose a new action
        Maxing the last two frames is necessary as certain objects only occur in even/odd frames
    '''
    def __init__(self, env = None, repeat = 4, clip_rewards = False, no_ops = 0, fire_first = False):  # paper repeats agents actions for 4 steps. Repeating action and playing game cheaper than choosing action
        super(RepeatActionAndMaxFrame, self).__init__(env)
        self.repeat = repeat
        self.shape = env.observation_space.low.shape        # lower bounds of observation space. It is an array of 4 values and is the negative of the high.
        self.frame_buffer = np.zeros_like((2, self.shape))  # returns 2 arrays with zeros in the shape of 'shape'
        self.clip_rewards = clip_rewards
        self.no_ops = no_ops
        self.fire_first = fire_first
        
    def step(self, action):
        t_reward = 0.0                           # total reward at each step
        done = False
        for i in range(self.repeat):             # num frames want to iterate
            obs, reward, done, info = self.env.step(action)
            if self.clip_rewards:
                reward = np.clip(np.array([reward]), -1, 1)[0]
            t_reward += reward
            idx = i % 2                          # idx is 0 or 1 for repeat in [0, 1, 2, 3]
            self.frame_buffer[idx] = obs         # save current frame in buffer in even or odd position
            if done:
                break

        max_frame = np.maximum(self.frame_buffer[0], self.frame_buffer[1])  # find max of 2 frames in frame buffer. This is to avoid flickering as some objects only appear in even/odd frames
        
        return max_frame, t_reward, done, info
    
    def reset(self):
        obs = self.env.reset()
        no_ops = np.random.randint(self.no_ops) + 1 if self.no_ops > 0 else 0
        for _ in range(no_ops):
            _, _, done, _ = self.env.step(0)
            if done:
                self.env.reset()
        if self.fire_first:
            assert self.env.unwrapped.get_action_meanings()[1] == 'FIRE'        # action meanings for first action are fire
            obs, _, _, _ = self.env.step(1)
            
        self.frame_buffer = np.zeros_like((2, self.shape))
        self.frame_buffer[0] = obs
        
        return obs
    
class PreprocessFrame(gym.ObservationWrapper):
    '''
        This class preprocess the frame before feeding it into the neural network. The paper used images of 84X84X4.
        The 4 comes from the stacking and this happens in the StackFrames class.
        We also rescale the frame and do necessary array reshaping to make it ready for pyTorch nn.
    '''
    def __init__(self, shape, env = None):
        super(PreprocessFrame, self).__init__(env)
        self.shape = (shape[2], shape[0], shape[1])    # openAI returns channels last but pytorch wants channels first
        self.observation_space = gym.spaces.Box(low = 0.0, high = 1.0, shape = self.shape, dtype = np.float32)
        
    def observation(self, obs):
        new_frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)                                       # convert to grayscale
        resized_screen = cv2.resize(new_frame, self.shape[1:], interpolation = cv2.INTER_AREA)  # resize image
        new_obs = np.array(resized_screen, dtype = np.uint8).reshape(self.shape)
        new_obs = new_obs / 255.0                                                               # Rescaling
        
        return new_obs
    
class StackFrames(gym.ObservationWrapper):
    '''
        Stacks the last four image son top of each other to feed into neural network. Paper does not say explicitly why
        they do this. Could be to induce fluidity and reduce flickering.
    '''
    def __init__(self, env, repeat):
        super(StackFrames, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            env.observation_space.low.repeat(repeat, axis = 0), 
            env.observation_space.high.repeat(repeat, axis = 0),
            dtype = np.float32)
        self.stack = collections.deque(maxlen = repeat)
        
    def reset(self):
        self.stack.clear()
        observation = self.env.reset()
        for _ in range(self.stack.maxlen):
            self.stack.append(observation)
        
        return np.array(self.stack).reshape(self.observation_space.low.shape)
    
    def observation(self, observation):
        self.stack.append(observation)
        return np.array(self.stack).reshape(self.observation_space.low.shape)  # returns stacked images in form of np array
    
def make_env(env_name, shape = (84,84,1), repeat = 4, clip_rewards = False, no_ops = 0, fire_first = False):
    '''
        Combines all mods we made during preprocessing. The params provided are identical to the paper
    '''
    env = gym.make(env_name)
    env = RepeatActionAndMaxFrame(env, repeat, clip_rewards, no_ops, fire_first)
    env = PreprocessFrame(shape, env)
    env = StackFrames(env, repeat)
    
    return env

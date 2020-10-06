import gym
import numpy as np
import matplotlib.pyplot as plt

_MAX_ITERATIONS = 1000

env = gym.make('FrozenLake-v0')
env.reset()
env.render()

print('Action Space - ', env.action_space)  # 4 action spaces
print('Observation Space - ', env.observation_space)  # 16 observation spaces

scores = []
win_percent = []
for i in range(0, _MAX_ITERATIONS):
    new_step = env.reset()
    done = False
    while not done:
        score = 0
        random_action = env.action_space.sample()
        new_step, rewards, done, info = env.step(random_action)
        score = score + rewards
    scores.append(score)
    if i % 10 == 0:
        win_percent.append(np.mean(scores[-10:]))

plt.plot(win_percent)
plt.show()



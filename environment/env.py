'''
Simple environment for driving.
'''

import gym
import torch


class Environment(gym.Env):
    def __init__(self, config = None):
        # top-down map
        self.top_map = self.build_map(10, 15)
        # dimensions
        # self.action_space = gym.spaces.Discrete()
        # self.observation_space = gym.spaces.Discrete()

    def build_map(self, x_len: int, y_len: int):
        '''
        Initializes map. Ones are unblocked, zeros are blocked.
        '''
        # border walls
        add_border = torch.nn.ZeroPad2d(padding=1)
        # initialize top-down map
        top_map = torch.ones(x_len, y_len)
        # add borders around it
        top_map = add_border(top_map)
        # # TODO: add more objects later
        return top_map

    def step(self, action: torch.Tensor):
        '''
        Args:
            action:         Action taken by model.
        Returns:
            observation:    Observation of env state.
            reward:         Float of agent reward.
            done:           Boolean of whether model is done.
            info:           Dict of additional information. Not to be used for
                            learning.
        '''

        return observation, reward, done, info

    def reset(self):
        pass


# tests
import matplotlib.pyplot as plt

e = Environment()
plt.imshow(e.top_map)
plt.show()

'''
Simple environment for driving.
'''

import gym
import torch
import numpy as np


class Config:
    def __init__(self, **kwargs):
        # map creation
        self.map_x_len = 10
        self.map_y_len = 10
        # vision
        self.camera_num = 4
        self.frame_height = 10
        self.frame_width = 10


class Environment(gym.Env):
    def __init__(self, config):
        # top-down map
        self.top_map = self.build_map(10, 15)
        #
        # action dimensions
        # gas: [0., 1.], brake: [0., 1.], steer: [-1., 1]
        # self.action_space = gym.spaces.Box(
            # torch.tensor([0., 0., -1.]), torch.tensor([1., 1., 1.])
        # )
        # observation dimensions
        #
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(
                config.camera_num, config.frame_height, config.frame_width, 3
            ),
            dtype=np.uint8
        )

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
        # # TODO: add more objects laterb
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

c = Config()
e = Environment(c)
e.render()
# plt.imshow(e.top_map)
# plt.show()

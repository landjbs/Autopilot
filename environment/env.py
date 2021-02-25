'''
Simple environment for driving.
'''

import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from cached_property import cached_property


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
        # cache
        self.map_x_len = config.map_x_len
        self.map_y_len = config.map_y_len
        # top-down map
        self.top_map = self.build_map(self.map_x_len, self.map_y_len)
        # place agent
        agent_x, agent_y = self.random_empty_locs(1)
        self.agent_x = agent_x
        self.agent_y = agent_y
        self.agent_speed = 0.
        self.agent_angle = 0.
        # waypoints
        self.waypoint = self.build_waypoint()
        # action dimensions
        # gas: [0., 1.], brake: [0., 1.], steer: [-1., 1]
        self.action_space = gym.spaces.Box(
            np.array([0., 0., -1.]), np.array([1., 1., 1.])
        )
        # observation dimensions
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(
                config.camera_num, config.frame_height, config.frame_width, 3
            ),
            dtype=np.uint8
        )
        # timing
        self.time_step = 0

    # initialization
    # @cached_property
    # def coord_map(self):
        # ''' Map converting ids on top-down map to x, y coords '''
#

    def random_empty_locs(self, n: int, top_map: np.array = None):
        ''' Gets random location that is currently empty from map '''
        top_map = top_map if top_map is not None else self.top_map
        i, j = np.nonzero(top_map)
        ix = np.random.choice(len(i), n, replace=False)
        return (i[ix], j[ix])

    def build_map(self, x_len: int, y_len: int):
        '''
        Initializes map. ones are unblocked, zeros are blocked.
        '''
        # initialize top-down map
        top_map = np.ones(shape=(x_len, y_len), dtype=np.int8)
        # add borders around it
        top_map = np.pad(
            top_map, pad_width=1, mode='constant', constant_values=0
        )
        # random objects
        # top_map[self.random_empty_locs(10, top_map)] = 0.
        # # TODO: add more objects later
        return top_map

    def build_waypoint(self):
        pass
        # x_loc =
        # return waypoint

    # vis
    def render(self):
        plt.imshow(self.top_map)
        plt.show()

    # movement
    def _move(self,):
        pass

    # reward
    def _get_reward(self):
        ''' Gets reward for current state '''
        # speed reward


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
        #

        return observation, reward, done, info

    def reset(self):
        pass


# tests
c = Config()
e = Environment(c)
e.render()

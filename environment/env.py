'''
Simple environment for driving.
'''

import gym
import torch


class Environment(gym.Environment):
    def __init__(self):
        self.action_space = gym.spaces.Discrete()
        self.observation_space = gym.spaces.Discrete()

    def step(self, action: torch.Tenosr):
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

    

import gym
import numpy as np
from skimage import transform
from gym.spaces import Box
from random import randint


class ResizeObservation(gym.ObservationWrapper):
    """
    A gym wrapper that resizes observation (states) to a specified shape.
    """
    def __init__(self, env, shape):
        """
        Constructor with which the target shape of the resize transformation can be configured.
        Args:
            env: environment to be wrapped
            shape: target shape
        """
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        # perform the actual resizing
        resize_obs = transform.resize(observation, self.shape)
        # cast float back to uint8
        resize_obs *= 255
        resize_obs = resize_obs.astype(np.uint8)
        return resize_obs


class SkipFrame(gym.Wrapper):
    """
    A wrapper that skips a specified number of frames, so that actions are performed and states are observed only on
    every i-th frame, where the skip parameter specifies i.
    """
    def __init__(self, env, skip_min, skip_max):
        """
        Constructor to configure how many frames shall be skipped
        Args:
            env: environment to be wrapped
            skip: number of frames to be skipped
        """
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip_min = skip_min
        self._skip_max = skip_max

    def step(self, action):
        """Here we repeat one action for several frames, sum the reward, and return the final observation."""
        total_reward = 0.0
        done = False
        actual_skip = randint(self._skip_min, self._skip_max)
        for i in range(actual_skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

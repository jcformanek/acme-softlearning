"""Wraps an OpenAI Gym environment to be used as a dm_env environment."""

from typing import List, Optional
import time

from acme import specs
from acme import types
from acme.utils.loggers import base

import dm_env
import gym
from gym import spaces
import numpy as np
import tensorflow as tf

class TFSummaryLogger(base.Logger):
    """Logs to a tf.summary created in a given logdir.
    """

    def __init__(self, logdir: str, label: str):
        """Initializes the logger.
        """
        self._time = time.time()
        self.label = label
        self._iter = 0
        self.summary = tf.summary.create_file_writer(logdir)

    def write(self, values: base.LoggingData) -> None:
        with self.summary.as_default():
            for key, value in values.items():
                tf.summary.scalar(f"{self.label}/{key}", value, step=self._iter)
            self._iter += 1


class BetterGymWrapper(dm_env.Environment):
    """A Better environment wrapper for OpenAI Gym environments."""

    def __init__(self, environment: gym.Env):

        self._environment = environment
        self._reset_next_step = True

        # Convert action and observation specs.
        obs_space = self._environment.observation_space
        act_space = self._environment.action_space
        self._observation_spec = _convert_to_spec(obs_space, name='observation')
        self._action_spec = _convert_to_spec(act_space, name='action')

    def reset(self) -> dm_env.TimeStep:
        """Resets the episode."""
        self._reset_next_step = False
        observation = self._environment.reset()
        observation = observation.astype(np.float32)
        return dm_env.restart(observation)

    def step(self, action: types.NestedArray) -> dm_env.TimeStep:
        """Steps the environment."""
        if self._reset_next_step:
            return self.reset()

        observation, reward, done, info = self._environment.step(action)
        self._reset_next_step = done

        observation = observation.astype(np.float32)
        reward = float(reward)

        if done:
            truncated = info.get('TimeLimit.truncated', False)
            if truncated:
                return dm_env.truncation(reward, observation)
            return dm_env.termination(reward, observation)
        return dm_env.transition(reward, observation)

    def observation_spec(self) -> types.NestedSpec:
        return self._observation_spec

    def action_spec(self) -> types.NestedSpec:
        return self._action_spec

    @property
    def environment(self) -> gym.Env:
        """Returns the wrapped environment."""
        return self._environment

    def __getattr__(self, name: str):
        # Expose any other attributes of the underlying environment.
        return getattr(self._environment, name)

    def close(self):
        self._environment.close()


def _convert_to_spec(space: gym.Space,
                     name: Optional[str] = None) -> types.NestedSpec:
    """Converts an OpenAI Gym space to a dm_env spec or nested structure of specs.
    Box, MultiBinary and MultiDiscrete Gym spaces are converted to BoundedArray
    specs. Discrete OpenAI spaces are converted to DiscreteArray specs. Tuple and
    Dict spaces are recursively converted to tuples and dictionaries of specs.
    Args:
        space: The Gym space to convert.
        name: Optional name to apply to all return spec(s).
    Returns:
        A dm_env spec or nested structure of specs, corresponding to the input
        space.
    """
    if isinstance(space, spaces.Discrete):
        return specs.DiscreteArray(num_values=space.n, dtype=space.dtype, name=name)

    elif isinstance(space, spaces.Box):
        return specs.BoundedArray(
            shape=space.shape,
            dtype=np.float32,
            minimum=space.low,
            maximum=space.high,
            name=name)

    elif isinstance(space, spaces.MultiBinary):
        return specs.BoundedArray(
            shape=space.shape,
            dtype=np.float32,
            minimum=0.0,
            maximum=1.0,
            name=name)

    elif isinstance(space, spaces.MultiDiscrete):
        return specs.BoundedArray(
            shape=space.shape,
            dtype=np.float32,
            minimum=np.zeros(space.shape),
            maximum=space.nvec - 1,
            name=name)

    elif isinstance(space, spaces.Tuple):
        return tuple(_convert_to_spec(s, name) for s in space.spaces)

    elif isinstance(space, spaces.Dict):
        return {
            key: _convert_to_spec(value, key) for key, value in space.spaces.items()
        }

    else:
        raise ValueError('Unexpected gym space: {}'.format(space))
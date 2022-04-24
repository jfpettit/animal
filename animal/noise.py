"""This module contains Environment wrappers for noise observation and action noise."""

import numpy as np
from gym import Env
from gym.core import ActionWrapper, ObservationWrapper
from gym.spaces import Box, Discrete


class BetaNoiseObservationWrapper(ObservationWrapper):
    """Gym wrapper for adding Beta noise to observation space."""

    def __init__(self, env: Env, alpha: int = 1, beta: int = 1) -> None:
        assert type(env.observation_space) in (Box, Discrete), "Only `Box` and `Discrete` spaces supported"

        super().__init__(env)

        space = env.observation_space

        self._env_obs_discrete = isinstance(space, Discrete)
        self._env_obs_dim = (1,) if self._env_obs_discrete else space.shape
        self._env_obs_min = np.array([0]) if self._env_obs_discrete else space.low
        self._env_obs_max = np.array([space.n]) if self._env_obs_discrete else space.high

        self._noise_param_alpha, self._noise_param_beta = alpha, beta
        self._noise_dist_mean = self._noise_param_alpha / (self._noise_param_alpha + self._noise_param_beta)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        noise = np.random.beta(self._noise_param_alpha, self._noise_param_beta, size=self._env_obs_dim)
        return np.clip(observation * (1 + noise - self._noise_dist_mean), self._env_obs_min, self._env_obs_max)


class BetaNoiseActionWrapper(ActionWrapper):
    """Gym wrapper for adding Beta noise to action space."""

    def __init__(self, env: Env, alpha: int = 1, beta: int = 1) -> None:
        assert type(env.action_space) in (Box, Discrete), "Only `Box` and `Discrete` spaces supported"

        super().__init__(env)

        space = env.action_space

        self._env_act_discrete = isinstance(space, Discrete)
        self._env_act_dim = (1,) if self._env_act_discrete else space.shape
        self._env_act_min = np.array([0]) if self._env_act_discrete else space.low
        self._env_act_max = np.array([space.n]) if self._env_act_discrete else space.high

        self._noise_param_alpha, self._noise_param_beta = alpha, beta
        self._noise_dist_mean = self._noise_param_alpha / (self._noise_param_alpha + self._noise_param_beta)

    def action(self, action: np.ndarray) -> np.ndarray:
        noise = np.random.beta(self._noise_param_alpha, self._noise_param_beta, size=self._env_act_dim)
        return np.clip(action * (1 + noise - self._noise_dist_mean), self._env_act_min, self._env_act_max)

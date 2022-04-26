"""Generic helper methods and enviroment objects for animal."""

from __future__ import annotations

import gym
import numpy as np
import torch
import yaml
from gym.core import Env
from gym.spaces import Box
from torch import Tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_yaml_file(filepath: str) -> dict:
    """Load YAML file into memory from path."""
    with open(filepath, "rb") as infile:
        contents = yaml.safe_load(infile)
    return contents


def write_yaml_file(contents: dict, filepath: str) -> None:
    """Load YAML file into memory from path."""
    with open(filepath, "w") as outfile:
        outfile.write(yaml.dump(contents))
    return None


class RunningActionStats:
    """Accumate distribution parameters from enviorment space."""

    def __init__(self, env: Env, beta: float = 1 - 1e-2, eps: float = 1e-8) -> None:
        self.env, self.beta, self.eps = env, beta, eps
        self.mu = np.zeros(self.env.action_space.shape)
        self.var = np.ones(self.env.action_space.shape)

    def update(self, action: np.ndarray) -> None:
        """Update distribution parameters from action."""
        self.mu = self.beta * self.mu + (1.0 - self.beta) * action
        self.var = self.beta * self.var + (1.0 - self.beta) * np.square(action - self.mu)
        return None

    def reset(self) -> None:
        """Reset learned distribution parameters from actions."""
        self.mu = np.zeros(self.env.action_space.shape)
        self.var = np.ones(self.env.action_space.shape)
        return None


class PyTorchWrapper(gym.core.Wrapper):
    """Gym wrapper for converting relevant objects to tensors.

    Note
    ----
    - Placing these Tensor transformations inside of a gym wrapper may
      break the (hierarchical) inheritance functionality of others gym
      wrappers. This should probably be moved out of here and directly
      into the consuming networks runners. Alternatively, a warning could
      be placed upon initialization to flag potential side-effects of
      consuming this object.

    """

    def __init__(self, env: Env, device: str = device) -> None:
        super().__init__(env)
        self.device = device

    def reset(self) -> Tensor:
        """Reset environment to an initial state."""
        return torch.as_tensor(self.env.reset(), dtype=torch.float32).to(self.device)

    def step(self, action: np.ndarray) -> tuple[Tensor, float, bool, dict]:
        """Run single timestep over environment."""
        observation, reward, done, info = self.env.step(action)
        observation = torch.as_tensor(observation, dtype=torch.float32).to(self.device)
        return observation, reward, done, info


class BetaNoiseObservationWrapper(gym.core.ObservationWrapper):
    """Gym wrapper for adding Beta noise to observation space."""

    def __init__(self, env: Env, alpha: int = 1, beta: int = 1) -> None:
        assert isinstance(env.observation_space, Box), "Only continuous `Box` spaces supported"
        self._param_obs_alpha, self._param_obs_beta = alpha, beta
        super().__init__(env)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """Apply perturbation to observation using BetaDistribution sampling."""
        alpha, beta, size = self._param_obs_alpha, self._param_obs_beta, self.observation_space.shape
        update = observation * (1 + np.random.beta(alpha, beta, size=size) - alpha / (alpha + beta))
        return np.clip(update, self.observation_space.low, self.observation_space.high)


class BetaNoiseActionWrapper(gym.core.ActionWrapper):
    """Gym wrapper for adding Beta noise to action space."""

    def __init__(self, env: Env, alpha: int = 1, beta: int = 1) -> None:
        assert isinstance(env.action_space, Box), "Only continuous `Box` spaces supported"
        self._param_act_alpha, self._param_act_beta = alpha, beta
        super().__init__(env)

    def action(self, action: np.ndarray) -> np.ndarray:
        """Apply perturbation to action using BetaDistribution sampling."""
        alpha, beta, size = self._param_act_alpha, self._param_act_beta, self.action_space.shape
        update = action * (1 + np.random.beta(alpha, beta, size=size) - alpha / (alpha + beta))
        return np.clip(update, self.action_space.low, self.action_space.high)

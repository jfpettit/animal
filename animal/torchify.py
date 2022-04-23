import torch
import gym
from gym import Env

class TorchifyEnv(gym.Wrapper):
    def __init__(self, env: Env) -> None:
        super().__init__(env)

    def reset(self):
        return torch.as_tensor(self.env.reset(), dtype=torch.float32)

    def step(self, a):
        obs, rew, done, info = self.env.step(a)
        obs = torch.as_tensor(list(obs), dtype=torch.float32)
        return obs, rew, done, info
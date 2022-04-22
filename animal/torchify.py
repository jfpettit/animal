import torch
import gym
from gym import Env

class TorchifyEnv(gym.Wrapper):
    def __init__(self, env: Env, device: str = "cpu") -> None:
        super().__init__(env)
        self.device = device

    def reset(self):
        return torch.as_tensor(self.env.reset(), dtype=torch.float32).to(self.device)

    def step(self, a):
        obs, rew, done, info = self.env.step(a)
        obs = torch.as_tensor(list(obs), dtype=torch.float32).to(self.device)
        return obs, rew, done, info
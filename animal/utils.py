import numpy as np

class RunningActionStats:
    def __init__(self, env, beta: float = 0.99, eps: float = 1e-8):
        self.env = env
        self.beta = beta
        self.eps = eps

        self.mu = np.zeros(self.env.action_space.shape)
        self.var = np.ones(self.env.action_space.shape)

    def update(self, action):
        self.mu = self.beta * self.mu + (1. - self.beta) * action
        self.var = self.beta * self.var + (1. - self.beta) * np.square(action - self.mu)

    def reset(self):
        self.mu = np.zeros(self.env.action_space.shape)
        self.var = np.ones(self.env.action_space.shape)

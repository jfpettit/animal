from itertools import chain
from copy import deepcopy
import torch
import torch.nn as nn
from kindling.buffers import ReplayBuffer
from kindling.neuralnets import FireSACActorCritic
from torchify import TorchifyEnv
import gym
import numpy as np

class SAC:
    def __init__(
        self,
        env: str,
        alpha: float = 0.2,
        gamma: float = 0.99
    ) -> None:

        self.alpha = alpha
        self.gamma = gamma
        self.bellman_backup_loss = nn.MSELoss()

        self.env = gym.make(env)
        self.ac = FireSACActorCritic(
            self.env.observation_space.shape[0],
            self.env.action_space
        )
        self.ac_targ = deepcopy(self.ac)

        self.policy_optimizer = torch.optim.Adam(self.ac.policy.parameters(), lr=1e-3)
        self.qfunc_params = chain(
            self.ac.qfunc1.parameters(),
            self.ac.qfunc2.parameters()
        )
        self.q_optimizer = torch.optim.Adam(self.qfunc_params, lr=1e-3)

    def calc_pol_loss(self, batch):
        states, next_states, acts, rews, dones = batch
        pi, logp_pi = self.ac.policy(states)
        q1_pi = self.ac.qfunc1(states, pi)
        q2_pi = self.ac.qfunc2(states, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        pi_info = dict(PolicyLogP=logp_pi.detach().numpy())

        return loss_pi, pi_info

    def calc_q_loss(self, batch):
        states, next_states, acts, rews, dones = batch

        q1 = self.ac.qfunc1(states, acts)
        q2 = self.ac.qfunc2(states, acts)

        with torch.no_grad():
            acts_next, logp_acts_next = self.ac.policy(next_states)

            q1_pi_targ = self.ac_targ.qfunc1(next_states, acts_next)
            q2_pi_targ = self.ac_targ.qfunc2(next_states, acts_next)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            bellman_backup = rews + self.gamma * (1- dones) * (q_pi_targ - self.alpha * logp_acts_next)

        loss_q1 = self.bellman_backup_loss(q1, bellman_backup)
        loss_q2 = self.bellman_backup_loss(q2, bellman_backup)
        loss_q = loss_q1 + loss_q2

        q_info = dict(
            Q1Values=q1.detach().numpy(),
            Q2Values=q2.detach().numpy()
        )

        return loss_q, q_info


    def get_batch(self):
        state = self.env.reset()
        reward = 0
        R = 0
        episode_length = 0
        rewlst = []
        lenlst = []

        for i in range(self.batch_size):
            action, logp, value = self.ac.step(state)
            next_state, reward, done, info = self.env.step(action)

            self.buffer.store(
                state, action, reward, value, logp
            )

            state = next_state
            episode_length += 1
            R += reward

            # If the episode has hit the horizon
            timeup = episode_length == self.horizon
            # If the episode is truly done or it has just hit the horizon
            over = done or timeup
            # if the full batch has been collected
            epoch_ended = i == self.batch_size - 1
            if over or epoch_ended:
                if timeup or epoch_ended:
                    last_val = self.ac.value_f(state).detach().numpy()
                else:
                    last_val = 0

                self.buffer.finish_path(last_val=last_val)

                if over:
                    rewlst.append(R)
                    lenlst.append(episode_length)
                
                state = self.env.reset()
                R = 0
                episode_length = 0
        
        track = {
            "MeanEpReturn": np.mean(rewlst),
            "StdEpReturn": np.std(rewlst),
            "MaxEpReturn": np.max(rewlst),
            "MinEpReturn": np.min(rewlst),
            "MeanEpLength": np.mean(lenlst)
        }
        self.tracker_dict.update(track)

        return self.buffer.get()


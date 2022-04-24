import click
import os
import yaml
from itertools import chain
from copy import deepcopy
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from kindling.buffers import ReplayBuffer
from kindling.neuralnets import FireSACActorCritic
from animal.torchify import TorchifyEnv
import gym
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import time

class SAC:
    def __init__(
        self,
        env: str,
        epochs: int = 100,
        alpha: float = 0.2,
        gamma: float = 0.99,
        batch_size: int = 64,
        steps_per_epoch: int = 4000,
        warmup_steps: int = 10000,
        update_after: int = 1000,
        update_every: int = 50,
        act_noise: float = 0.1,
        num_test_episodes: int = 10,
        replay_size: int = int(1e6),
        polyak: float = 0.995,
        horizon: int = 1000,
        bellman_loss: str = "mse",
        env_kwargs: dict = {},
        device: str = "cpu",
        agent_name: str = 'Agent',
        seed: int = 0
    ) -> None:

        self.alpha = alpha
        self.gamma = gamma
        if bellman_loss.lower() == "mse":
            self.bellman_backup_loss = nn.MSELoss()
        elif bellman_loss.lower() == "huber":
            self.bellman_backup_loss = nn.HuberLoss()
        else:
            raise ValueError(f"Loss {bellman_loss} is not an option, pick `mse` or `huber`.")
        self.start = 0
        self.device = device
        self.steps_per_epoch = steps_per_epoch
        self.warmup_steps = warmup_steps
        self.steps = steps_per_epoch * epochs
        self.update_after = update_after
        self.update_every = update_every
        self.act_noise = act_noise
        self.num_test_episodes = num_test_episodes
        self.polyak = polyak
        self.horizon = horizon
        if ':' in env:
            self.env_name = env.split(':')[1]
        else:
            self.env_name = env
        self.agent_name = f"{self.env_name}-agent_name-{int(time.time())}"
        self.episodes_completed = 0
        self.batch_size = batch_size

        self.env = TorchifyEnv(gym.make(env, **env_kwargs), device=device)
        self.test_env = TorchifyEnv(gym.make(env, **env_kwargs), device=device)
        self.act_dim = self.env.action_space.shape[0]
        self.act_limit = self.env.action_space.high[0]

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)

        self.buffer = ReplayBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            replay_size
        )
        self.ac = FireSACActorCritic(
            self.env.observation_space.shape[0],
            self.env.action_space
        )
        self.ac_targ = deepcopy(self.ac)

        self.ac.to(self.device)
        self.ac_targ.to(self.device)

        for param in self.ac_targ.parameters():
            param.requires_grad = False

        self.policy_optimizer = torch.optim.Adam(self.ac.policy.parameters(), lr=1e-3)
        self.qfunc_params = chain(
            self.ac.qfunc1.parameters(),
            self.ac.qfunc2.parameters()
        )
        self.q_optimizer = torch.optim.Adam(self.qfunc_params, lr=1e-3)

        self.t = 0
        self.tracker_dict = {}

    def calc_pol_loss(self, batch):
        states, _, _, _, _ = batch
        states = torch.as_tensor(states, dtype=torch.float32).to(self.device)
        pi, logp_pi = self.ac.policy(states)
        pi = torch.as_tensor(pi, dtype=torch.float32).to(self.device)
        q1_pi = self.ac.qfunc1(states, pi)
        q2_pi = self.ac.qfunc2(states, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        pi_info = dict(
            MeanPolicyLogP=logp_pi.mean().cpu().detach().numpy(),
            PolicyLoss=loss_pi
        )
        self.tracker_dict.update(pi_info)

        return loss_pi, pi_info

    def calc_q_loss(self, batch):
        states, next_states, acts, rews, dones = batch
        states = torch.as_tensor(states, dtype=torch.float32).to(self.device)
        next_states = torch.as_tensor(next_states, dtype=torch.float32).to(self.device)
        acts = torch.as_tensor(acts, dtype=torch.float32).to(self.device)
        rews = torch.as_tensor(rews, dtype=torch.float32).to(self.device)
        dones = torch.as_tensor(dones).to(self.device)

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
            Q1MeanValues=q1.mean().cpu().detach().numpy(),
            Q2MeanValues=q2.mean().cpu().detach().numpy(),
            Q1Loss=loss_q1,
            Q2Loss=loss_q2
        )
        self.tracker_dict.update(q_info)

        return loss_q, q_info

    def test_agent(self, num_test_episodes, max_ep_len):
        test_return = []
        test_length = []
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = self.test_env.reset(), False, 0, 0
            while not (d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = self.test_env.step(self.get_action(o, 0))
                ep_ret += r
                ep_len += 1
            test_return.append(ep_ret)
            test_length.append(ep_len)
            trackit = dict(MeanTestEpReturn=np.mean(test_return), MeanTestEpLength=np.mean(test_length))
        return trackit

    def get_action(self, o, noise_scale):
        a = self.ac.act(torch.as_tensor(o, dtype=torch.float32))
        a += noise_scale * np.random.randn(self.act_dim)
        return np.clip(a, -self.act_limit, self.act_limit)

    def run(self):
        max_ep_len = self.horizon
        state, episode_return, episode_length = self.env.reset(), 0, 0
        rewlst = []
        lenlst = []
        epoch = 0
        for i in tqdm(range(self.steps)):
            # Main loop: collect experience in env and update/log each epoch

            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards,
            # use the learned policy (with some noise, via act_noise).
            if self.t > self.warmup_steps:
                action = self.get_action(state, self.act_noise)
            else:
                action = self.env.action_space.sample()

            # Step the env
            next_state, reward, done, _ = self.env.step(action)
            episode_return += reward
            episode_length += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            done = False if episode_length == max_ep_len else done

            # Store experience to replay buffer
            self.buffer.store(state.cpu(), action, reward, next_state.cpu(), done)

            # Super critical, easy to overlook step: make sure to update
            # most recent observation!
            state = next_state

            # End of trajectory handling
            if done or (episode_length == max_ep_len):
                rewlst.append(episode_return)
                lenlst.append(episode_length)
                self.summary_writer.add_scalar('Episode Reward', episode_return, self.episodes_completed)
                self.episodes_completed += 1
                state, episode_return, episode_length = self.env.reset(), 0, 0

            self.t += 1
            if self.t > self.update_after and self.t % self.update_every == 0:
                trackit = {
                    "MeanEpReturn": np.mean(rewlst[-self.update_every:]),
                    "StdEpReturn": np.std(rewlst[-self.update_every:]),
                    "MaxEpReturn": np.max(rewlst[-self.update_every:]),
                    "MinEpReturn": np.min(rewlst[-self.update_every:]),
                    "MeanEpLength": np.mean(lenlst[-self.update_every:]),
                }
                self.summary_writer.add_scalar('Mean Episode Reward', np.mean(rewlst[-self.update_every:]), epoch)
                self.summary_writer.add_scalar('Std Episode Reward', np.std(rewlst[-self.update_every:]), epoch)
                self.summary_writer.add_scalar('Max Episode Reward', np.max(rewlst[-self.update_every:]), epoch)
                self.summary_writer.add_scalar('Min Episode Reward', np.mean(lenlst[-self.update_every:]), epoch)
                self.tracker_dict.update(trackit)
                self.update()
            # End of epoch handling
            if (self.t + 1) % self.steps_per_epoch == 0:

                # Test the performance of the deterministic version of the agent.
                testtrack = self.test_agent(
                    num_test_episodes=self.num_test_episodes, max_ep_len=max_ep_len
                )
                self.tracker_dict.update(testtrack)
                print(f"=== EPOCH {epoch} ===")
                self.printdict()
                state = self.env.reset()
                episode_length = 0
                episode_return = 0
                epoch += 1

        return self.tracker_dict

    def printdict(self) -> None:
        r"""
        Print the contents of the epoch tracking dict to stdout or to a file.
        Args:
            out_file (sys.stdout or string): File for output. If writing to a file, opening it for writing should be handled in :func:`on_epoch_end`.
        """
        for k, v in self.tracker_dict.items():
            print(f"{k}: {v}")
        print("\n")

    def update(self):
        trackit = {}
        batch = self.buffer.sample_batch(batch_size=self.batch_size)

        self.q_optimizer.zero_grad()
        loss_q, q_info = self.calc_q_loss(batch)
        loss_q.backward()
        self.q_optimizer.step()

        trackit["QLoss"] = loss_q.item()
        trackit.update(q_info)

        for p in self.qfunc_params:
            p.requires_grad = False

        self.policy_optimizer.zero_grad()
        loss_pi, pi_info = self.calc_pol_loss(batch)
        loss_pi.backward()
        self.policy_optimizer.step()

        for p in self.qfunc_params:
            p.requires_grad = True

        trackit["PolicyLoss"] = loss_pi.item()
        trackit.update(pi_info)
        
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
        
    def set_file_structure(self,config):
        folder = os.getcwd()
        folders = [folder,'tensorboards', 'SAC', self.env_name, self.agent_name ]
        folder = os.path.join(*folders ) 
        os.makedirs(folder)
        self.summary_writer = SummaryWriter(logdir=folder)

        yaml_file = self.agent_name + '.yaml'
        yaml_path = os.path.join(folder, yaml_file)

        with open(yaml_path, 'w') as yf:
            yf.write(yaml.dump(config))

def genConfigs(config):
    temp = [[]]
    params = []
    for param in config['param_search']:
        params = []
        for val in config[param]:
            for item in temp:
                copy = deepcopy(item)
                copy.append(val)
                params.append(copy)
        temp = deepcopy(params)

    return params

def trainSAC(config):
    agent = SAC(**config)
    agent.set_file_structure(config)
    agent.run()
        
@click.command()
@click.option("--config-file", "-cf", default="configs/default_sac.yaml")
def main(config_file):
    with open(config_file, 'rb') as f:
        config = yaml.safe_load(f)

    if 'param_search' in config.keys():
        params = genConfigs(config)
        params_to_search = config['param_search']
        del[config['param_search']]
        for agent_num, param_set in enumerate(params):
            config['agent_name'] = 'Agent' + str(agent_num)
            for i in range(len(params_to_search)):
                config[params_to_search[i]] = param_set[i]
            trainSAC(config)
    else:
        trainSAC(config)

if __name__ == "__main__":
    main()
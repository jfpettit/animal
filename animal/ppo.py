import click
import os
import gym
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from kindling.buffers import PGBuffer
from kindling.datasets import PolicyGradientRLDataset
from kindling.neuralnets import FireActorCritic
from kindling import utils as utils
from torchify import TorchifyEnv
from utils import RunningActionStats
import yaml
from tqdm import tqdm
from copy import deepcopy

# TODO:
#   1. Make minibatch size actually do something. One way is to tie into torch.Datasets and torch.DataLoaders, create new dataset and dataloader at the end of each batch
#   2. Add config loading
#   3. Support more options via CLI
#   4. Test the code
#   5. Add option to learn action distribution variance as well as mean
#   6. Make play nice with ContinuousCartPole. Current action sampling disregards bounds of the action space, this can be a problem (like it is with ContinuousCartPole)
class PPO:
    def __init__(self,
        env: str,
        epochs: 100,
        val_loss: str = "mse",
        clipratio: float = 0.2,
        train_iters: int = 80,
        batch_size: int = 4000,
        minibatch_size: int = 256,
        gamma: float = 0.99,
        lam: float = 0.95,
        horizon: int = 1000,
        maxkl: float = 0.01,
        env_kwargs: dict = {},
        agent_name: str = 'Agent',
        seed: int = 0,
    ) -> None:
        if val_loss.lower() == "mse":
            self.val_loss = nn.MSELoss()
        elif val_loss.lower() == "huber":
            self.val_loss = nn.HuberLoss()
        else:
            raise ValueError(f"Value loss {val_loss} not supported. Use `mse` or `huber`.")
        self.tracker_dict = {}
        self.env = TorchifyEnv(gym.make(env, **env_kwargs))
        self.clipratio = clipratio
        self.batch_size = batch_size
        self.train_iters = train_iters
        self.minibatch_size = minibatch_size
        self.horizon = horizon
        self.maxkl = maxkl
        self.epochs = epochs
        self.action_stats = RunningActionStats(self.env)
        self.agent_name = agent_name
        self.episode_reward = 0 
        self.episodes_completed = 0 
        torch.manual_seed(seed)
        
        self.buffer = PGBuffer(
            obs_dim=self.env.observation_space.shape[0],
            act_dim=self.env.action_space.shape,
            size=batch_size,
            gamma=gamma,
            lam=lam
        )

        self.ac = FireActorCritic(
            self.env.observation_space.shape[0],
            self.env.action_space
        )

        self.policy_optimizer = torch.optim.Adam(self.ac.policy.parameters(), lr=3e-4)
        self.value_optimizer = torch.optim.Adam(self.ac.value_f.parameters(), lr=1e-3)

    def calc_pol_loss(self, logps, logps_old, advs):
        ratio = torch.exp(logps - logps_old)
        clipped_adv = torch.clamp(ratio, 1 - self.clipratio, 1 + self.clipratio) * advs
        pol_loss = -(torch.min(ratio * advs, clipped_adv)).mean()

        kl = (logps_old - logps).mean().item()
        return pol_loss, kl
    
    def calc_val_loss(self, values, rets):
        return self.val_loss(values, rets)

    def value_update(self, batch):
        states, _, _, rets, _ = batch

        values_old = self.ac.value_f(states)
        val_loss_old = self.calc_val_loss(values_old, rets)
        for i in range(self.train_iters):
            self.value_optimizer.zero_grad()
            values = self.ac.value_f(states)
            val_loss = self.calc_val_loss(values, rets)
            val_loss.backward()
            self.value_optimizer.step()

        delta_val_loss = (val_loss - val_loss_old).item()
        log = {"ValueLoss": val_loss_old.item(), "DeltaValLoss": delta_val_loss}
        loss = val_loss

        self.tracker_dict.update(log)
        return {"loss": loss, "log": log}

    def policy_update(self, batch):
        states, actions, advs, _, logps_old = batch
        stops = 0
        stopslst = []
        policy, logps = self.ac.policy(states, actions)
        pol_loss_old, kl = self.calc_pol_loss(logps, logps_old, advs)

        for i in range(self.train_iters):
            self.policy_optimizer.zero_grad()
            policy, logps = self.ac.policy(states, actions)
            pol_loss, kl = self.calc_pol_loss(logps, logps_old, advs)
            if kl > 1.5 * self.maxkl:
                stops += 1
                stopslst.append(i)
                break
            pol_loss.backward()
            self.policy_optimizer.step()

        log = {
            "PolicyLoss": pol_loss_old.item(),
            "DeltaPolLoss": (pol_loss - pol_loss_old).item(),
            "KL": kl,
            "Entropy": policy.entropy().mean().item(),
            "TimesEarlyStopped": stops,
            "AvgEarlyStopStep": np.mean(stopslst) if len(stopslst) > 0 else 0
        }
        self.tracker_dict.update(log)
        loss = pol_loss_old
        return {"loss": loss, "log": log}

    def update(self, batch):
        pol_out = self.policy_update(batch)
        val_out = self.value_update(batch)

        return pol_out, val_out

    def get_batch(self) -> None:
        
        state = self.env.reset()
        reward = 0
        R = 0
        episode_length = 0
        rewlst = []
        lenlst = []

        for i in range(self.batch_size):
            action, logp, value = self.ac.step(state)
            self.action_stats.update(action)
            next_state, reward, done, info = self.env.step(action)

            self.buffer.store(
                state, action, reward, value, logp
            )

            state = next_state
            episode_length += 1
            R += reward
            self.episode_reward += reward

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
                    self.episodes_completed += 1
                    self.summary_writer.add_scalar('episode_reward', self.episode_reward, self.episodes_completed)
                    self.summary_writer.add_scalar('episode_length', episode_length, self.episodes_completed)
                    self.episode_reward = 0
                
                state = self.env.reset()
                R = 0
                episode_length = 0
        
        track = {
            "MeanEpReturn": np.mean(rewlst),
            "StdEpReturn": np.std(rewlst),
            "MaxEpReturn": np.max(rewlst),
            "MinEpReturn": np.min(rewlst),
            "MeanEpLength": np.mean(lenlst),
            #"PolicyDistVariance": self.ac.policy.logstd.mean().exp().sqrt().item(),
            "ActionsTakenMean": self.action_stats.mu,
            "ActionsTakenVariance": self.action_stats.var
        }
        self.tracker_dict.update(track)

        return self.buffer.get()

    def run(self):
        for i in tqdm(range(self.epochs)):
            batch = self.get_batch()
            batch = [torch.as_tensor(b) for b in batch]
            pol_out, val_out = self.update(batch)
            print(f"\n=== EPOCH {i} ===")
            for k, v in self.tracker_dict.items():
                print(f"{k}: {v}")

    def set_file_structure(self,config):
        folder = os.path.join(os.getcwd(), 'tensorboards', self.agent_name) 
        if not os.path.isdir(folder):
            os.mkdir(folder)
        self.summary_writer = SummaryWriter(logdir=folder)
        yaml_file = self.agent_name + '.yaml'
        yaml_path = os.path.join(folder, yaml_file)
        with open(yaml_path, 'w') as yf:
            yaml.dump(config, yf)


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

def trainPPO(config):
    agent = PPO(**config)
    agent.set_file_structure(config)
    agent.run()
           

@click.command()
@click.option("--config-file", "-cf", default="configs/default_ppo.yaml")
def main(
    config_file,
):
    with open(config_file, "rb") as f:
        config = yaml.safe_load(f)
        #set_file_structure(config)
    
    if 'param_search' in config.keys():
        params = genConfigs(config)
        params_to_search = config['param_search']
        del[config['param_search']]
        for agent_num, param_set in enumerate(params):
            config['agent_name'] = 'Agent' + str(agent_num)
            for i in range(len(params_to_search)):
                config[params_to_search[i]] = param_set[i]
            trainPPO(config)
    else:
        trainPPO(config)

           

if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import torch.nn.functional as F
import torch.nn.init as init
import torch.nn.parameter as parameter
import torchrl
from torchrl import data
import pathlib
import tensordict
import numpy as np
import utils
import tqdm
import typing
import math
import datetime
from agents import state_encoder
from agetns import agent

class Actor(nn.Module):
    def __init__(self, embed_size, action_size, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(3, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        self.max_action = max_action

    def forward(self, embed):
        x = F.silu(self.l1(embed))
        x = F.silu(self.l2(x))
        x = self.max_action * F.tanh(self.l3(x))
        return x

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(4, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, lat, action):
        x = torch.cat((lat, action), dim=-1)
        x = F.silu(self.l1(x))
        x = F.silu(self.l2(x))
        x = self.l3(x)
        return x

class PPOAgent(agent.Agent):
    def __init__(self, env, config):
        self.env = env
        self.config = config
        # Device management
        if torch.cuda.is_available():
            self.device = torch.device(self.config.device)
        else:
            self.device = torch.device("cpu")

        # Networks
        self.actor = Actor(self.config.embed_size, self.config.action_size, self.config.max_action).to(self.device)
        self.critic = Critic().to(self.device)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config.critic_lr)

    def select_action(self, obs, probabilistic=False):
        with torch.no_grad():
            action = self.actor(obs.unsqueeze(0))
        return action.squeeze(0).numpy(force=True)

    def update(self):
        pass

    def step(self, num_steps=0, evaluate=False):
        done = False
        episode_steps = 0
        episode_reward = 0.0
        obs, _ = self.env.reset()
        while not done:
            obs = torch.tensor(obs).to(self.device)
            action = self.select_action(obs, probabilistic=not evaluate)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            if not evaluate:
                self.update()
            episode_reward += reward
            episode_steps += 1
            if episode_steps < num_steps and done:
                obs, _ = self.env.reset()
                done = False
            else:
                obs = next_obs
        return episode_reward, episode_steps, info

    def _to_tensordict(self, obs, action, reward, next_obs, done):
        d = {
            "obs": obs,
            "action": action.astype(np.float32),
            "rewards": np.expand_dims(reward, 0).astype(np.float32),
            "next_obs": next_obs,
            "dones": np.expand_dims(done, 0).astype(np.float32),
        }
        return tensordict.TensorDict(d)

    def train(self, num_episodes, num_steps=0):
        # train agent
        print("Start training")
        history = {
            "rewards": np.zeros(num_episodes, dtype=np.float32),
            "steps": np.zeros(num_episodes, dtype=np.uint32),
        }
        for episode in range(num_episodes):
            start_time = datetime.datetime.now()
            reward, steps, info = self.step(num_steps=num_steps)
            end_time = datetime.datetime.now()
            history["rewards"][episodes] = reward.astype(np.float32)
            history["steps"][episode] = steps
            print(f"{episode}: {steps=} {reward=} (elapsed time: {end_time - start_time})")
        return history

    def evaluate(self):
        reward, steps, info = self.step(evaluate=True)

    def save(self, file=None):
        file = pathlib.Path(file if file is not None else self.config.save_file).resolve()
        checkpoint = dict(
            actor=self.actor.state_dict(),
            critic=self.critic.state_dict(),
            actor_optimizer=self.actor_optimizer.state_dict(),
            critic_optimizer=self.critic_optimizer.state_dict(),
        )
        torch.save(checkpoint, file)

    def load(self, file=None):
        file = pathlib.Path(file if file is not None else self.config.save_file).resolve()
        checkpoint = torch.load(file)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

class PPOAgentConfig:
    device = "cuda"
    batch_size = 256
    num_update = 100
    update_interval = 10
    embed_size = 2048
    action_size = 4
    max_action = 2.0
    gamma = 0.99
    reward_scale = 0.01
    actor_lr = 3e-4
    critic_lr = 3e-4
    save_file = pathlib.Path("./chkpts/ppo_agent").resolve()

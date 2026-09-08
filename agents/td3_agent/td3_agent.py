import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import torch.optim as optim
import torchrl
from torchrl import data
import pathlib
import tensordict
import numpy as np
import tqdm
from agents import state_encoder
from agents import agent
import utils

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(768, 512)
        self.l2 = nn.Linear(512, 128)
        self.l3 = nn.Linear(128, 4)

    def forward(self, lat):
        x = F.silu(self.l1(lat))
        x = F.silu(self.l2(x))
        x = F.sigmoid(self.l3(x))
        return x

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(772, 512)
        self.l2 = nn.Linear(512, 128)
        self.l3 = nn.Linear(128, 1)

    def forward(self, lat, action):
        x = torch.cat((lat, action), dim=-1)
        x = F.silu(self.l1(x))
        x = F.silu(self.l2(x))
        return self.l3(x)

class TD3Agent(agent.Agent):
    def __init__(self, env, config):
        self.env = env
        self.config = config
        # Device management
        if torch.cuda.is_available():
            self.device = torch.device(self.config.device)
        else:
            self.device = torch.device("cpu")

        # Replay Buffer
        self.replay_buffer = data.TensorDictReplayBuffer(storage=data.replay_buffers.LazyMemmapStorage(max_size=self.config.buffer_size, scratch_dir=self.config.replay_buffer_dir, auto_cleanup=True), batch_size=self.config.batch_size)

        # Networks
        self.state_encoder = state_encoder.StateEncoder().to(self.device)
        self.p_actor = Actor().to(self.device)
        self.p_target = Actor().to(self.device)
        self.q1_critic = Critic().to(self.device)
        self.q2_critic = Critic().to(self.device)
        self.q1_target = Critic().to(self.device)
        self.q2_target = Critic().to(self.device)
        self.p_target.load_state_dict(self.p_actor.state_dict())
        self.q1_target.load_state_dict(self.q1_critic.state_dict())
        self.q2_target.load_state_dict(self.q2_critic.state_dict())

        # Optimizers
        self.state_encoder_optimizer = optim.SGD(self.state_encoder.parameters(), lr=self.config.state_encoder_lr)
        self.p_optimizer = optim.SGD(self.p_actor.parameters(), lr=self.config.actor_lr, maximize=True)
        self.q1_optimizer = optim.SGD(self.q1_critic.parameters(), lr=self.config.critic_lr)
        self.q2_optimizer = optim.SGD(self.q2_critic.parameters(), lr=self.config.critic_lr)

    def select_action(self, obs, probabilistic=False):
        with torch.no_grad():
            lat = self.state_encoder.encode(obs.unsqueeze(0))
            action = self.p_actor(lat)
            e = torch.randn_like(action) if probabilistic else 0.0
            action = torch.clip(action + e, 0.0, 1.0)
        return action.squeeze().numpy(force=True)

    def soft_update(self):
        for pa, pt in zip(self.p_actor.parameters(), self.p_target.parameters()):
            pt.data.mul_(self.config.tau)
            pt.data.add_((1.0 - self.config.tau) * pa.data)
        for pc, pt in zip(self.q1_critic.parameters(), self.q1_target.parameters()):
            pt.data.mul_(self.config.tau)
            pt.data.add_((1.0 - self.config.tau) * pc.data)
        for pc, pt in zip(self.q2_critic.parameters(), self.q2_target.parameters()):
            pt.data.mul_(self.config.tau)
            pt.data.add_((1.0 - self.config.tau) * pc.data)

    def update(self):
        assert self.config.batch_size <= self.replay_buffer.write_count, "Replay buffer is not sufficiently filled for sampling!"
        for j in range(self.config.num_updates):
            transitions = self.replay_buffer.sample(self.config.batch_size).to(self.device)
            with torch.no_grad():
                next_lat = self.state_encoder.encode(transitions["next_obs"])
                next_samples = self.p_target(next_lat)
                e = torch.clip(torch.randn_like(next_samples) * self.config.update_noise, -self.config.noise_clip, self.config.noise_clip)
                next_actions = torch.clip(next_samples + e, 0.0, 1.0)
                q1_t = self.q1_target(next_lat, next_actions)
                q2_t = self.q2_target(next_lat, next_actions)
                q_t = torch.min(q1_t, q2_t)
                q_t = transitions["rewards"] + (1.0 - transitions["dones"]) * self.config.gamma * q_t

            # update state encoder
            self.state_encoder_optimizer.zero_grad()
            lat = self.state_encoder.encode(transitions["obs"])
            lat_detached = lat.detach().clone()
            obs_reconstructed = self.state_encoder.decode(lat)
            state_encoder_loss = self.state_encoder.loss(obs_reconstructed, transitions["obs"])
            state_encoder_loss.backward()
            #print(f"{state_encoder_loss=}")
            self.state_encoder_optimizer.step()

            # update critics
            self.q1_optimizer.zero_grad()
            self.q2_optimizer.zero_grad()
            q1_c = self.q1_critic(lat_detached, transitions["actions"])
            q2_c = self.q2_critic(lat_detached, transitions["actions"])
            q1_loss = F.mse_loss(q1_c, q_t)
            q2_loss = F.mse_loss(q2_c, q_t)
            q1_loss.backward()
            q2_loss.backward()
            self.q1_optimizer.step()
            self.q2_optimizer.step()

            if (j + 1) % self.config.update_interval == 0:
                self.p_optimizer.zero_grad()
                actions = self.p_actor(lat_detached)
                q_c = self.q1_critic(lat_detached, actions)
                p_loss = q_c.mean()
                p_loss.backward()
                self.p_optimizer.step()
                self.soft_update()

    def step(self, num_steps=0, evaluate=False):
        done = False
        episode_steps = 0
        episode_reward = 0.0
        obs, _ = self.env.reset()
        while not done:
            obs = obs.to(self.device)
            action = self.select_action(obs, probabilistic=not evaluate)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            if not evaluate:
                self.replay_buffer.add(self._to_tensordict(obs, action, reward, next_obs, done))
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
            "actions": action.astype(np.float32),
            "rewards": np.expand_dims(reward, 0).astype(np.float32),
            "next_obs": next_obs,
            "dones": np.expand_dims(done, 0).astype(np.float32),
        }
        return tensordict.TensorDict(d)

    def train(self, num_episodes, num_steps=0):
        # prefill replay buffer
        missing = max(self.config.batch_size, self.config.replay_buffer_prefill_size) - len(self.replay_buffer)
        if missing > 0:
            render_mode_bak = self.env.render_mode
            self.env.render_mode = None
            obs, _ = self.env.reset()
            for _ in tqdm.tqdm(range(missing)):
                action = self.env.action_space.sample()
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                self.replay_buffer.add(self._to_tensordict(obs, action, reward, next_obs, done))
                if done:
                    obs, _  = self.env.reset()
                else:
                    obs = next_obs
            self.env.render_mode = render_mode_bak
        # train agent
        print("Start training")
        history = {
            "rewards": np.zeros(num_episodes, dtype=np.float32),
            "steps": np.zeros(num_episodes, dtype=np.uint32),
        }
        for episode in range(num_episodes):
            reward, steps, info = self.step(num_steps=num_steps)
            history["rewards"][episode] = reward
            history["steps"][episode] = steps
            print(f"{episode}: {steps=} {reward=}")
        return history

    def evaluate(self):
        reward, steps, info = self.step(evaluate=True)

    def save(self, file=None):
        file = pathlib.Path(file if file is not None else self.config.save_file).resolve()
        checkpoint = dict(
            state_encoder=self.state_encoder.state_dict(),
            p_actor=self.p_actor.state_dict(),
            q1_critic=self.q1_critic.state_dict(),
            q2_critic=self.q2_critic.state_dict(),
            state_encoder_optimizer=self.state_encoder_optimizer.state_dict(),
            p_optimizer=self.p_optimizer.state_dict(),
            q1_optimizer=self.q1_optimizer.state_dict(),
            q2_optimizer=self.q2_optimizer.state_dict(),
        )
        torch.save(checkpoint, file)

    def load(self, file=None):
        file = pathlib.Path(file if file is not None else self.config.save_file).resolve()
        checkpoint = torch.load(file)
        self.state_encoder.load_state_dict(checkpoint["state_encoder"])
        self.p_actor.load_state_dict(checkpoint["p_actor"])
        self.p_target.load_state_dict(checkpoint["p_actor"])
        self.q1_critic.load_state_dict(checkpoint["q1_critic"])
        self.q1_target.load_state_dict(checkpoint["q1_critic"])
        self.q2_critic.load_state_dict(checkpoint["q2_critic"])
        self.q2_target.load_state_dict(checkpoint["q2_critic"])
        self.state_encoder_optimizer.load_state_dict(checkpoint["state_encoder_optimizer"])
        self.p_optimizer.load_state_dict(checkpoint["p_optimizer"])
        self.q1_optimizer.load_state_dict(checkpoint["q1_optimizer"])
        self.q2_optimizer.load_state_dict(checkpoint["q2_optimizer"])

class TD3AgentConfig:
    device = "cuda"
    buffer_size = 3200
    batch_size = 16
    num_updates = 100
    update_interval = 10
    update_noise = 0.2
    noise_clip = 0.5
    replay_buffer_prefill_size = 256
    replay_buffer_dir = pathlib.Path("/DATA1/mlinke/modelling-seminar/td3_buffer")
    embed_size = 2048
    action_size = 4
    max_action = 13.0
    tau = 0.995
    gamma = 0.99
    state_encoder_lr = 0.001
    actor_lr = 0.001
    critic_lr = 0.001
    save_file = pathlib.Path("./chkpts/td3_agent").resolve()

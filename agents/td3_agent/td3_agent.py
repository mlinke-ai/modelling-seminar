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
import datetime
from agents import state_encoder
from agents import agent
import utils
import itertools


class Actor(nn.Module):
    def __init__(self, latent_size, hidden_size, action_size, min_action, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(latent_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, hidden_size)
        self.l5 = nn.Linear(hidden_size, action_size)

        self.min_action = min_action
        self.max_action = max_action

    def forward(self, latent):
        x = F.silu(self.l1(latent))
        x = F.silu(self.l2(x))
        x = F.silu(self.l3(x))
        x = F.silu(self.l4(x))
        x = F.sliu(self.l5(x))
        return self.min_action + (self.max_action - self.min_action) * x


class Critic(nn.Module):
    def __init__(self, latent_size, hidden_size, action_size):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(latent_size + action_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, hidden_size)
        self.l5 = nn.Linear(hidden_size, 1)

    def forward(self, latent, action):
        x = torch.cat((latent, action), dim=-1)
        x = F.silu(self.l1(x))
        x = F.silu(self.l2(x))
        x = F.silu(self.l3(x))
        x = F.silu(self.l4(x))
        x = self.l5(x)
        return x


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
        def _sample_transform(sample):
            sample["obs", "rgbd"] = sample["obs", "rgbd"].to(torch.float32) / 255.0
            sample["next_obs", "rgbd"] = sample["next_obs", "rgbd"].to(torch.float32) / 255.0
            return sample

        self.replay_buffer = data.TensorDictReplayBuffer(
            storage=data.replay_buffers.LazyMemmapStorage(
                max_size=self.config.buffer_size, scratch_dir=self.config.replay_buffer_dir, auto_cleanup=True
            ),
            batch_size=self.config.batch_size,
            transform=_sample_transform,
        )

        # Networks
        self.p_actor_encoder = state_encoder.StateEncoder().to(self.device)
        self.p_target_encoder = state_encoder.StateEncoder().to(self.device)
        self.critic_encoder = state_encoder.StateEncoder().to(self.device)
        self.target_encoder = state_encoder.StateEncoder().to(self.device)
        self.p_actor = Actor(
            self.config.latent_size, self.config.hidden_size, self.config.action_size, self.config.min_action, self.config.max_action
        ).to(self.device)
        self.p_target = Actor(
            self.config.latent_size, self.config.hidden_size, self.config.action_size, self.config.min_action, self.config.max_action
        ).to(self.device)
        self.q1_critic = Critic(self.config.latent_size, self.config.hidden_size, self.config.action_size).to(self.device)
        self.q2_critic = Critic(self.config.latent_size, self.config.hidden_size, self.config.action_size).to(self.device)
        self.q1_target = Critic(self.config.latent_size, self.config.hidden_size, self.config.action_size).to(self.device)
        self.q2_target = Critic(self.config.latent_size, self.config.hidden_size, self.config.action_size).to(self.device)
        self.p_target_encoder.load_state_dict(self.p_actor_encoder.state_dict())
        self.target_encoder.load_state_dict(self.critic_encoder.state_dict())
        self.p_target.load_state_dict(self.p_actor.state_dict())
        self.q1_target.load_state_dict(self.q1_critic.state_dict())
        self.q2_target.load_state_dict(self.q2_critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(itertools.chain(self.p_actor_encoder.parameters(), self.p_actor.parameters()), lr=self.config.actor_lr)
        self.critic_optimizer = optim.Adam(itertools.chain(self.critic_encoder.parameters(), self.q1_critic.parameters(), self.q2_critic.parameters()), lr=self.config.critic_lr)

    def select_action(self, obs, probabilistic=False):
        with torch.no_grad():
            latent = self.actor_encoder(obs.unsqueeze(0).to(torch.float32) / 255.0)
            action = self.p_actor(latent)
            e = torch.randn_like(action) if probabilistic else 0.0
            action = torch.clip(action + e, -self.config.max_action, self.config.max_action)
        return action.squeeze(0).numpy(force=True)

    def soft_update(self):
        for pa, pt in zip(self.p_actor_encoder.parameters(), self.p_target_encoder.parameters()):
            pt.data.mul_(self.config.tau)
            pt.data.add_((1.0 - self.config.tau) * pa.data)
        for pa, pt in zip(self.p_actor.parameters(), self.p_target.parameters()):
            pt.data.mul_(self.config.tau)
            pt.data.add_((1.0 - self.config.tau) * pa.data)
        for pc, pt in zip(self.critic_encoder.parameters(), self.target_encoder.parameters()):
            pt.data.mul_(self.config.tau)
            pt.data.add_((1.0 - self.config.tau) * pc.data)
        for pc, pt in zip(self.q1_critic.parameters(), self.q1_target.parameters()):
            pt.data.mul_(self.config.tau)
            pt.data.add_((1.0 - self.config.tau) * pc.data)
        for pc, pt in zip(self.q2_critic.parameters(), self.q2_target.parameters()):
            pt.data.mul_(self.config.tau)
            pt.data.add_((1.0 - self.config.tau) * pc.data)

    def update(self):
        assert (
            self.config.batch_size <= self.replay_buffer.write_count
        ), "Replay buffer is not sufficiently filled for sampling!"
        for j in range(self.config.num_updates):
            transitions = self.replay_buffer.sample(self.config.batch_size).to(self.device)
            with torch.no_grad():
                next_actor_latent = self.p_target_encoder(transitions["next_obs"])
                next_critic_latent = self.target_encoder(transitions["next_obs"])
                next_samples = self.p_target(next_actor_latent)
                e = torch.clip(
                    torch.randn_like(next_samples) * self.config.update_noise,
                    -self.config.noise_clip,
                    self.config.noise_clip,
                )
                next_actions = torch.clip(next_samples + e, -self.config.max_action, self.config.max_action)
                q1_t = self.q1_target(next_critic_latent, next_actions)
                q2_t = self.q2_target(next_critic_latent, next_actions)
                q_t = torch.min(q1_t, q2_t)
                q_t = (
                    self.config.reward_scale * transitions["rewards"]
                    + (1.0 - transitions["dones"]) * self.config.gamma * q_t
                )

            # update critics
            self.critic_optimizer.zero_grad()
            critic_latent = self.critic_encoder(transitions["obs"])
            q1_c = self.q1_critic(critic_latent, transitions["actions"])
            q2_c = self.q2_critic(critic_latent, transitions["actions"])
            q1_loss = F.mse_loss(q1_c, q_t)
            q2_loss = F.mse_loss(q2_c, q_t)
            # print(f"q_t={q_t.max().item()}, q1_c={q1_c.max().item()}")
            # print(f"q1_loss={q1_loss.item()}, q2_loss={q2_loss.item()}")
            (q1_loss + q2_loss).backward()
            # nn.utils.clip_grad_norm_(self.q1_critic.parameters(), 5.0)
            # nn.utils.clip_grad_norm_(self.q2_critic.parameters(), 5.0)
            self.critic_optimizer.step()

            if (j + 1) % self.config.update_interval == 0:
                for p in self.critic_encoder.parameters():
                    p.requires_grad_(False)
                for p in self.q1_critic.parameters():
                    p.requires_grad_(False)
                self.actor_optimizer.zero_grad()
                actor_latent = self.p_actor_encoder(transitions["obs"])
                critic_latent = self.critic_encoder(transitions["obs"])
                actions = self.p_actor(actor_latent)
                q_c = self.q1_critic(critic_latent, actions)
                p_loss = -q_c.mean()
                # print(f"p_loss={p_loss.item()}")
                p_loss.backward()
                # nn.utils.clip_grad_norm_(self.p_actor.parameters(), 5.0)
                self.actor_optimizer.step()
                for p in self.critic_encoder.parameters():
                    p.requires_grad_(True)
                for p in self.q1_critic.parameters():
                    p.requires_grad_(True)
                self.soft_update()

    def step(self, evaluate=False):
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
                self.replay_buffer.add(self._to_tensordict(obs, action, reward, next_obs, terminated))
                self.update()
            episode_reward += reward
            episode_steps += 1
            obs = next_obs
            # print(action, reward, terminated, truncated, info)
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

    def train(self, train_episodes, eval_episodes=10, eval_interval=10, save_interval=10):
        # prefill replay buffer
        missing = max(self.config.batch_size, self.config.replay_buffer_prefill_size) - len(self.replay_buffer)
        if missing > 0:
            obs, _ = self.env.reset()
            for _ in tqdm.tqdm(range(missing)):
                action = self.env.action_space.sample()
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                self.replay_buffer.add(self._to_tensordict(obs, action, reward, next_obs, terminated))
                if done:
                    obs, _ = self.env.reset()
                else:
                    obs = next_obs
        # train agent
        print("Start training")
        history = {
            "train_rewards": np.zeros(train_episodes, dtype=np.float32),
            "train_steps": np.zeros(train_episodes, dtype=np.uint32),
            "eval_rewards": np.zeros(eval_episodes * (train_episodes // eval_episodes + 1), dtype=np.float32),
            "eval_steps": np.zeros(eval_episodes * (train_episodes // eval_episodes + 1), dtype=np.uint32),
        }
        idx = 0
        start_training = datetime.datetime.now()
        for train_episode in range(train_episodes):
            start_time = datetime.datetime.now()
            train_reward, train_steps, _ = self.step()
            history["train_rewards"][train_episode] = train_reward.astype(np.float32)
            history["train_steps"][train_episode] = train_steps
            if (train_episode + 1) % eval_interval == 0:
                for eval_episode in range(eval_episodes):
                    eval_reward, eval_steps, _ = self.evaluate()
                    history["eval_rewards"][idx] = eval_reward.astype(np.float32)
                    history["eval_steps"][idx] = eval_steps
                    idx += 1
            if (train_episode + 1) % save_interval == 0:
                self.save()
            end_time = datetime.datetime.now()
            print(f"{train_episode}: {train_steps=} {train_reward=} (elapsed time: {end_time - start_time})")
        for eval_episode in range(eval_episodes):
            eval_reward, eval_steps, _ = self.evaluate()
            history["eval_rewards"][idx] = eval_reward.astype(np.float32)
            history["eval_steps"][idx] = eval_steps
            idx += 1
        self.save()
        end_training = datetime.datetime.now()
        print(f"training and evaluating took {end_training - start_training}")
        return history

    def evaluate(self):
        return self.step(evaluate=True)

    def save(self, file=None):
        file = pathlib.Path(file if file is not None else self.config.save_file).resolve()
        checkpoint = dict(
            p_actor=self.p_actor.state_dict(),
            p_actor_encoder=self.p_actor_encoder.state_dict(),
            q1_critic=self.q1_critic.state_dict(),
            q2_critic=self.q2_critic.state_dict(),
            critic_encoder=self.critic_encoder.state_dict(),
            actor_optimizer=self.actor_optimizer.state_dict(),
            critic_optimizer=self.critic_optimizer.state_dict(),
        )
        torch.save(checkpoint, file)

    def load(self, file=None):
        file = pathlib.Path(file if file is not None else self.config.save_file).resolve()
        checkpoint = torch.load(file)
        self.p_actor.load_state_dict(checkpoint["p_actor"])
        self.p_target.load_state_dict(checkpoint["p_actor"])
        self.p_actor_encoder.load_state_dict(checkpoint["p_actor_encoder"])
        self.p_target_encoder.load_state_dict(checkpoint["p_actor_encoder"])
        self.q1_critic.load_state_dict(checkpoint["q1_critic"])
        self.q1_target.load_state_dict(checkpoint["q1_critic"])
        self.q2_critic.load_state_dict(checkpoint["q2_critic"])
        self.q2_target.load_state_dict(checkpoint["q2_critic"])
        self.critic_encoder.load_state_dict(checkpoint["critic_encoder"])
        self.target_encoder.load_state_dict(checkpoint["critic_encoder"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])


class TD3AgentConfig:
    device = "cuda"
    buffer_size = 1000000
    batch_size = 16
    num_updates = 2
    update_interval = 2
    update_noise = 0.1
    noise_clip = 0.5
    replay_buffer_prefill_size = 256
    replay_buffer_dir = pathlib.Path("/DATA1/mlinke/modelling-seminar/td3_buffer")
    latent_size = 512
    hidden_size = 2560
    action_size = 4
    min_action = 0.0
    max_action = 13.0
    tau = 0.995
    gamma = 0.99
    reward_scale = 0.01
    state_encoder_lr = 3e-4
    actor_lr = 3e-4
    critic_lr = 3e-4
    save_file = pathlib.Path("./chkpts/td3_agent").resolve()

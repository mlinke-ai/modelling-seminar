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


class UnfoldLinear(nn.Module):
    def __init__(self, in_features: int, out_features: tuple[int, ...], device=None, dtype=None) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(UnfoldLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        shape = (self.in_features, *self.out_features)
        self.weight = nn.Parameter(torch.empty(shape, **factory_kwargs))
        self.bias = nn.Parameter(torch.empty(self.out_features, **factory_kwargs))
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...c,...cCHW->...CHW", input, self.weight) + self.bias


class StateEncoder(nn.Module):
    def __init__(self):
        super(StateEncoder, self).__init__()

        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=3, padding=2),
            nn.GroupNorm(4, 64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, 64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, 128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, 256),
            nn.SiLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, 512),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.SiLU(),
        )
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=3, padding=2),
            nn.GroupNorm(4, 64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, 64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, 128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, 256),
            nn.SiLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, 512),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.SiLU(),
        )
        self.gyro_encoder = nn.Sequential(
            nn.Linear(3, 16),
            nn.SiLU(inplace=True),
            nn.Linear(16, 16),
            nn.SiLU(inplace=True),
            nn.Linear(16, 16),
            nn.SiLU(inplace=True),
        )
        self.accel_encoder = nn.Sequential(
            nn.Linear(3, 16),
            nn.SiLU(inplace=True),
            nn.Linear(16, 16),
            nn.SiLU(inplace=True),
            nn.Linear(16, 16),
            nn.SiLU(inplace=True),
        )
        self.velo_encoder = nn.Sequential(
            nn.Linear(3, 16),
            nn.SiLU(inplace=True),
            nn.Linear(16, 16),
            nn.SiLU(inplace=True),
            nn.Linear(16, 16),
            nn.SiLU(inplace=True),
        )
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, 16),
            nn.SiLU(inplace=True),
            nn.Linear(16, 16),
            nn.SiLU(inplace=True),
            nn.Linear(16, 16),
            nn.SiLU(inplace=True),
        )
        self.ori_encoder = nn.Sequential(
            nn.Linear(3, 16),
            nn.SiLU(inplace=True),
            nn.Linear(16, 16),
            nn.SiLU(inplace=True),
            nn.Linear(16, 16),
            nn.SiLU(inplace=True),
        )
        self.tilt_encoder = nn.Sequential(
            nn.Linear(3, 16),
            nn.SiLU(inplace=True),
            nn.Linear(16, 16),
            nn.SiLU(inplace=True),
            nn.Linear(16, 16),
            nn.SiLU(inplace=True),
        )
        self.wp_dir_encoder = nn.Sequential(
            nn.Linear(3, 16),
            nn.SiLU(inplace=True),
            nn.Linear(16, 16),
            nn.SiLU(inplace=True),
            nn.Linear(16, 16),
            nn.SiLU(inplace=True),
        )
        self.wp_dist_encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.SiLU(inplace=True),
            nn.Linear(16, 16),
            nn.SiLU(inplace=True),
            nn.Linear(16, 16),
            nn.SiLU(inplace=True),
        )
        self.fused_encoder = nn.Sequential(
            nn.Linear(1152, 1024),
            nn.SiLU(inplace=True),
            nn.Linear(1024, 768),
            #nn.Tanh(),
            nn.LayerNorm(768),
        )

        self.image_decoder = nn.Sequential(
            UnfoldLinear(768, (512, 23, 40)),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(4, 256),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(4, 128),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(4, 64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=(0, 2)),
            nn.GroupNorm(4, 64),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=5, stride=3, padding=2, output_padding=2),
            nn.Sigmoid(),
        )
        self.depth_decoder = nn.Sequential(
            UnfoldLinear(768, (512, 23, 40)),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(4, 256),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(4, 128),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(4, 64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=(0, 2)),
            nn.GroupNorm(4, 64),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=5, stride=3, padding=2, output_padding=2),
            nn.ReLU(),
        )
        self.gyro_decoder = nn.Sequential(
            nn.Linear(768, 16),
            nn.SiLU(inplace=True),
            nn.Linear(16, 16),
            nn.SiLU(inplace=True),
            nn.Linear(16, 3),
        )
        self.accel_decoder = nn.Sequential(
            nn.Linear(768, 16),
            nn.SiLU(inplace=True),
            nn.Linear(16, 16),
            nn.SiLU(inplace=True),
            nn.Linear(16, 3),
        )
        self.velo_decoder = nn.Sequential(
            nn.Linear(768, 16),
            nn.SiLU(inplace=True),
            nn.Linear(16, 16),
            nn.SiLU(inplace=True),
            nn.Linear(16, 3),
        )
        self.pos_decoder = nn.Sequential(
            nn.Linear(768, 16),
            nn.SiLU(inplace=True),
            nn.Linear(16, 16),
            nn.SiLU(inplace=True),
            nn.Linear(16, 3),
        )
        self.ori_decoder = nn.Sequential(
            nn.Linear(768, 16),
            nn.SiLU(inplace=True),
            nn.Linear(16, 16),
            nn.SiLU(inplace=True),
            nn.Linear(16, 3),
            nn.Tanh(),
        )
        self.tilt_decoder = nn.Sequential(
            nn.Linear(768, 16),
            nn.SiLU(inplace=True),
            nn.Linear(16, 16),
            nn.SiLU(inplace=True),
            nn.Linear(16, 3),
            nn.Tanh(),
        )
        self.wp_dir_decoder = nn.Sequential(
            nn.Linear(768, 16),
            nn.SiLU(inplace=True),
            nn.Linear(16, 16),
            nn.SiLU(inplace=True),
            nn.Linear(16, 3),
            nn.Tanh(),
        )
        self.wp_dist_decoder = nn.Sequential(
            nn.Linear(768, 16),
            nn.SiLU(inplace=True),
            nn.Linear(16, 16),
            nn.SiLU(inplace=True),
            nn.Linear(16, 1),
            nn.ReLU(),
        )

    def forward(self, obs):
        enc = self.encode(obs)
        dec = self.decode(enc)
        return dec

    def encode(self, obs):
        enc_image = self.image_encoder(obs["image"])
        enc_depth = self.depth_encoder(torch.log_(1.0 + obs["depth"]))
        enc_gyro = self.gyro_encoder(obs["gyro"])
        enc_accel = self.accel_encoder(obs["accel"])
        enc_velo = self.velo_encoder(obs["velo"])
        enc_pos = self.pos_encoder(obs["pos"])
        enc_ori = self.ori_encoder(obs["ori"])
        enc_tilt = self.tilt_encoder(obs["tilt"])
        enc_wp_dir = self.wp_dir_encoder(obs["wp_dir"])
        enc_wp_dist = self.wp_dist_encoder(obs["wp_dist"])
        enc_fused = torch.cat(
            (enc_image, enc_depth, enc_gyro, enc_accel, enc_velo, enc_pos, enc_ori, enc_tilt, enc_wp_dir, enc_wp_dist),
            dim=-1,
        )
        return self.fused_encoder(enc_fused)

    def decode(self, lat):
        dec_image = self.image_decoder(lat)
        dec_depth = self.depth_decoder(lat)
        dec_gyro = self.gyro_decoder(lat)
        dec_accel = self.accel_decoder(lat)
        dec_velo = self.velo_decoder(lat)
        dec_pos = self.pos_decoder(lat)
        dec_ori = self.ori_decoder(lat)
        dec_tilt = self.tilt_decoder(lat)
        dec_wp_dir = self.wp_dir_decoder(lat)
        dec_wp_dist = self.wp_dist_decoder(lat)
        return tensordict.TensorDict(
            {
                "image": dec_image,
                "depth": dec_depth,
                "gyro": dec_gyro,
                "accel": dec_accel,
                "velo": dec_velo,
                "pos": dec_pos,
                "ori": dec_ori,
                "tilt": dec_tilt,
                "wp_dir": dec_wp_dir,
                "wp_dist": dec_wp_dist,
            }
        )

    def loss(self, prediction, target):
        loss = F.mse_loss(prediction["image"], target["image"])
        loss += F.mse_loss(prediction["depth"], target["depth"])
        loss += F.mse_loss(prediction["gyro"], target["gyro"])
        loss += F.mse_loss(prediction["accel"], target["accel"])
        loss += F.mse_loss(prediction["velo"], target["velo"])
        loss += F.mse_loss(prediction["pos"], target["pos"])
        loss += F.mse_loss(prediction["ori"], target["ori"])
        loss += F.mse_loss(prediction["tilt"], target["tilt"])
        loss += F.mse_loss(prediction["wp_dir"], target["wp_dir"])
        loss += F.mse_loss(prediction["wp_dist"], target["wp_dist"])
        return loss


class Actor(nn.Module):
    def __init__(self, embed_size, action_size, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(768, 512)
        self.l2 = nn.Linear(512, 128)
        self.mu = nn.Linear(128, 4)
        self.sd = nn.Linear(128, 4)

        self.max_action = max_action

    def forward(self, embed):
        x = F.silu(self.l1(embed))
        x = F.silu(self.l2(x))
        mu = self.mu(x)
        sd = F.softplus(self.sd(x)) + utils.SMALL
        return mu, sd

    def sample(self, embed, probabilistic=True):
        mu, sd = self.forward(embed)
        if probabilistic:
            normal_dist = dist.Normal(mu, sd)
            sample = normal_dist.rsample()
            log_prob = normal_dist.log_prob(sample).sum(-1)
            # log_prob -= torch.log(1.0 - action.pow(2) + utils.SMALL).sum(-1)
            action = F.sigmoid(sample)
        else:
            action = F.sigmoid(mu)
            log_prob = None
        #action *= self.max_action
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, embed_size, action_size):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(772, 512)
        self.l2 = nn.Linear(512, 128)
        self.l3 = nn.Linear(128, 1)

    def forward(self, embed, action):
        x = torch.cat((embed, action), dim=-1)
        x = F.silu(self.l1(x))
        x = F.silu(self.l2(x))
        x = self.l3(x)
        return x


class SACAgent:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.steps = 0
        # Device management
        if torch.cuda.is_available():
            self.device = torch.device(self.config.device)
        else:
            self.device = torch.device("cpu")

        # Replay Buffer
        self.replay_buffer = data.TensorDictReplayBuffer(
            storage=data.replay_buffers.LazyMemmapStorage(
                max_size=self.config.buffer_size, scratch_dir=self.config.replay_buffer_dir, auto_cleanup=True
            ),
            batch_size=self.config.batch_size,
        )  # prefetch: int (prefetch N batches), transform: Callable (convert uint8[0; 255] images to float32[0.0; 1.0])

        # Networks
        self.state_encoder = StateEncoder().to(self.device)
        self.actor = Actor(self.config.embed_size, self.config.action_size, self.config.max_action).to(self.device)
        self.q1_critic = Critic(self.config.embed_size, self.config.action_size).to(self.device)
        self.q2_critic = Critic(self.config.embed_size, self.config.action_size).to(self.device)
        self.q1_target = Critic(self.config.embed_size, self.config.action_size).to(self.device)
        self.q2_target = Critic(self.config.embed_size, self.config.action_size).to(self.device)
        self.q1_target.load_state_dict(self.q1_critic.state_dict())
        self.q2_target.load_state_dict(self.q2_critic.state_dict())

        # Optimizers
        self.state_encoder_optimizer = optim.SGD(
            self.state_encoder.parameters(), lr=self.config.state_encoder_lr
        )
        self.actor_optimizer = optim.SGD(self.actor.parameters(), lr=self.config.actor_lr, maximize=True)
        self.q1_optimizer = optim.SGD(self.q1_critic.parameters(), lr=self.config.critic_lr)
        self.q2_optimizer = optim.SGD(self.q2_critic.parameters(), lr=self.config.critic_lr)

        # Entropy coefficient
        self.target_entropy = float(-self.config.action_size)
        self.alpha = torch.tensor(self.config.alpha_init, dtype=torch.float32, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.SGD([self.alpha], lr=self.config.alpha_lr)

        # reset gradients before first update
        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        self.alpha_optimizer.zero_grad()

    def select_action(self, obs, probabilistic=False):
        with torch.no_grad():
            lat = self.state_encoder.encode(obs.unsqueeze(0))
            action, _ = self.actor.sample(lat, probabilistic=probabilistic)
        return action.cpu().data.numpy().flatten()

    def soft_update(self):
        for (
            pc,
            pt,
        ) in zip(self.q1_critic.parameters(), self.q1_target.parameters()):
            pt.data.mul_(self.config.tau)
            pt.data.add_((1.0 - self.config.tau) * pc.data)
        for (
            pc,
            pt,
        ) in zip(self.q2_critic.parameters(), self.q2_target.parameters()):
            pt.data.mul_(self.config.tau)
            pt.data.add_((1.0 - self.config.tau) * pc.data)

    def update(self):
        assert (
            self.config.batch_size <= self.replay_buffer.write_count
        ), "Replay buffer is not sufficiently filled for sampling!"
        trajectory = self.replay_buffer.sample(self.config.batch_size).to(self.device)
        with torch.no_grad():
            next_lat = self.state_encoder.encode(trajectory["next_obs"])
            next_actions, next_log_probs = self.actor.sample(next_lat)
            q1_t = self.q1_target(next_lat, next_actions)
            q2_t = self.q2_target(next_lat, next_actions)
            q_t = torch.min(q1_t, q2_t) - torch.unsqueeze(self.alpha * next_log_probs, -1)
            q_t = trajectory["rewards"] + (1.0 - trajectory["dones"]) * self.config.gamma * q_t

        # update state encoder
        self.state_encoder_optimizer.zero_grad()
        lat = self.state_encoder.encode(trajectory["obs"])
        lat_detached = lat.detach().clone()
        obs_reconstructed = self.state_encoder.decode(lat)
        state_encoder_loss = self.state_encoder.loss(obs_reconstructed, trajectory["obs"])
        state_encoder_loss.backward()
        #print(f"{state_encoder_loss=}")
        self.state_encoder_optimizer.step()

        # update critics
        q1_c = self.q1_critic(lat_detached, trajectory["actions"])
        q2_c = self.q2_critic(lat_detached, trajectory["actions"])
        q1_loss = F.mse_loss(q1_c, q_t)
        q2_loss = F.mse_loss(q2_c, q_t)
        q1_loss.backward()
        q2_loss.backward()

        # update actor
        new_actions, new_log_probs = self.actor.sample(lat_detached)
        a = new_actions.detach().clone()
        with torch.no_grad():
            new_q1_c = self.q1_critic(lat_detached, a)
            new_q2_c = self.q2_critic(lat_detached, a)
            new_q_c = torch.min(new_q1_c, new_q2_c)
        actor_loss = (new_q_c - torch.unsqueeze(self.alpha * new_log_probs, -1)).mean()
        actor_loss.backward()

        # update entropy
        alpha_loss = -(self.alpha * (new_log_probs + self.target_entropy).detach()).mean()
        alpha_loss.backward()

        # update targets
        if self.steps % self.config.update_interval == 0:
            nn.utils.clip_grad_norm_(self.q1_critic.parameters(), 1.0)
            nn.utils.clip_grad_norm_(self.q2_critic.parameters(), 1.0)
            self.q1_optimizer.step()
            self.q2_optimizer.step()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()
            self.alpha_optimizer.step()
            self.q1_optimizer.zero_grad()
            self.q2_optimizer.zero_grad()
            self.actor_optimizer.zero_grad()
            self.alpha_optimizer.zero_grad()
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
                self.replay_buffer.add(self._to_tensordict(obs, action, reward, next_obs, done))
                self.update()
            obs = next_obs
            episode_reward += reward
            episode_steps += 1
            self.steps += 1
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

    def train(self, num_episodes):
        # fill replay buffer with random transitions
        print("Prefill replay buffer")
        render_mode_bak = self.env.render_mode
        self.env.render_mode = None
        obs, _ = self.env.reset()
        p_bar = tqdm.tqdm(total=max(self.config.batch_size, self.config.replay_buffer_prefill_size))
        while self.replay_buffer.write_count < max(self.config.batch_size, self.config.replay_buffer_prefill_size):
            action = self.env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            self.replay_buffer.add(self._to_tensordict(obs, action, reward, next_obs, done))
            if done:
                obs, _ = self.env.reset()
            else:
                obs = next_obs
            p_bar.update()
        p_bar.close()
        self.env.render_mode = render_mode_bak
        # train agent
        print("Start training")
        history = {
            "rewards": np.zeros(num_episodes, dtype=np.float32),
            "steps": np.zeros(num_episodes, dtype=np.uint32),
        }
        for episode in range(num_episodes):
            reward, steps, info = self.step()
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
            actor=self.actor.state_dict(),
            q1_critic=self.q1_critic.state_dict(),
            q2_critic=self.q2_critic.state_dict(),
            alpha=self.alpha,
            state_encoder_optimizer=self.state_encoder_optimizer.state_dict(),
            actor_optimizer=self.actor_optimizer.state_dict(),
            q1_optimizer=self.q1_optimizer.state_dict(),
            q2_optimizer=self.q2_optimizer.state_dict(),
            alpha_optimizer=self.alpha_optimizer.state_dict(),
        )
        torch.save(checkpoint, file)

    def load(self, file=None):
        file = pathlib.Path(file if file is not None else self.config.save_file).resolve()
        checkpoint = torch.load(file)
        self.state_encoder.load_state_dict(checkpoint["state_encoder"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.q1_critic.load_state_dict(checkpoint["q1_critic"])
        self.q1_target.load_state_dict(checkpoint["q1_critic"])
        self.q2_critic.load_state_dict(checkpoint["q2_critic"])
        self.q2_target.load_state_dict(checkpoint["q2_critic"])
        self.alpha = checkpoint["alpha"]
        self.state_encoder_optimizer.load_state_dict(checkpoint["state_encoder_optimizer"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.q1_optimizer.load_state_dict(checkpoint["q1_optimizer"])
        self.q2_optimizer.load_state_dict(checkpoint["q2_optimizer"])
        self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])


class SACAgentConfig:
    device = "cuda"
    buffer_size = 3200
    batch_size = 16
    update_interval = 1
    replay_buffer_prefill_size = 256
    replay_buffer_dir = pathlib.Path("/DATA1/mlinke/modelling-seminar/sac_buffer")
    embed_size = 2048
    action_size = 4
    max_action = 13.0
    tau = 0.995
    gamma = 0.99
    alpha_init = 0.2
    state_encoder_lr = 0.001
    actor_lr = 0.001
    critic_lr = 0.001
    alpha_lr = 0.001
    save_file = pathlib.Path("./chkpts/sac_agent").resolve()

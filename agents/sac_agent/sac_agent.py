import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import torch.nn.functional as F
import torchrl
from torchrl import data
import pathlib
import tensordict
import numpy as np
import utils
import tqdm


class StateEmbedding(nn.Module):
    def __init__(self, embed_size):
        super(StateEmbedding, self).__init__()

        self.relu = nn.ReLU()
        self.adp_pool = nn.AdaptiveMaxPool2d(16)
        self.c1 = nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1)
        self.c2 = nn.Conv2d(16, 32, kernel_size=7, stride=7, padding=3)
        self.l1 = nn.Linear(8192, 8192)
        self.l2 = nn.Linear(8214, 2048)

    def forward(self, obs):
        e = self.relu(self.c1(obs["rgbd"]))
        e = self.adp_pool(self.relu(self.c2(e)))
        e = e.flatten(start_dim=-3, end_dim=-1)
        e = self.relu(self.l1(e))
        e = torch.cat(
            (e, obs["gyro"], obs["acceleration"], obs["velocity"], obs["position"], obs["orientation"], obs["tilting"], obs["wp_direction"], obs["wp_distance"]), dim=-1
        )
        e = self.relu(self.l2(e))
        return e


class Actor(nn.Module):
    def __init__(self, embed_size, action_size, max_action):
        super(Actor, self).__init__()

        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.l1 = nn.Linear(2048, 512)
        self.l2 = nn.Linear(512, 128)
        self.mu = nn.Linear(128, 4)
        self.sd = nn.Linear(128, 4)

        self.max_action = max_action

    def forward(self, embed):
        x = self.relu(self.l1(embed))
        x = self.relu(self.l2(x))
        mu = self.mu(x)
        sd = self.softplus(self.sd(x)) + utils.SMALL
        return mu, sd

    def sample(self, embed, probabilistic=True):
        mu, sd = self.forward(embed)
        if probabilistic:
            normal_dist = dist.Normal(mu, sd)
            sample = normal_dist.rsample()
            log_prob = normal_dist.log_prob(sample).sum(-1)
            #log_prob -= torch.log(1.0 - action.pow(2) + utils.SMALL).sum(-1)
            action = F.sigmoid(sample)
        else:
            action = F.sigmoid(mu)
            log_prob = None
        action *= self.max_action
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, embed_size, action_size):
        super(Critic, self).__init__()

        self.relu = nn.ReLU()
        self.l1 = nn.Linear(embed_size + action_size, 512)
        self.l2 = nn.Linear(512, 128)
        self.l3 = nn.Linear(128, 1)

    def forward(self, embed, action):
        x = torch.cat((embed, action), dim=-1)
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.l3(x)
        return x


class SACAgent:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        # Device management
        if torch.cuda.is_available():
            self.device = torch.device(self.config.device)
        else:
            self.device = torch.device("cpu")

        # Replay Buffer
        self.replay_buffer = data.TensorDictReplayBuffer(
            storage=data.replay_buffers.LazyMemmapStorage(
                max_size=self.config.buffer_size, scratch_dir="./buffer", auto_cleanup=True
            ),
            batch_size=self.config.batch_size,
        )  # prefetch: int (prefetch N batches), transform: Callable (convert uint8[0; 255] images to float32[0.0; 1.0])

        # Networks
        self.state_embedding = StateEmbedding(self.config.embed_size).to(self.device)
        self.actor = Actor(self.config.embed_size, self.config.action_size, self.config.max_action).to(self.device)
        self.q1_critic = Critic(self.config.embed_size, self.config.action_size).to(self.device)
        self.q2_critic = Critic(self.config.embed_size, self.config.action_size).to(self.device)
        self.q1_target = Critic(self.config.embed_size, self.config.action_size).to(self.device)
        self.q2_target = Critic(self.config.embed_size, self.config.action_size).to(self.device)
        self.q1_target.load_state_dict(self.q1_critic.state_dict())
        self.q2_target.load_state_dict(self.q2_critic.state_dict())

        # Optimizers
        self.state_embedding_optimizer = optim.Adam(
            self.state_embedding.parameters(), lr=self.config.state_embedding_lr
        )
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.actor_lr, maximize=True)
        self.q1_optimizer = optim.Adam(self.q1_critic.parameters(), lr=self.config.critic_lr)
        self.q2_optimizer = optim.Adam(self.q2_critic.parameters(), lr=self.config.critic_lr)

        # Entropy coefficient
        self.target_entropy = float(-self.config.action_size)
        self.alpha = torch.tensor(self.config.alpha_init, dtype=torch.float32, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.alpha], lr=self.config.alpha_lr)

    def select_action(self, obs, probabilistic=False):
        with torch.no_grad():
            embed = self.state_embedding(obs)
            action, _ = self.actor.sample(embed, probabilistic=probabilistic)
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
            next_embeddings = self.state_embedding(trajectory["next_obs"])
            next_actions, next_log_probs = self.actor.sample(next_embeddings)
            q1_t = self.q1_target(next_embeddings, next_actions)
            q2_t = self.q2_target(next_embeddings, next_actions)
            q_t = torch.min(q1_t, q2_t) - torch.unsqueeze(self.alpha * next_log_probs, -1)
            q_t = trajectory["rewards"] + (1.0 - trajectory["dones"]) * self.config.gamma * q_t

        # reset gradients
        self.state_embedding_optimizer.zero_grad()
        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()

        embeddings = self.state_embedding(trajectory["obs"])

        # update critics
        q1_c = self.q1_critic(embeddings, trajectory["actions"])
        q2_c = self.q2_critic(embeddings, trajectory["actions"])
        q1_loss = F.mse_loss(q1_c, q_t)
        q2_loss = F.mse_loss(q2_c, q_t)
        print(f"{q1_loss=}; {q2_loss=}")

        q1_loss.backward(retain_graph=True)
        self.q1_optimizer.step()

        q2_loss.backward(retain_graph=True)
        self.q2_optimizer.step()

        # update actor
        new_actions, new_log_probs = self.actor.sample(embeddings)
        a = new_actions.detach().clone()
        new_q1_c = self.q1_critic(embeddings, a)
        new_q2_c = self.q2_critic(embeddings, a)
        #new_q1_c = torch.tensor(1.0, requires_grad=True, device=self.device)
        #new_q2_c = torch.tensor(0.0, requires_grad=True, device=self.device)
        new_q_c = torch.min(new_q1_c, new_q2_c)
        actor_loss = F.mse_loss(new_q_c, torch.unsqueeze(self.alpha * new_log_probs, -1))
        print(f"{actor_loss=}")

        actor_loss.backward()
        self.actor_optimizer.step()

        for p in self.state_embedding.parameters():
            p.grad /= 3.0
        self.state_embedding_optimizer.step()

        # update entropy
        alpha_loss = -(self.alpha * (new_log_probs + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # update targets
        self.soft_update()

    def step(self, evaluate=False):
        done = False
        episode_steps = 0
        episode_reward = 0.0
        observation, _ = self.env.reset()
        while not done:
            observation = observation.to(self.device)
            action = self.select_action(observation, probabilistic=not evaluate)
            next_observation, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            if not evaluate:
                self.replay_buffer.add(self._to_tensordict(observation, action, reward, next_observation, done))
                self.update()
            observation = next_observation
            episode_reward += reward
            episode_steps += 1
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
        observation, _ = self.env.reset()
        p_bar = tqdm.tqdm(total=max(self.config.batch_size, self.config.replay_buffer_prefill_size))
        while self.replay_buffer.write_count < max(self.config.batch_size, self.config.replay_buffer_prefill_size):
            action = self.env.action_space.sample()
            next_observation, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            self.replay_buffer.add(self._to_tensordict(observation, action, reward, next_observation, done))
            if done:
                observation, _ = self.env.reset()
            else:
                observation = next_observation
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
            state_embedding=self.state_embedding.state_dict(),
            actor=self.actor.state_dict(),
            q1_critic=self.q1_critic.state_dict(),
            q2_critic=self.q2_critic.state_dict(),
            alpha=self.alpha,
            state_embedding_optimizer=self.state_embedding_optimizer.state_dict(),
            actor_optimizer=self.actor_optimizer.state_dict(),
            q1_optimizer=self.q1_optimizer.state_dict(),
            q2_optimizer=self.q2_optimizer.state_dict(),
            alpha_optimizer=self.alpha_optimizer.state_dict(),
        )
        torch.save(checkpoint, file)

    def load(self, file=None):
        file = pathlib.Path(file if file is not None else self.config.save_file).resolve()
        checkpoint = torch.load(file)
        self.state_embedding.load_state_dict(checkpoint["state_embedding"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.q1_critic.load_state_dict(checkpoint["q1_critic"])
        self.q1_target.load_state_dict(checkpoint["q1_critic"])
        self.q2_critic.load_state_dict(checkpoint["q2_critic"])
        self.q2_target.load_state_dict(checkpoint["q2_critic"])
        self.alpha = checkpoint["alpha"]
        self.state_embedding_optimizer.load_state_dict(checkpoint["state_embedding_optimizer"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.q1_optimizer.load_state_dict(checkpoint["q1_optimizer"])
        self.q2_optimizer.load_state_dict(checkpoint["q2_optimizer"])
        self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])


class SACAgentConfig:
    device = "cuda"
    buffer_size = 1e6
    batch_size = 32
    replay_buffer_prefill_size = 256
    replay_buffer_dir = pathlib.Path("/HOME1/students/mlinke")
    embed_size = 2048
    action_size = 4
    max_action = 13.0
    tau = 0.995
    gamma = 0.99
    alpha_init = 0.2
    state_embedding_lr = 3e-4
    actor_lr = 3e-4
    critic_lr = 3e-4
    alpha_lr = 3e-4
    save_file = pathlib.Path("./chkpts/sac_agent").resolve()

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

        self.encoder = nn.Sequential(
            nn.Conv2d(26, 128, kernel_size=3, stride=3, padding=1),
            nn.GroupNorm(16, 128),
            nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=(3, 1)),
            nn.GroupNorm(16, 256),
            nn.SiLU(),
            nn.Conv2d(256, 384, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, 384),
            nn.SiLU(),
            nn.Conv2d(384, 512, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, 512),
            nn.SiLU(),
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, 1024),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1024, 768),
        )

    def forward(self, obs):
        packed = self.pack(obs)
        encoded = self.encoder(packed)
        return encoded

    def pack(self, obs):
        ndim = max(v.ndim for v in obs.values())
        shape = np.zeros(ndim, dtype=np.int64)
        batch_dims = len(obs.batch_size)
        for k in obs.keys():
            s = np.array(obs[k].shape)
            shape[:s.size] += s
        shape[:batch_dims] = np.array(obs.batch_size)
        shape = (*(obs.batch_size), 26, 540, 960)
        packed = torch.zeros(shape, dtype=obs.dtype, device=obs.device)
        packed[..., :3, :, :] = obs["image"]
        packed[..., 3:4, :, :] = obs["depth"]
        packed[..., 4:7, :, :] = obs["gyro"][..., :, None, None].repeat(1, 1, *(shape[-2:]))
        packed[..., 7:10, :, :] = obs["accel"][..., :, None, None].repeat(1, 1, *(shape[-2:]))
        packed[..., 10:13, :, :] = obs["velo"][..., :, None, None].repeat(1, 1, *(shape[-2:]))
        packed[..., 13:16, :, :] = obs["pos"][..., :, None, None].repeat(1, 1, *(shape[-2:]))
        packed[..., 16:19, :, :] = obs["ori"][..., :, None, None].repeat(1, 1, *(shape[-2:]))
        packed[..., 19:22, :, :] = obs["tilt"][..., :, None, None].repeat(1, 1, *(shape[-2:]))
        packed[..., 22:25, :, :] = obs["wp_dir"][..., :, None, None].repeat(1, 1, *(shape[-2:]))
        packed[..., 25:, :, :] = obs["wp_dist"][..., :, None, None].repeat(1, 1, *(shape[-2:]))
        return packed

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


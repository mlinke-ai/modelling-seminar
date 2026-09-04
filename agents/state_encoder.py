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


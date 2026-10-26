import torch
import torch.nn as nn
import torchvision


class StateEncoder(nn.Module):
    def __init__(self):
        super(StateEncoder, self).__init__()

        self.img_net = torchvision.models.squeezenet1_0(weights=None)
        _ = self.img_net.features.pop(0)
        self.img_net.features.insert(0, nn.Conv2d(4, 96, kernel_size=7, stride=2))
        self.img_net.classifier = nn.AdaptiveAvgPool2d((1, 1))

        self.aux_net = nn.Sequential(
            nn.LayerNorm(19),
            nn.Linear(19, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 64),
        )

        self.encoder = nn.Sequential(nn.Dropout(p=0.2, inplace=True), nn.Linear(1088, 512))

    def forward(self, obs):
        imgs = torch.tensor_split(obs["rgbd"], 2, dim=1)
        img_feats = torch.cat(tuple(map(self.img_net, imgs)), dim=-1)
        aux_feats = self.aux_net(obs["aux_dat"])
        features = torch.cat((img_feats, aux_feats), dim=-1)
        return self.encoder(features)

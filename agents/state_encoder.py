import torch
import torch.nn as nn
import torchvision


class StateEncoder(nn.Module):
    def __init__(self):
        super(StateEncoder, self).__init__()

        self.rgb_net = torchvision.models.squeezenet1_0(weights=None)
        _ = self.rgb_net.features.pop(0)
        self.rgb_net.features.insert(0, nn.Conv2d(6, 96, kernel_size=7, stride=2))
        self.rgb_net.classifier = nn.AdaptiveAvgPool2d((1, 1))

        self.dep_net = torchvision.models.squeezenet1_0(weights=None)
        _ = self.dep_net.features.pop(0)
        self.dep_net.features.insert(0, nn.Conv2d(2, 96, kernel_size=7, stride=2))
        self.dep_net.classifier = nn.AdaptiveAvgPool2d((1, 1))

        self.aux_net = nn.Sequential(
            nn.Linear(21, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 64),
        )

        self.encoder = nn.Sequential(nn.Dropout(p=0.2, inplace=True), nn.Linear(1088, 512))

    def forward(self, obs):
        img_feats = self.rgb_net(obs["img_dat", "rgb"])
        dep_feats = self.dep_net(obs["img_dat", "dep"])
        aux_feats = self.aux_net(obs["aux_dat"])
        features = torch.cat((img_feats, dep_feats, aux_feats), dim=-1)
        return self.encoder(features)

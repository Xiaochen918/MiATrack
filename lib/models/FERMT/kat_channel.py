import torch
import torch.nn as nn

from kat_rational import KAT_Group


class ChannelKat(nn.Module):
    def __init__(self, channel):
        super(ChannelKat, self).__init__()

        device = "cuda"
        # device = 'cpu'

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        
        mlp_ratio = 4
        self.fc1 = nn.Linear(channel, channel*mlp_ratio, bias=True)
        self.fc2 = nn.Linear(channel*mlp_ratio, channel, bias=True)
        self.act_kan1 = KAT_Group(mode="relu", device=device)
        self.act_kan2 = KAT_Group(mode="relu", device=device)


    def forward(self, x, H_s):  # x: [B, N, C]
        B, _, C = x.size()

        x = x.transpose(-2,-1).reshape(B,C,H_s,H_s)
        y_avg = self.avg_pool(x).view(B, C)

        x_ch = self.fc1(self.act_kan1(y_avg))
        x_ch = self.fc2(self.act_kan2(x_ch))
        x_ch = torch.sigmoid(x_ch).view(B,C,1,1)
    
        x = x * x_ch.expand_as(x)
        x = x.reshape(B,C,-1).transpose(-2,-1)

        return x
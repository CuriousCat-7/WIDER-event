import torch
from torch import nn
import torchvision.models as models

class Top1000Model(nn.Module):
    def __init__(self,):
        super(Top1000Model, self).__init__()
        self.resnet = models.resnet101(pretrained=True)
        self.fc1 = nn.ModuleList([nn.Linear(1000,500), nn.Linear(1000,500)])
        self.fc2 = nn.Linear(1000,61)
    def forward(self,x):
        for child in list(self.resnet.children())[:-1]:
            x = child(x)
        x = x.mean(-1).mean(-1)
        a, a_idx = x.topk(k=1000, dim=-1)
        a_idx = a_idx.float()
        x = torch.cat([self.fc1[0](a), self.fc1[1](a_idx)],dim=-1)
        x = self.fc2(x)
        return x

class TwoLinearModel(nn.Module):
    def __init__(self,):
        super(TwoLinearModel, self).__init__()
        self.resnet = models.resnet101(pretrained=True)
        self.fc1 = nn.Linear(2048,1000)
        self.fc2 = nn.Linear(1000,61)
    def forward(self,x):
        for child in list(self.resnet.children())[:-1]:
            x = child(x)
        x = x.mean(-1).mean(-1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

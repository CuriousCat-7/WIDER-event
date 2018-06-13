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

class GateModel(nn.Module):
    def __init__(self):
        super(GateModel,self).__init__()
        self.resnet = self.get_resnet()
        self.gate = nn.Sequential(nn.Linear(2048,100),
                                  nn.ReLU(),
                                  nn.Linear(100,3),
                                  nn.Softmax(dim=-1)
                                  )
        self.fc1 = nn.Sequential(nn.Linear(2048, 100),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(100,61)
                                 )
        self.fc2 = nn.Sequential(nn.Linear(2048, 100),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(100,61)
                                 )
        self.fc2 = nn.Sequential(nn.Linear(2048, 100),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(100,61)
                                 )
    def forward(self,x):
        x = self.resnet(x)
        feature = x.mean(-1).mean(-1)
        gate_out = self.gate(feature)
        fc1_out = self.fc1(feature)
        fc2_out = self.fc2(feature)
        fc3_out = self.fc3(feature)
        fc_out = torch.stack(fc1_out.data, fc2_out.data, fc3_out.data, dim=-1) # shape N, 61, 3
        gate_out = gate_out.unsqueeze(1) # shape N, 1, 3
        gate_output = fc_out.mul(gate_out).sum(-1) 
        if not self.training:
            return gate_output
        # else
        return gate_output, fc1_out, fc2_out, fc3_out
        

    def train_parameters(self):
        mo = nn.ModuleList([self.fc1, self.fc2, self.fc3, self.gate])
        return mo.parameters()
    def finetue_parameters(self):
        return self.resnet.parameters()

    def get_resnet(self):
        resnet = models.resnet101(pretrained=True)
        net = []
        net = list(resnet.children())[:-1]
        return nn.Sequential(*net)
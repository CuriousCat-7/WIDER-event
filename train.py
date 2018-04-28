from data import ImageFilelist
from torch import nn
import torch
from torchvision import transforms
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable

class Model(nn.Module):
    def __init__(self,):
        super(Model, self).__init__()
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

config = {}
config['batch_size']=32
config['gpu']=1

trans = transforms.Compose(
    [
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
    ]
)
dataSet = ImageFilelist(root = './data', flist='/data/limingyao/eventRe/data/WIDER_v0.1/train.lst', transform = trans)
testdataSet = ImageFilelist(root = './data', flist='/data/limingyao/eventRe/data/WIDER_v0.1/test.lst', transform = trans)
dataloader = torch.utils.data.DataLoader(dataSet, batch_size = config['batch_size'], shuffle=True,)
testdataloader = torch.utils.data.DataLoader(testdataSet, batch_size = config['batch_size'], shuffle=True,)
cerit = F.cross_entropy
net = Model()
print (net)
net.cuda(config['gpu'])
train_net = nn.ModuleList([net.fc1, net.fc2])
optim = torch.optim.Adam( train_net.parameters(),0.0001, )
optim1 = torch.optim.Adam(nn.ModuleList(list(net.resnet.children())[:-1] ).parameters(),0.000001, )

for epoch in range(25):
    # train
#    net.train()
#    for idx, (img, label) in enumerate(dataloader):
#        img, label = Variable(img).cuda(config['gpu']), Variable(label).cuda(config['gpu'])
#        pre = net(img)
#        loss =cerit(pre, label)
#        optim.zero_grad()
#        optim1.zero_grad()
#        loss.backward()
#        optim.step()
#        optim1.step()
#        if idx%50==0:
#            print ("TRAIN in epoch {} , step {}, loss = {}".format(epoch, idx, loss.data.cpu()[0]))
#
    # test
    net.eval()
    losses = []
    acces = []
    for idx, (img, label) in enumerate(testdataloader):
        img, label = Variable(img,volatile=True).cuda(config['gpu']), Variable(label,volatile=True).cuda(config['gpu'])
        pre = net(img)
        loss =cerit(pre, label)
        acc = pre.topk(k=1,dim=1)[-1].squeeze(-1).eq(label).float().mean().mul(100)# b 64
        losses.append(loss.data.cpu())
        acces.append(acc.data.cpu())

    loss = torch.stack(losses).mean()
    acc = torch.stack(acces).mean()
    print ("TEST in epoch{}, loss = {}, acc = {}%".format(epoch, loss, acc))
    torch.save(net.state_dict(), 'model.save')



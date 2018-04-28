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
        self.fc1 = nn.Linear(2048,1000)
        self.fc2 = nn.Linear(1000,61)
    def forward(self,x):
        for child in list(self.resnet.children())[:-1]:
            x = child(x)
        x = x.mean(-1).mean(-1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

config = {}
config['batch_size']=64
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
    net.train()
    for idx, (img, label) in enumerate(dataloader):
        img, label = Variable(img).cuda(config['gpu']), Variable(label).cuda(config['gpu'])
        pre = net(img)
        loss =cerit(pre, label)
        optim.zero_grad()
        optim1.zero_grad()
        loss.backward()
        optim.step()
        optim1.step()
        if idx%50==0:
            print ("TRAIN in epoch {} , step {}, loss = {}".format(epoch, idx, loss.data.cpu()[0]))

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

    loss = torch.stack(losses).mean()[0]
    acc = torch.stack(acces).mean()[0]
    print ("TEST in epoch{}, loss = {}, acc = {}%".format(epoch, loss, acc))
    torch.save(net.state_dict(), 'model.save')



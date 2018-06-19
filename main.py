'''Train WIDER with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision

import os
import argparse

from models import *
from utils import progress_bar

homePath = os.environ['HOME']
parser = argparse.ArgumentParser(description='PyTorch WIDER Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--net', '-n', default='gatemodel',type=str, help='define the name of the net')
parser.add_argument('--epochs', '-e', default=30, type=int, help='total epochs')
parser.add_argument('--batch_size', '-b', default=32, type=int)
parser.add_argument('--root', '-ro', default=homePath+'/data', type=str)
parser.add_argument('--traintxt', '-tr', default=homePath+'/data/WIDER_v0.1/train.lst', type=str)
parser.add_argument('--testtxt', '-te', default=homePath+'/data/WIDER_v0.1/test.lst', type=str)
parser.add_argument('--anotation', '-a', default='', type=str)
parser.add_argument('--sgd', default=False, type=bool)
parser.add_argument('--decay_step', default=-1, type=int)
parser.add_argument('--decay_rate', default=0.1, type=float)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = -1  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
from data import trainset as _trainset
from data import testset as _testset
trainset = _trainset(args.root, args.traintxt)
testset = _testset(args.root, args.testtxt)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=8)

# Model
print('==> Building model..')
from models import *
if args.net == 'gatemodel':
    net = GateModel()
elif args.net == 'gatemodelbig':
    net = GateModelBig()
elif args.net == 'gatemodelorigin':
    net = GateModelOrigin()
elif args.net == 'gatemodelmodified':
    net = GateModelModified()
elif args.net == 'gatemodelhuge':
    net = GateModelHuge()
elif args.net == 'originalmodel':
    net = OriginalModel()
netname = type(net).__name__ + args.anotation

if device == 'cuda':
    net.cuda()

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/'+netname+'.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
if not args.sgd:
    optimizer = optim.Adam(net.train_parameters(), lr=args.lr)
else:
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
if args.decay_step>0:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.decay_rate, last_epoch=start_epoch)
print(net)
print ('training on {}, of net {}, with lr {}, batch_size {}, sgd={}'.format(device, netname, args.lr, args.batch_size, args.sgd))

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        if isinstance(outputs, (list, tuple)):
            loss = 0.0
            for out in outputs:
                loss += criterion(out, targets)
        else:
            loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        if isinstance(outputs, (list, tuple)):
            _,predicted = outputs[0].max(1)
        else:
            _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/'+netname+'.t7')
        best_acc = acc


for epoch in range(start_epoch+1, start_epoch+ args.epochs):
    if args.decay_step>0:
        scheduler.step()
    train(epoch)
    test(epoch)

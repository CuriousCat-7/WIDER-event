'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import functional as F


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def speicalize_train(ouputs, targets, criterion, args, device):
    '''
    outputs: list with [real_output, expert1_output, expert2_output ..., gate_output]
    if any experts get right:
        train the right experts to depress other wrong labels
        train the gate expert to point to that experts 
        do not care other experts

    else:
        train all experts
        train the gate to equal ?
    '''
    loss = 0.0
    gate_loss = 0.0
    outs = torch.stack(ouputs[1:-1],dim=-1) # shape B,61,3
    gate_outs = ouputs[-1] # shape B, 3
    B, E = outs.shape
    corrfun = lambda out: targets.eq(out.max(1)[-1]) # size B for correct
    corrs = torch.stack(map(corrfun, outs),dim=-1) #  list of corrrect vectors shape B,3
    for i in range(B):
        corr = corrs[i] # shape 3
        out = outs[i] # shape 61,3
        gate_out = gate_outs[i] # shape 3
        location = []
        for j in range(E):
            ou = out[:,j] # shape 61
            cor = corr[j].itme()
            if cor: # right, match
                loss += criterion(ou, targets[i])
                location.append(j)
        if location is not []: # get right
            eta = 1.0/len(location)
            gate_template = torch.zeros(E)
            for a in location:
                gate_template[a] = eta
            if device == 'cuda':
                gate_template = gate_template.cuda()
            gate_loss += F.mse_loss(gate_out, gate_template)
        else: # all wrong
            for j in range(E):
                ou = out[:,j] # shape 61
                cor = corr[j]
                loss += criterion(ou, targets[i])
            if args.gate_equal:
                eta = 1.0/E
                gate_template = torch.zeros(E).fill_(eta)
                if device == 'cuda':
                    gate_template = gate_template.cuda()
                gate_loss += F.mse_loss(gate_out, gate_template)
        return loss + gate_loss








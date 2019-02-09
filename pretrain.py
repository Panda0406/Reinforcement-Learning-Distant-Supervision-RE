# -*- coding: utf-8 -*-
from __future__ import division
from copy import deepcopy
from Networks import Policy_CNN
from utils import *

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor

def pretrain_policy_CNN(pos_data, neg_data, inputs, embeddings, args, rv, batch_size):
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    print "Pretraining Policy Network, positive size: %d, negative size: %d" % (len(pos_data), len(neg_data))

    Policy_model = Policy_CNN(embeddings, args)
    Policy_model = Policy_model.cuda() if use_cuda else Policy_model

    x_ = pos_data+neg_data
    y_ = [1]*len(pos_data)+[0]*len(neg_data)

    trainloader = generate_trainloader(inputs, x_, y_, batch_size, shuf=True)

    parameters = filter(lambda p: p.requires_grad, Policy_model.parameters())
    optimizer = optim.Adam(parameters, lr=0.0005)

    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99, last_epoch=-1)

    signal = 0
    for epoch in range(10):
        Policy_model.train()
        for i, (x, y) in enumerate(trainloader, 0):
            rv_matrix = rv.repeat(x.size(0), 1)
            logits, _ = calculate_logits(Policy_model, x, args.max_sent, rv_matrix=rv_matrix)
            loss = calculate_loss(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        accuracy, accuracy_non_NA = calculate_accuracy(trainloader, Policy_model, args, epoch, Policy=True)

        # Avoid over-fitting
        if accuracy_non_NA > 80.0 and signal == 0:
            signal = 1
            Policy_model.embed.weight.requires_grad = False
            Policy_model.embed_pf1.weight.requires_grad = False
            Policy_model.embed_pf2.weight.requires_grad = False
        if accuracy_non_NA > 85.0 and epoch >= 5:
            break
        if accuracy_non_NA > 90.0:
            break

    torch.save(Policy_model.state_dict(), './models/Policy_pretrain.pkl')

    return Policy_model

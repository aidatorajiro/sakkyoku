import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import random
import os
from read import *
from const import *

netGen = Gen()

# netGen.apply(init_weights)

criterion = nn.MSELoss()
optimizer = optim.Adam(netGen.parameters(), lr=0.0002, betas=(0.5, 0.999))
inp = torch.FloatTensor(batchsize, corps_len, 2, classes)
correct = torch.FloatTensor(batchsize, 2, classes)

if gen_load != '':
  netGen.load_state_dict(torch.load(gen_load))

if gen_load != '':
  optimizer.load_state_dict(torch.load(optim_load))

if cuda:
  netGen.cuda()
  criterion.cuda()
  inp = inp.cuda()
  correct = correct.cuda()

import datetime

for epoch in range(iters):
  for i, data in enumerate(dataloader, 0):
    print("okok")
    netGen.zero_grad()
    bsize = data.size(0)

    inp.resize_(bsize, corps_len, 2, classes).fill_(0)
    correct.resize_(bsize, 2, classes).fill_(0)

    data.clamp_(0, classes - 1)

    for batchid in range(0, bsize):
      for noteid in range(0, corps_len):
        inp[batchid][noteid][0][data[batchid][noteid][0]] = 1
        inp[batchid][noteid][1][data[batchid][noteid][1]] = 1

    for batchid in range(0, bsize):
      correct[batchid][0][data[batchid][-1][0]] = 1
      correct[batchid][1][data[batchid][-1][1]] = 1

    print("okok 2")

    output = netGen(inp)
    loss = criterion(output, correct)

    print(output, correct, loss)

    loss.backward()

    optimizer.step()

  print('epoch ' + str(epoch))

  if epoch % 100 == 0:
    torch.save(optimizer.state_dict(), "optim_" + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    torch.save(netGen.state_dict(), "gen_" + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))

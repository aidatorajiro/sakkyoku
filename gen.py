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
from conv import *

gentype = 'select_after_one'
#gentype = 'select_after_all'

netGen = Gen()

inp = torch.FloatTensor(corps_len, 2, classes)
next_note = torch.zeros(1, 2, classes)

if gen_load != '' and cuda:
  netGen.load_state_dict(torch.load(gen_load, map_location='cuda'))

if gen_load != '' and not cuda:
  netGen.load_state_dict(torch.load(gen_load, map_location='cpu'))

if cuda:
  netGen.cuda()
  inp = inp.cuda()
  next_note = next_note.cuda()

data = dataloader.__iter__().__next__()[0]

notes = []

for i in range(1000):
  inp.fill_(0)

  data.clamp_(0, classes - 1)

  for noteid in range(0, corps_len):
    inp[noteid][0][data[noteid][0]] = 1
    inp[noteid][1][data[noteid][1]] = 1

  output = netGen(inp.view(1, corps_len, 2, classes))

  if gentype == 'select_after_all':
    inp = torch.cat([inp[:-1], output])

  if gentype == 'select_after_one':
    output = output.view(2, classes)
    out_index_0 = numpy.random.choice(range(classes), p=output[0].cpu().detach().numpy())
    out_index_1 = numpy.random.choice(range(classes), p=output[1].cpu().detach().numpy())
    notes.append((out_index_0, out_index_1))
    next_note.fill_(0)
    next_note[0][0][out_index_0] = 1
    next_note[0][1][out_index_1] = 1
    inp = torch.cat([inp[:-1], next_note])

notes_to_mid(notes).save("out.mid")
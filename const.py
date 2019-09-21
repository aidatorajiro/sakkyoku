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

current_dir = os.path.dirname(os.path.abspath(__file__))

gen_load = 'gen_2019-09-05-08-58-10'
optim_load = 'optim_2019-09-05-08-58-08'

# gen_load = ''
# optim_load = ''

outdir = './out'
cuda = False
corps_len = 16
classes = 128
batchsize = 128
iters = 10000

random.seed(12345)
if cuda:
    torch.cuda.manual_seed_all(12345)

class Gen(nn.Module):
    def __init__(self):
        super(Gen, self).__init__()
        self.in_feats = corps_len * 2 * classes
        self.out_feats = 2 * classes

        self.network = nn.Sequential(
          nn.Linear(self.in_feats, (self.in_feats // 4) * 3),
          nn.ReLU(),
          nn.Linear((self.in_feats // 4) * 3, (self.in_feats // 4) * 2),
          nn.ReLU(),
          nn.Linear((self.in_feats // 4) * 2, (self.in_feats // 4) * 1),
          nn.ReLU(),
          nn.Linear((self.in_feats // 4) * 1, self.out_feats)
        )
    
    def forward(self, input):
        bsize = input.size(0)
        output = self.network(input.view(bsize, self.in_feats))
        output = output.view(bsize, 2, classes)
        output = torch.softmax(output, 2)
        return output

dataset = MidiDataset(
    dir=os.path.join(current_dir, 'dataset'),
    program_type='melodic',
    split_width=corps_len + 1
)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batchsize,
    shuffle=True
)

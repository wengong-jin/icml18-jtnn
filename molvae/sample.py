import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable

import math, random, sys
from optparse import OptionParser
from collections import deque
import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import Draw

from jtnn import *

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = OptionParser()
parser.add_option("-n", "--nsample", dest="nsample")
parser.add_option("-v", "--vocab", dest="vocab_path")
parser.add_option("-m", "--model", dest="model_path")
parser.add_option("-w", "--hidden", dest="hidden_size", default=200)
parser.add_option("-l", "--latent", dest="latent_size", default=56)
parser.add_option("-d", "--depth", dest="depth", default=3)
parser.add_option("-e", "--stereo", dest="stereo", default=1)
opts,args = parser.parse_args()
   
vocab = [x.strip("\r\n ") for x in open(opts.vocab_path)] 
vocab = Vocab(vocab)

hidden_size = int(opts.hidden_size)
latent_size = int(opts.latent_size)
depth = int(opts.depth)
nsample = int(opts.nsample)
stereo = True if int(opts.stereo) == 1 else False

model = JTNNVAE(vocab, hidden_size, latent_size, depth, stereo=stereo)
load_dict = torch.load(opts.model_path)
missing = {k: v for k, v in model.state_dict().items() if k not in load_dict}
load_dict.update(missing) 
model.load_state_dict(load_dict)
model = model.cuda()

torch.manual_seed(0)
for i in xrange(nsample):
    print model.sample_prior(prob_decode=False)

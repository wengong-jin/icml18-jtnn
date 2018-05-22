import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import math, random, sys
from optparse import OptionParser
from collections import deque

from jtnn import *

parser = OptionParser()
parser.add_option("-z", "--test", dest="test_path")
parser.add_option("-m", "--model", dest="model_path")
parser.add_option("-b", "--batch", dest="batch_size", default=50)
parser.add_option("-w", "--hidden", dest="hidden_size", default=300)
parser.add_option("-d", "--depth", dest="depth", default=3)
parser.add_option("-a", "--attention", dest="attention", default=0)
opts,args = parser.parse_args()
   
batch_size = int(opts.batch_size)
depth = int(opts.depth)
hidden_size = int(opts.hidden_size)
attention = int(opts.attention)

if attention > 0:
    encoder = AttMPN(hidden_size, depth)
else:
    encoder = MPN(hidden_size, depth)

model = nn.Sequential(
        encoder,
        nn.Linear(hidden_size, hidden_size), 
        nn.ReLU(), 
        nn.Linear(hidden_size, 1)
    )

model.load_state_dict(torch.load(opts.model_path))
model = model.cuda()

def get_data(path):
    data = []
    with open(path) as f:
        for line in f:
            r = line.strip("\r\n ").split()[0]
            data.append(r)
    return data


def valid_loss(data):
    model.train(False)
    for i in xrange(0, len(data), batch_size):
        mol_batch = data[i:i+batch_size]
        mol_batch = mol2graph(mol_batch)
        preds = model(mol_batch).view(-1)
        preds = preds.data.tolist()
        for i in xrange(len(mol_batch)):
            print preds[i]

test = get_data(opts.test_path)
valid_loss(test)

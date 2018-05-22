import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import math, random, sys
from optparse import OptionParser
from collections import deque

from jtnn import *

parser = OptionParser()
parser.add_option("-t", "--train", dest="train_path")
parser.add_option("-v", "--valid", dest="valid_path")
parser.add_option("-z", "--test", dest="test_path")
parser.add_option("-m", "--save_dir", dest="save_path")
parser.add_option("-b", "--batch", dest="batch_size", default=50)
parser.add_option("-w", "--hidden", dest="hidden_size", default=300)
parser.add_option("-d", "--depth", dest="depth", default=3)
parser.add_option("-e", "--epoch", dest="epoch", default=20)
parser.add_option("-a", "--attention", dest="attention", default=0)
parser.add_option("-p", "--dropout", dest="dropout", default=0.2)
opts,args = parser.parse_args()
   
batch_size = int(opts.batch_size)
depth = int(opts.depth)
hidden_size = int(opts.hidden_size)
num_epoch = int(opts.epoch)
attention = int(opts.attention)
dropout = float(opts.dropout)

if attention > 0:
    encoder = AttMPN(hidden_size, depth, dropout=dropout)
else:
    encoder = MPN(hidden_size, depth)

model = nn.Sequential(
        encoder,
        nn.Linear(hidden_size, hidden_size), 
        nn.ReLU(), 
        nn.Linear(hidden_size, 1)
    )
loss_fn = nn.MSELoss().cuda()
model = model.cuda()

for param in model.parameters():
    if param.dim() == 1:
        nn.init.constant(param, 0)
    else:
        nn.init.xavier_normal(param)

print "Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = lr_scheduler.ExponentialLR(optimizer, 0.9)
scheduler.step()

def get_data(path):
    data = []
    with open(path) as f:
        for line in f:
            r,v = line.strip("\r\n ").split()
            data.append((r,float(v)))
    return data

train = get_data(opts.train_path)
valid = get_data(opts.valid_path)
test = get_data(opts.test_path)

random.shuffle(train)

def valid_loss(data):
    mse = 0
    model.train(False)
    for i in xrange(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        mol_batch, label_batch = zip(*batch)
        mol_batch = mol2graph(mol_batch)
        labels = create_var(torch.Tensor(label_batch))

        preds = model(mol_batch).view(-1)
        loss = loss_fn(preds, labels)
        mse += loss.data[0] * batch_size 

    model.train(True)
    return mse / len(data)
     
best_loss = 1e5
for epoch in xrange(num_epoch):
    mse,it = 0,0
    print "learning rate: %.6f" % scheduler.get_lr()[0]
    for i in xrange(0, len(train), batch_size):
        batch = train[i:i+batch_size]
        mol_batch, label_batch = zip(*batch)
        mol_batch = mol2graph(mol_batch)
        labels = create_var(torch.Tensor(label_batch))

        model.zero_grad()
        preds = model(mol_batch).view(-1)
        loss = loss_fn(preds, labels)
        mse += loss.data[0] * batch_size 
        it += batch_size
        loss.backward()
        optimizer.step()

        if i % 1000 == 0:
            pnorm = math.sqrt(sum([p.norm().data[0] ** 2 for p in model.parameters()]))
            gnorm = math.sqrt(sum([p.grad.norm().data[0] ** 2 for p in model.parameters()]))
            print "RMSE=%.4f,PNorm=%.2f,GNorm=%.2f" % (math.sqrt(mse / it), pnorm, gnorm)
            sys.stdout.flush()
            mse,it = 0,0

    scheduler.step()
    cur_loss = valid_loss(valid)
    print "validation loss: %.4f" % cur_loss
    if opts.save_path is not None:
        torch.save(model.state_dict(), opts.save_path + "/model.iter-" + str(epoch))
        if cur_loss < best_loss:
            best_loss = cur_loss
            torch.save(model.state_dict(), opts.save_path + "/model.best")


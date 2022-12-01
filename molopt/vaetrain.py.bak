import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable

import math, random, sys
from optparse import OptionParser
from collections import deque

from jtnn import *
import rdkit

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = OptionParser()
parser.add_option("-t", "--train", dest="train_path")
parser.add_option("-v", "--vocab", dest="vocab_path")
parser.add_option("-p", "--prop", dest="prop_path")
parser.add_option("-s", "--save_dir", dest="save_path")
parser.add_option("-m", "--model", dest="model_path", default=None)
parser.add_option("-b", "--batch", dest="batch_size", default=40)
parser.add_option("-w", "--hidden", dest="hidden_size", default=200)
parser.add_option("-l", "--latent", dest="latent_size", default=56)
parser.add_option("-d", "--depth", dest="depth", default=3)
parser.add_option("-z", "--beta", dest="beta", default=1.0)
parser.add_option("-q", "--lr", dest="lr", default=1e-3)
opts,args = parser.parse_args()
   
vocab = [x.strip("\r\n ") for x in open(opts.vocab_path)] 
vocab = Vocab(vocab)

batch_size = int(opts.batch_size)
hidden_size = int(opts.hidden_size)
latent_size = int(opts.latent_size)
depth = int(opts.depth)
beta = float(opts.beta)
lr = float(opts.lr)

model = JTPropVAE(vocab, hidden_size, latent_size, depth)

if opts.model_path is not None:
    model.load_state_dict(torch.load(opts.model_path))
else:
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant(param, 0)
        else:
            nn.init.xavier_normal(param)

model = model.cuda()
print "Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,)

optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.ExponentialLR(optimizer, 0.9)
scheduler.step()

dataset = PropDataset(opts.train_path, opts.prop_path)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=lambda x:x, drop_last=True)

MAX_EPOCH = 6
PRINT_ITER = 20

for epoch in xrange(MAX_EPOCH):
    word_acc,topo_acc,assm_acc,steo_acc,prop_acc = 0,0,0,0,0

    for it, batch in enumerate(dataloader):
        for mol_tree,_ in batch:
            for node in mol_tree.nodes:
                if node.label not in node.cands:
                    node.cands.append(node.label)
                    node.cand_mols.append(node.label_mol)

        model.zero_grad()
        loss, kl_div, wacc, tacc, sacc, dacc, pacc = model(batch, beta)
        loss.backward()
        optimizer.step()

        word_acc += wacc
        topo_acc += tacc
        assm_acc += sacc
        steo_acc += dacc
        prop_acc += pacc

        if (it + 1) % PRINT_ITER == 0:
            word_acc = word_acc / PRINT_ITER * 100
            topo_acc = topo_acc / PRINT_ITER * 100
            assm_acc = assm_acc / PRINT_ITER * 100
            steo_acc = steo_acc / PRINT_ITER * 100
            prop_acc /= PRINT_ITER

            print "KL: %.1f, Word: %.2f, Topo: %.2f, Assm: %.2f, Steo: %.2f, Prop: %.4f" % (kl_div, word_acc, topo_acc, assm_acc, steo_acc, prop_acc)
            word_acc,topo_acc,assm_acc,steo_acc,prop_acc = 0,0,0,0,0
            sys.stdout.flush()

        if (it + 1) % 1500 == 0: #Fast annealing
            scheduler.step()
            print "learning rate: %.6f" % scheduler.get_lr()[0]
            torch.save(model.state_dict(), opts.save_path + "/model.iter-%d-%d" % (epoch, it + 1))

    scheduler.step()
    print "learning rate: %.6f" % scheduler.get_lr()[0]
    torch.save(model.state_dict(), opts.save_path + "/model.iter-" + str(epoch))


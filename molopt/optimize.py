import torch
import torch.nn as nn
from torch.autograd import Variable

import math, random, sys
from optparse import OptionParser
from collections import deque

import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import Descriptors
import sascorer

from jtnn import *

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = OptionParser()
parser.add_option("-t", "--test", dest="test_path")
parser.add_option("-v", "--vocab", dest="vocab_path")
parser.add_option("-m", "--model", dest="model_path")
parser.add_option("-w", "--hidden", dest="hidden_size", default=200)
parser.add_option("-l", "--latent", dest="latent_size", default=56)
parser.add_option("-d", "--depth", dest="depth", default=3)
parser.add_option("-s", "--sim", dest="cutoff", default=0.0)
opts,args = parser.parse_args()
   
vocab = [x.strip("\r\n ") for x in open(opts.vocab_path)] 
vocab = Vocab(vocab)

hidden_size = int(opts.hidden_size)
latent_size = int(opts.latent_size)
depth = int(opts.depth)
sim_cutoff = float(opts.cutoff)

model = JTPropVAE(vocab, hidden_size, latent_size, depth)
model.load_state_dict(torch.load(opts.model_path))
model = model.cuda()

data = []
with open(opts.test_path) as f:
    for line in f:
        s = line.strip("\r\n ").split()[0]
        data.append(s)

res = []
for smiles in data:
    mol = Chem.MolFromSmiles(smiles)
    score = Descriptors.MolLogP(mol) - sascorer.calculateScore(mol)

    new_smiles,sim = model.optimize(smiles, sim_cutoff=sim_cutoff, lr=2, num_iter=80)
    new_mol = Chem.MolFromSmiles(new_smiles)
    new_score = Descriptors.MolLogP(new_mol) - sascorer.calculateScore(new_mol)

    res.append( (new_score - score, sim, score, new_score, smiles, new_smiles) )
    print(new_score - score, sim, score, new_score, smiles, new_smiles)

print(sum([x[0] for x in res]), sum([x[1] for x in res]))

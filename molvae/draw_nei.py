import torch
import torch.nn as nn
from torch.autograd import Variable

import math, random, sys
from optparse import OptionParser

import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import Draw

import numpy as np
from jtnn import *

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = OptionParser()
parser.add_option("-v", "--vocab", dest="vocab_path")
parser.add_option("-m", "--model", dest="model_path")
parser.add_option("-w", "--hidden", dest="hidden_size", default=200)
parser.add_option("-l", "--latent", dest="latent_size", default=56)
parser.add_option("-d", "--depth", dest="depth", default=3)
opts,args = parser.parse_args()
   
vocab = [x.strip("\r\n ") for x in open(opts.vocab_path)] 
vocab = Vocab(vocab)

hidden_size = int(opts.hidden_size)
latent_size = int(opts.latent_size)
depth = int(opts.depth)

model = JTNNVAE(vocab, hidden_size, latent_size, depth)
model.load_state_dict(torch.load(opts.model_path))
model = model.cuda()

np.random.seed(0)
x = np.random.randn(latent_size)
x /= np.linalg.norm(x)

y = np.random.randn(latent_size)
y -= y.dot(x) * x
y /= np.linalg.norm(y)

#z0 = "CN1C(C2=CC(NC3C[C@H](C)C[C@@H](C)C3)=CN=C2)=NN=C1"
z0 = "COC1=CC(OC)=CC([C@@H]2C[NH+](CCC(F)(F)F)CC2)=C1"
z0 = model.encode_latent_mean([z0]).squeeze()
z0 = z0.data.cpu().numpy()

delta = 1
nei_mols = []
for dx in xrange(-6,7):
    for dy in xrange(-6,7):
        z = z0 + x * delta * dx + y * delta * dy
        tree_z, mol_z = torch.Tensor(z).unsqueeze(0).chunk(2, dim=1)
        tree_z, mol_z = create_var(tree_z), create_var(mol_z)
        nei_mols.append( model.decode(tree_z, mol_z, prob_decode=False) )

nei_mols = [Chem.MolFromSmiles(s) for s in nei_mols]
img = Draw.MolsToGridImage(nei_mols, molsPerRow=13, subImgSize=(200,200), useSVG=True)
print img


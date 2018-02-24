import pickle
import gzip
from sparse_gp import SparseGP
import scipy.stats as sps
import numpy as np
import os.path

import rdkit
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem import Descriptors

import torch
import torch.nn as nn
from jtnn import create_var, JTNNVAE, Vocab

from optparse import OptionParser

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

# We define the functions used to load and save objects
def save_object(obj, filename):
    result = pickle.dumps(obj)
    with gzip.GzipFile(filename, 'wb') as dest: dest.write(result)
    dest.close()

def load_object(filename):
    with gzip.GzipFile(filename, 'rb') as source: result = source.read()
    ret = pickle.loads(result)
    source.close()
    return ret

parser = OptionParser()
parser.add_option("-v", "--vocab", dest="vocab_path")
parser.add_option("-m", "--model", dest="model_path")
parser.add_option("-o", "--save_dir", dest="save_dir")
parser.add_option("-w", "--hidden", dest="hidden_size", default=200)
parser.add_option("-l", "--latent", dest="latent_size", default=56)
parser.add_option("-d", "--depth", dest="depth", default=3)
parser.add_option("-r", "--seed", dest="random_seed", default=None)
opts,args = parser.parse_args()

vocab = [x.strip("\r\n ") for x in open(opts.vocab_path)] 
vocab = Vocab(vocab)

hidden_size = int(opts.hidden_size)
latent_size = int(opts.latent_size)
depth = int(opts.depth)
random_seed = int(opts.random_seed)

model = JTNNVAE(vocab, hidden_size, latent_size, depth)
model.load_state_dict(torch.load(opts.model_path))
model = model.cuda()

# We load the random seed
np.random.seed(random_seed)

# We load the data (y is minued!)
X = np.loadtxt('latent_features.txt')
y = -np.loadtxt('targets.txt')
y = y.reshape((-1, 1))

n = X.shape[ 0 ]
permutation = np.random.choice(n, n, replace = False)

X_train = X[ permutation, : ][ 0 : np.int(np.round(0.9 * n)), : ]
X_test = X[ permutation, : ][ np.int(np.round(0.9 * n)) :, : ]

y_train = y[ permutation ][ 0 : np.int(np.round(0.9 * n)) ]
y_test = y[ permutation ][ np.int(np.round(0.9 * n)) : ]

np.random.seed(random_seed)

logP_values = np.loadtxt('logP_values.txt')
SA_scores = np.loadtxt('SA_scores.txt')
cycle_scores = np.loadtxt('cycle_scores.txt')
SA_scores_normalized = (np.array(SA_scores) - np.mean(SA_scores)) / np.std(SA_scores)
logP_values_normalized = (np.array(logP_values) - np.mean(logP_values)) / np.std(logP_values)
cycle_scores_normalized = (np.array(cycle_scores) - np.mean(cycle_scores)) / np.std(cycle_scores)

iteration = 0
while iteration < 5:
    # We fit the GP
    np.random.seed(iteration * random_seed)
    M = 500
    sgp = SparseGP(X_train, 0 * X_train, y_train, M)
    sgp.train_via_ADAM(X_train, 0 * X_train, y_train, X_test, X_test * 0, y_test, minibatch_size = 10 * M, max_iterations = 100, learning_rate = 0.001)

    pred, uncert = sgp.predict(X_test, 0 * X_test)
    error = np.sqrt(np.mean((pred - y_test)**2))
    testll = np.mean(sps.norm.logpdf(pred - y_test, scale = np.sqrt(uncert)))
    print 'Test RMSE: ', error
    print 'Test ll: ', testll

    pred, uncert = sgp.predict(X_train, 0 * X_train)
    error = np.sqrt(np.mean((pred - y_train)**2))
    trainll = np.mean(sps.norm.logpdf(pred - y_train, scale = np.sqrt(uncert)))
    print 'Train RMSE: ', error
    print 'Train ll: ', trainll

    # We pick the next 60 inputs
    next_inputs = sgp.batched_greedy_ei(60, np.min(X_train, 0), np.max(X_train, 0))
    valid_smiles = []
    new_features = []
    for i in xrange(60):
        all_vec = next_inputs[i].reshape((1,-1))
        tree_vec,mol_vec = np.hsplit(all_vec, 2)
        tree_vec = create_var(torch.from_numpy(tree_vec).float())
        mol_vec = create_var(torch.from_numpy(mol_vec).float())
        s = model.decode(tree_vec, mol_vec, prob_decode=False)
        if s is not None: 
            valid_smiles.append(s)
            new_features.append(all_vec)
    
    print len(valid_smiles), "molecules are found"
    valid_smiles = valid_smiles[:50]
    new_features = next_inputs[:50]
    new_features = np.vstack(new_features)
    save_object(valid_smiles, opts.save_dir + "/valid_smiles{}.dat".format(iteration))

    import sascorer
    import networkx as nx
    from rdkit.Chem import rdmolops

    scores = []
    for i in range(len(valid_smiles)):
        current_log_P_value = Descriptors.MolLogP(MolFromSmiles(valid_smiles[ i ]))
        current_SA_score = -sascorer.calculateScore(MolFromSmiles(valid_smiles[ i ]))
        cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(MolFromSmiles(valid_smiles[ i ]))))
        if len(cycle_list) == 0:
            cycle_length = 0
        else:
            cycle_length = max([ len(j) for j in cycle_list ])
        if cycle_length <= 6:
            cycle_length = 0
        else:
            cycle_length = cycle_length - 6

        current_cycle_score = -cycle_length
     
        current_SA_score_normalized = (current_SA_score - np.mean(SA_scores)) / np.std(SA_scores)
        current_log_P_value_normalized = (current_log_P_value - np.mean(logP_values)) / np.std(logP_values)
        current_cycle_score_normalized = (current_cycle_score - np.mean(cycle_scores)) / np.std(cycle_scores)

        score = current_SA_score_normalized + current_log_P_value_normalized + current_cycle_score_normalized
        scores.append(-score) #target is always minused

    print valid_smiles
    print scores 

    save_object(scores, opts.save_dir + "/scores{}.dat".format(iteration))

    if len(new_features) > 0:
        X_train = np.concatenate([ X_train, new_features ], 0)
        y_train = np.concatenate([ y_train, np.array(scores)[ :, None ] ], 0)

    iteration += 1

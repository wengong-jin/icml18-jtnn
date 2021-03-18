import torch
import torch.nn as nn
from multiprocessing import Pool

import math, random, sys
from optparse import OptionParser
import pickle

from fast_jtnn import *
import rdkit

def tensorize(smiles, assm=True):
    mol_tree = MolTree(smiles)
    mol_tree.recover()
    if assm:
        mol_tree.assemble()
        for node in mol_tree.nodes:
            if node.label not in node.cands:
                node.cands.append(node.label)

    del mol_tree.mol
    for node in mol_tree.nodes:
        del node.mol

    return mol_tree


if __name__ == "__main__":
    import csv
    from tqdm import tqdm

    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = OptionParser()
    parser.add_option("-t", "--train", dest="train_path", help="csv file containing smiles columns and label columns.")
    parser.add_option("-d", "--data_col", dest="data_col", type=int)
    parser.add_option("-n", "--split", dest="nsplits", default=1)
    parser.add_option("-j", "--jobs", dest="njobs", default=8)
    opts,args = parser.parse_args()
    opts.njobs = int(opts.njobs)

    pool = Pool(opts.njobs)
    num_splits = int(opts.nsplits)

    reader = csv.reader(open(opts.train_path, 'r'), delimiter=',')
    next(reader)
    data = list(reader)
    data = map(list, zip(*data))

    labels = [x for x in range(len(data)) if x != opts.data_col]

    smiles = data[opts.data_col]
    all_labels = [data[c] for c in labels]

    all_mol_tree = list(tqdm(pool.imap(tensorize, smiles), total=len(smiles)))

    le = (len(all_mol_tree) + num_splits - 1) / num_splits

    for split_id in xrange(num_splits):
        st = split_id * le
        sub_data = [all_mol_tree[st : st + le]] + [label[st : st + le] for label in all_labels]

        with open('tensors-%d.pkl' % split_id, 'wb') as f:
            pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)

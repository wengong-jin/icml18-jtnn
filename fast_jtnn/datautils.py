import torch
from torch.utils.data import Dataset, DataLoader
from mol_tree import MolTree
import numpy as np
from jtnn_enc import JTNNEncoder
from mpn import MPN
from jtmpn import JTMPN
import cPickle as pickle
import os, random
from itertools import cycle

class PairTreeFolder(object):

    def __init__(self, data_folder, vocab, batch_size, num_workers=4, shuffle=True, y_assm=True, replicate=None):
        self.data_folder = data_folder
        self.data_files = [fn for fn in os.listdir(data_folder)]
        self.batch_size = batch_size
        self.vocab = vocab
        self.num_workers = num_workers
        self.y_assm = y_assm
        self.shuffle = shuffle

        if replicate is not None: #expand is int
            self.data_files = self.data_files * replicate

    def __iter__(self):
        for fn in self.data_files:
            fn = os.path.join(self.data_folder, fn)
            with open(fn) as f:
                data = pickle.load(f)

            if self.shuffle:
                random.shuffle(data) #shuffle data before batch

            batches = [data[i : i + self.batch_size] for i in xrange(0, len(data), self.batch_size)]
            if len(batches[-1]) < self.batch_size:
                batches.pop()

            dataset = PairTreeDataset(batches, self.vocab, self.y_assm)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, collate_fn=lambda x:x[0])

            for b in dataloader:
                yield b

            del data, batches, dataset, dataloader

class MolTreeFolder(object):

    def __init__(self, data_folder, vocab, label_idx=0, batch_size=4, shuffle=True, assm=True, replicate=None):
        self.data_folder = data_folder
        self.data_files = [fn for fn in os.listdir(data_folder)]
        self.vocab = vocab
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.assm = assm

        if replicate is not None: #expand is int
            self.data_files = self.data_files * replicate

        data = []

        for fn in self.data_files:
            fn = os.path.join(self.data_folder, fn)
            with open(fn) as f:
                subset_data = pickle.load(f)
                data.append(np.array(subset_data).T)

        data = np.concatenate(data)

        if self.shuffle:
            np.random.shuffle(data)

        self.data = data

    def __repr__(self):
        return len(self.data)

    def __iter__(self):

        molecule_embeds = np.array([row[0] for row in self.data])
        labels = np.array([row[1] for row in self.data])

        labeled_indices = np.where(labels != "")
        unlabeled_indices = np.where(labels == "")

        # Split data based on labelled and unlabelled
        supervised_moltree = np.array(molecule_embeds[labeled_indices])
        unsupervised_moltree = np.array(molecule_embeds[unlabeled_indices])
        supervised_labels = np.array(labels[labeled_indices], dtype=np.float)
        placeholder_labels = np.array([0 for i in range(len(unlabeled_indices[0]))])

        # Create batch
        supervised_moltree = [supervised_moltree[i : i + self.batch_size] for i in xrange(0, len(supervised_moltree), self.batch_size)]
        supervised_labels = [supervised_labels[i : i + self.batch_size] for i in xrange(0, len(supervised_labels), self.batch_size)]
        unsupervised_moltree = [unsupervised_moltree[i : i + self.batch_size] for i in xrange(0, len(unsupervised_moltree), self.batch_size)]
        placeholder_labels = [placeholder_labels[i : i + self.batch_size] for i in xrange(0, len(placeholder_labels), self.batch_size)]

        supervised_dataset = MolTreeDataset(supervised_moltree, supervised_labels, self.vocab, self.assm)
        unsupervised_dataset = MolTreeDataset(unsupervised_moltree, placeholder_labels, self.vocab, self.assm)
        supervised_dataloader = DataLoader(supervised_dataset, batch_size=1, shuffle=False, collate_fn=lambda x:x[0])
        unsupervised_dataloader = DataLoader(unsupervised_dataset, batch_size=1, shuffle=False, collate_fn=lambda x:x[0])

        for supervised, unsupervised in zip(cycle(supervised_dataloader), unsupervised_dataloader):
            yield (supervised, unsupervised)


class PairTreeDataset(Dataset):

    def __init__(self, data, vocab, y_assm):
        self.data = data
        self.vocab = vocab
        self.y_assm = y_assm

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        batch0, batch1 = zip(*self.data[idx])
        return tensorize(batch0, self.vocab, assm=False), tensorize(batch1, self.vocab, assm=self.y_assm)


class MolTreeDataset(Dataset):

    def __init__(self, data, labels, vocab, assm=True):
        self.data = data
        self.labels = labels
        self.vocab = vocab
        self.assm = assm

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {'data': tensorize(self.data[idx], self.vocab, assm=self.assm),
            'labels': self.labels[idx]}


def tensorize(tree_batch, vocab, assm=True):
    set_batch_nodeID(tree_batch, vocab)
    smiles_batch = [tree.smiles for tree in tree_batch]
    jtenc_holder,mess_dict = JTNNEncoder.tensorize(tree_batch)
    jtenc_holder = jtenc_holder
    mpn_holder = MPN.tensorize(smiles_batch)

    if assm is False:
        return tree_batch, jtenc_holder, mpn_holder

    cands = []
    batch_idx = []
    for i,mol_tree in enumerate(tree_batch):
        for node in mol_tree.nodes:
            #Leaf node's attachment is determined by neighboring node's attachment
            if node.is_leaf or len(node.cands) == 1: continue
            cands.extend( [(cand, mol_tree.nodes, node) for cand in node.cands] )
            batch_idx.extend([i] * len(node.cands))

    jtmpn_holder = JTMPN.tensorize(cands, mess_dict)
    batch_idx = torch.LongTensor(batch_idx)

    return tree_batch, jtenc_holder, mpn_holder, (jtmpn_holder,batch_idx)

def set_batch_nodeID(mol_batch, vocab):
    tot = 0
    for mol_tree in mol_batch:
        for node in mol_tree.nodes:
            node.idx = tot
            node.wid = vocab.get_index(node.smiles)
            tot += 1

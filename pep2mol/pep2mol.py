import os
os.environ["PYTHONPATH"] = "~/Users/maykcaldas/Documents/WhiteLab/pep2mol/icml18-jtnn"

import rdkit
import numpy as np
import json
from fast_jtnn.mol_tree import MolTreeNode, MolTree
import sys
lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)


def create_voc(smiles):
    voc = set()
    voc.add("null")
    for i,smi in enumerate(smiles):
        if smi == None: continue
        print("{}/{}".format(i, len(smiles)), end="\r")
        mol = MolTree(smi)
        for c in mol.nodes:
            voc.add(c.smiles)
    return voc


def add_noise(tree, mean, var):
    for node in tree.nodes:
        node.feat_vec += np.random.normal(mean, var, node.feat_vec.shape)
        node.feat_vec = [n if n > 0 else 0 for n in node.feat_vec]
        node.feat_vec = [n if n < 1 else 1 for n in node.feat_vec]

def hide_leafs(tree, prob, vocab):
    null_nodes = []
    for node in tree.nodes:
        if node.is_leaf and np.random.random() < prob:
            node.smiles = "null"
            node.is_leaf = False
            node.feat_vec = np.zeros(len(vocab)) 
            node.feat_vec[vocab[node.smiles]] = 1
            null_nodes.append(node)
    for node in null_nodes:
        # I'll still consider the null nodes as neighbors, 
        # but they won't be considered to evaluate if a node is a leaf
        for nei in node.neighbors:
            nei.is_leaf = (np.sum([n.smiles != 'null' for n in nei.neighbors]) == 1)

def main():
    VERBOSE=False
    CREATE_VOC=False
    # mols_list = sys.stdin.readlines()
    mols_list = []
    i=0
    with open('validation-db.json', 'r') as f:
        for line in f:
            i+=1
            mols_list.append(json.loads(line)['text'])
            if i>1000:
                break


    if CREATE_VOC:
        voc = create_voc(mols_list)
        stoi = {c:i for i,c in enumerate(voc)}
        with open("vocab.json", "w") as f:
            json.dump(dict(vocab=list(voc), vocab_stoi=stoi), f)
    else:
        with open("vocab.json", "r") as f:
            json_file = json.load(f)
            voc = json_file['vocab']
            stoi = json_file['vocab_stoi']

    if VERBOSE:
        print("vocab: ", voc)
        print("stoi: ", stoi)

    mols=[]
    for mol in mols_list:
        try:
            mols.append(MolTree(mol, stoi))
        except:
            pass

    print(f"we had {len(mols_list)} mols, but {len(mols)} were valid")

    for mol in mols:
        # print(mol.smiles)
        # print(mol.size(),"nodes: ",  [stoi[n.smiles] for n in mol.nodes]) #[n.smiles for n in mol.nodes])
        # print(len(mol.edges),"edges: ", mol.edges)

        if VERBOSE:
            for node in mol.nodes:
                print("{}:{:>10s} {}".format(node.nid, node.smiles, np.array(node.feat_vec)))

        add_noise(mol, 0, 0.01)
        for _ in range(3):
            hide_leafs(mol, 0.1, stoi)

        if VERBOSE:
            with np.printoptions(precision=4):
                for node in mol.nodes:
                    print("{}:{:>10s} {}".format(node.nid, node.smiles, np.array(node.feat_vec)))

if __name__ == "__main__":
    main
import sys
import gzip
import pickle
import rdkit.Chem as Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
import sascorer

def save_object(obj, filename):
    result = pickle.dumps(obj)
    with gzip.GzipFile(filename, 'wb') as dest: dest.write(result)
    dest.close()

def load_object(filename):
    with gzip.GzipFile(filename, 'rb') as source: result = source.read()
    ret = pickle.loads(result)
    source.close()
    return ret

all_smiles = []
for i in xrange(1,11):
    for j in xrange(5):
        fn = 'results%d/scores%d.dat' % (i,j)
        scores = load_object(fn)
        fn = 'results%d/valid_smiles%d.dat' % (i,j)
        smiles = load_object(fn)
        all_smiles.extend(zip(smiles, scores))

all_smiles = [(x,-y) for x,y in all_smiles]
all_smiles = sorted(all_smiles, key=lambda x:x[1], reverse=True)
for s,v in all_smiles:
    print s,v
#mols = [Chem.MolFromSmiles(s) for s,_ in all_smiles[:50]]
#vals = ["%.2f" % y for _,y in all_smiles[:50]]
#img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(200,135), legends=vals, useSVG=True)
#print img

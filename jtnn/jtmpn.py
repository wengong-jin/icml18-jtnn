import torch
import torch.nn as nn
from .nnutils import create_var, index_select_ND
from .chemutils import get_mol
#from mpn import atom_features, bond_features, ATOM_FDIM, BOND_FDIM
import rdkit.Chem as Chem

ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn', 'H', 'Cu', 'Mn', 'unknown']

ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 1
BOND_FDIM = 5 
MAX_NB = 10

def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def atom_features(atom):
    return torch.Tensor(onek_encoding_unk(atom.GetSymbol(), ELEM_LIST) 
            + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5]) 
            + onek_encoding_unk(atom.GetFormalCharge(), [-1,-2,1,2,0])
            + [atom.GetIsAromatic()])

def bond_features(bond):
    bt = bond.GetBondType()
    return torch.Tensor([bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing()])

class JTMPN(nn.Module):

    def __init__(self, hidden_size, depth):
        super(JTMPN, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth

        self.W_i = nn.Linear(ATOM_FDIM + BOND_FDIM, hidden_size, bias=False)
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_o = nn.Linear(ATOM_FDIM + hidden_size, hidden_size)

    def forward(self, cand_batch, tree_mess):
        fatoms,fbonds = [],[] 
        in_bonds,all_bonds = [],[] 
        mess_dict,all_mess = {},[create_var(torch.zeros(self.hidden_size))] #Ensure index 0 is vec(0)
        total_atoms = 0
        scope = []

        for e,vec in tree_mess.items():
            mess_dict[e] = len(all_mess)
            all_mess.append(vec)

        for mol,all_nodes,ctr_node in cand_batch:
            n_atoms = mol.GetNumAtoms()
            ctr_bid = ctr_node.idx

            for atom in mol.GetAtoms():
                fatoms.append( atom_features(atom) )
                in_bonds.append([]) 
        
            for bond in mol.GetBonds():
                a1 = bond.GetBeginAtom()
                a2 = bond.GetEndAtom()
                x = a1.GetIdx() + total_atoms
                y = a2.GetIdx() + total_atoms
                #Here x_nid,y_nid could be 0
                x_nid,y_nid = a1.GetAtomMapNum(),a2.GetAtomMapNum()
                x_bid = all_nodes[x_nid - 1].idx if x_nid > 0 else -1
                y_bid = all_nodes[y_nid - 1].idx if y_nid > 0 else -1

                bfeature = bond_features(bond)

                b = len(all_mess) + len(all_bonds)  #bond idx offseted by len(all_mess)
                all_bonds.append((x,y))
                fbonds.append( torch.cat([fatoms[x], bfeature], 0) )
                in_bonds[y].append(b)

                b = len(all_mess) + len(all_bonds)
                all_bonds.append((y,x))
                fbonds.append( torch.cat([fatoms[y], bfeature], 0) )
                in_bonds[x].append(b)

                if x_bid >= 0 and y_bid >= 0 and x_bid != y_bid:
                    if (x_bid,y_bid) in mess_dict:
                        mess_idx = mess_dict[(x_bid,y_bid)]
                        in_bonds[y].append(mess_idx)
                    if (y_bid,x_bid) in mess_dict:
                        mess_idx = mess_dict[(y_bid,x_bid)]
                        in_bonds[x].append(mess_idx)
            
            scope.append((total_atoms,n_atoms))
            total_atoms += n_atoms
        
        total_bonds = len(all_bonds)
        total_mess = len(all_mess)
        fatoms = torch.stack(fatoms, 0)
        fbonds = torch.stack(fbonds, 0)
        agraph = torch.zeros(total_atoms,MAX_NB).long()
        bgraph = torch.zeros(total_bonds,MAX_NB).long()
        tree_message = torch.stack(all_mess, dim=0)

        for a in range(total_atoms):
            for i,b in enumerate(in_bonds[a]):
                agraph[a,i] = b

        for b1 in range(total_bonds):
            x,y = all_bonds[b1]
            for i,b2 in enumerate(in_bonds[x]): #b2 is offseted by len(all_mess)
                if b2 < total_mess or all_bonds[b2-total_mess][0] != y:
                    bgraph[b1,i] = b2

        fatoms = create_var(fatoms)
        fbonds = create_var(fbonds)
        agraph = create_var(agraph)
        bgraph = create_var(bgraph)

        binput = self.W_i(fbonds)
        graph_message = nn.ReLU()(binput)

        for i in range(self.depth - 1):
            message = torch.cat([tree_message,graph_message], dim=0)
            nei_message = index_select_ND(message, 0, bgraph)
            nei_message = nei_message.sum(dim=1)
            nei_message = self.W_h(nei_message)
            graph_message = nn.ReLU()(binput + nei_message)

        message = torch.cat([tree_message,graph_message], dim=0)
        nei_message = index_select_ND(message, 0, agraph)
        nei_message = nei_message.sum(dim=1)
        ainput = torch.cat([fatoms, nei_message], dim=1)
        atom_hiddens = nn.ReLU()(self.W_o(ainput))
        
        mol_vecs = []
        for st,le in scope:
            mol_vec = atom_hiddens.narrow(0, st, le).sum(dim=0) / le
            mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)
        return mol_vecs


import pandas as pd
import numpy as np
import random
from tqdm import tqdm

import torch
# from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import torch.nn.functional as F

import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem

ATOM_LIST = list(range(1,119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [
    BT.SINGLE, 
    BT.DOUBLE, 
    BT.TRIPLE, 
    BT.AROMATIC
]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]

class USPTO50_contrastive(Dataset):
    def __init__(self, df, return_index: bool=False, split: str='train'):
        super(USPTO50_contrastive, self).__init__()
        self.return_index = return_index
        if split == 'all':
            self.df = df
        else:
            self.df = df[df['set']==split]

    def create_graph(self, mol):
        N = mol.GetNumAtoms()
        M = mol.GetNumBonds()

        type_idx = []
        chirality_idx = []
        atomic_number = []
        # aromatic = []
        # sp, sp2, sp3, sp3d = [], [], [], []
        # num_hs = []
        for atom in mol.GetAtoms():
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            atomic_number.append(atom.GetAtomicNum())
            # aromatic.append(1 if atom.GetIsAromatic() else 0)
            # hybridization = atom.GetHybridization()
            # sp.append(1 if hybridization == HybridizationType.SP else 0)
            # sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
            # sp3.append(1 if hybridization == HybridizationType.SP3 else 0)
            # sp3d.append(1 if hybridization == HybridizationType.SP3D else 0)

        # z = torch.tensor(atomic_number, dtype=torch.long)
        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
        x = torch.cat([x1, x2], dim=-1)
        # x2 = torch.tensor([atomic_number, aromatic, sp, sp2, sp3, sp3d, num_hs],
        #                     dtype=torch.float).t().contiguous()
        # x = torch.cat([x1.to(torch.float), x2], dim=-1)

        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            # edge_type += 2 * [MOL_BONDS[bond.GetBondType()]]
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)

        # edge_index --> graph connectivity
        # edge_attr --> edge features
        # x --> node features

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        return data
    
    def get(self, index):
        reactant, product, reaction_type, _set, exclude_indices = self.df.iloc[index]
        
        anchor = product
        positive_sample = reactant
        
        # randomly pick negative sample from the dataset apart from the exluded indices
        # negative_sample_index = np.random.choice(list(set(range(len(self.df))) - set(exclude_indices)))
        negative_sample_index = random.choice([i for i in range(self.len()) if i not in exclude_indices])
        negative_sample = self.df.iloc[negative_sample_index]['reactants_mol']

        # create graph for all three molecules
        anchor_data = self.create_graph(anchor) # this is a tuple of edge_index, edge_attr, x
        positive_sample_data = self.create_graph(positive_sample)
        negative_sample_data = self.create_graph(negative_sample)

        # return data
        if self.return_index:
            return anchor_data, positive_sample_data, negative_sample_data, index
        else:
            return anchor_data, positive_sample_data, negative_sample_data # returns the graphs for all three molecules
    
    def len(self):
        return len(self.df)
    
class MoleculeDatasetWrapper(object):
    def __init__(self, batch_size, num_workers, data_path):
        super(object, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.retrieval_df = pd.read_pickle(data_path)

    def get_data_loaders(self):
        train_dataset = USPTO50_contrastive(self.retrieval_df, split='train')
        val_dataset = USPTO50_contrastive(self.retrieval_df, split='valid')
        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset, val_dataset)
        return train_loader, valid_loader

    def get_train_validation_data_loaders(self, train_dataset, val_dataset):
        # obtain training indices that will be used for validation
        # num_train = len(train_dataset)
        # indices = list(range(num_train))
        # np.random.shuffle(indices)

        # split = int(np.floor(self.valid_size * num_train))
        # train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        # train_sampler = SubsetRandomSampler(train_idx)
        # valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  num_workers=self.num_workers)

        valid_loader = DataLoader(val_dataset, batch_size=self.batch_size,
                                  num_workers=self.num_workers)
        
        # train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
        #                           num_workers=self.num_workers, drop_last=True)

        # valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
        #                           num_workers=self.num_workers, drop_last=True)

        return train_loader, valid_loader

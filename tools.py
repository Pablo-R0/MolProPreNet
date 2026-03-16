from rdkit import Chem
from torch_geometric.data import Data
import torch

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    # Create the edge_index [2, 2*num_edges]
    edge_index = [[],[]]
    for bond in mol.GetBonds():
        beginatomidx, endatomidx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index[0].extend([beginatomidx, endatomidx])
        edge_index[1].extend([endatomidx, beginatomidx])
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    # Create the nodes features [num_nodes, num_node_features]
    nodes_features = []
    for atom in mol.GetAtoms():
        atom_features = [atom.GetAtomicNum(), 
                         atom.GetDegree(), 
                         atom.GetFormalCharge(), 
                         atom.IsInRing(), 
                         atom.GetIsAromatic(), 
                         int(atom.GetHybridization()),
                         atom.GetTotalNumHs(), 
                         atom.IsInRingSize(5), 
                         atom.IsInRingSize(6)]
        nodes_features.append(atom_features)
    nodes_features = torch.tensor(nodes_features, dtype=torch.float)

    # Create the data object
    data = Data(x=nodes_features, edge_index=edge_index)
    return data
from rdkit import Chem
import pandas as pd
import numpy as np
import networkx as nx
from utils import *
from pubchemfp import GetPubChemFPs




def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr','Pt','Hg','Pb','Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetHybridization(), [
                        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2,'other']) +
                    [atom.GetIsAromatic()])
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()
    fp = []
    fp_maccs = AllChem.GetMACCSKeysFingerprint(mol)  # 167
    fp_phaErGfp = AllChem.GetErGFingerprint(mol, fuzzIncrement=0.3, maxPath=21, minPath=1)  # 441
    fp_pubcfp = GetPubChemFPs(mol)  # 881
    fp_ecfp2 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    fp.extend(fp_maccs)
    fp.extend(fp_phaErGfp)
    fp.extend(fp_pubcfp)
    fp.extend(fp_ecfp2)

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])



    return c_size, features, edge_index,fp



for i in range(1,2):
  compound_iso_smiles = []
  opts = ['train','test']
  for opt in opts:
      df = pd.read_csv(f'data_HLM/fold-{i}/' + opt + '.csv')
      compound_iso_smiles += list( df['SMILES'] )
  compound_iso_smiles = set(compound_iso_smiles)


  smile_graph = {}
  for smile in compound_iso_smiles:
      g = smile_to_graph(smile)
      smile_graph[smile] = g


  # convert to PyTorch data format
  processed_train = f'data_HLM/processed/fold-{i}/' + 'train.pt'
  processed_test = f'data_HLM/processed/fold-{i}/' + 'test.pt'
  if ((not os.path.isfile(processed_train)) or (not os.path.isfile(processed_test))):

      df = pd.read_csv(f'data_HLM/fold-{i}/' + 'train.csv')
      train_compounds = list(df['SMILES'])
      train_compounds = np.asarray(train_compounds)

      train_Y = np.array(pd.DataFrame(df['Label'])).tolist()
      df = pd.read_csv(f'data_HLM/fold-{i}/' + 'test.csv')
      test_compounds, test_Y = list(df['SMILES']), list(df['Label'])
      test_compounds= np.asarray(test_compounds)

      test_Y = np.array(pd.DataFrame(df['Label'])).tolist()

      # make data PyTorch Geometric ready
      print('preparing,' + 'train.pt in pytorch format!')
      train_data = TestbedDataset(root=f'data_HLM/fold-{i}/', dataset='train', xd=train_compounds, y=train_Y,
                                  smile_graph=smile_graph)
      print('preparing,' + 'test.pt in pytorch format!')
      test_data = TestbedDataset(root=f'data_HLM/fold-{i}/', dataset= 'test', xd=test_compounds, y=test_Y,
                                smile_graph=smile_graph)
      print(processed_train, ' and ', processed_test, ' have been created')
  else:
      print(processed_train, ' and ', processed_test, ' are already created')

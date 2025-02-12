import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import numpy as np



# 读取包含 SMILES 的三个 CSV 文件
train_data = pd.read_excel('HLM.xlsx')
#test_data = pd.read_excel('HLM-test.xlsx')
external_data = pd.read_excel('HLM-external.xlsx')

# 提取 SMILES 列
train_smiles = train_data['Smiles'].tolist()
#test_smiles = test_data['Smiles'].tolist()
external_smiles = external_data['Smiles'].tolist()

# 生成指纹并保存到列表中
def generate_fingerprints(smiles_list):
    fingerprints = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        fingerprint_array = fingerprint.ToBitString()
        fingerprints.append([int(bit) for bit in fingerprint_array])
    return fingerprints

train_fingerprints = generate_fingerprints(train_smiles)
#test_fingerprints = generate_fingerprints(test_smiles)
external_fingerprints = generate_fingerprints(external_smiles)
train_fingerprints = np.array(train_fingerprints)
#test_fingerprints = np.array(test_fingerprints)
external_fingerprints = np.array(external_fingerprints)
# 使用 t-SNE 对指纹进行降维
tsne = TSNE(n_components=2)
train_embedded = tsne.fit_transform(train_fingerprints)
#test_embedded = tsne.fit_transform(test_fingerprints)
external_embedded = tsne.fit_transform(external_fingerprints)

# 可视化降维后的数据
plt.figure(figsize=(8, 6))

# 绘制训练集数据点（红色）
plt.scatter(train_embedded[:, 0], train_embedded[:, 1], marker='o', c='r', alpha=0.5, label='HLM',s=5)

# 绘制测试集数据点（绿色）
#plt.scatter(test_embedded[:, 0], test_embedded[:, 1], marker='o', c='g', alpha=0.5, label='Test',s=5)

# 绘制外部集数据点（橙色）orange
plt.scatter(external_embedded[:, 0], external_embedded[:, 1], marker='o', c='g', alpha=0.5, label='External',s=5)


plt.legend(loc='upper left')  # 将图例放在左上角
plt.grid(False)  # 去掉网格线
plt.xticks([])  # 去掉 x 轴坐标
plt.yticks([])  # 去掉 y 轴坐标

plt.show()

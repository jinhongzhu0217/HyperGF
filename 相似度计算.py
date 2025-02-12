import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity
import matplotlib.pyplot as plt

# 读取外部数据集和人肝微粒体数据集
external_data = pd.read_excel('/Users/zhujinhong/Documents/graduation design/HyperGF/HLM-external.xlsx')
liver_mitochondria_data = pd.read_excel('/Users/zhujinhong/Documents/graduation design/HyperGF/HLM.xlsx')

# 处理数据集中的分子
def process_molecules(data):
    molecules = []
    for smi in data['Smiles']:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            molecules.append(mol)
    return molecules

external_molecules = process_molecules(external_data)
liver_mitochondria_molecules = process_molecules(liver_mitochondria_data)

# 使用ECFP计算分子指纹
def calculate_ecfp_fingerprint(mol):
    ecfp4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    return ecfp4

# 计算谷本相似度
def calculate_tanimoto_similarity(external_molecule, liver_mitochondria_molecules):
    max_similarity = 0
    fp1 = calculate_ecfp_fingerprint(external_molecule)
    for mol in liver_mitochondria_molecules:
        fp2 = calculate_ecfp_fingerprint(mol)
        similarity = TanimotoSimilarity(fp1, fp2)
        max_similarity = max(max_similarity, similarity)
    return max_similarity

# 计算每个外部数据集中分子与人肝微粒体数据集中分子的最大谷本相似度
similarities = []
for mol in external_molecules:
    similarity = calculate_tanimoto_similarity(mol, liver_mitochondria_molecules)
    similarities.append(similarity)

# 将结果添加到外部数据集中
external_data['Max_Tanimoto_Similarity'] = similarities

# 统计不同相似度类别的比例
total_count = len(similarities)
similar_count = len([sim for sim in similarities if sim > 0.7])
moderate_count = len([sim for sim in similarities if 0.5 <= sim <= 0.7])
dissimilar_count = total_count - similar_count - moderate_count

similar_percentage = (similar_count / total_count) * 100
moderate_percentage = (moderate_count / total_count) * 100
dissimilar_percentage = (dissimilar_count / total_count) * 100

average_similarity_score = sum(similarities) / total_count

# 输出结果
print("相似度类别统计：")
print(f"不相似的比例：{dissimilar_percentage:.2f}%")
print(f"中等相似的比例：{moderate_percentage:.2f}%")
print(f"高度相似的比例：{similar_percentage:.2f}%")
print(f"总体平均谷本相似度得分：{average_similarity_score:.2f}")

# 绘制柱状图
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

plt.figure(figsize=(10, 6))
plt.hist(similarities, bins=20, color='skyblue', edgecolor='black')
plt.xlabel('谷本相似度得分')
plt.ylabel('外部数据集分子数目')
plt.title('外部数据集与人肝微粒体数据集谷本相似度得分分布')
plt.grid(True)
plt.show()

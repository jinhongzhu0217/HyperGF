import os
import pandas as pd
from sklearn.model_selection import KFold

# 读取xlsx文件
file_path = 'HLM.xlsx'
df = pd.read_excel(file_path)

# 定义KFold对象，进行十折交叉验证
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# 创建保存十折数据的文件夹
output_dir = 'data_HLM_external'
os.makedirs(output_dir, exist_ok=True)

# 生成十折数据，并保存为csv文件
fold = 1
for train_index, test_index in kf.split(df):
    train_df = df.iloc[train_index]
    test_df = df.iloc[test_index]

    # 创建每一折的文件夹
    fold_dir = os.path.join(output_dir, f'fold-{fold}')
    os.makedirs(fold_dir, exist_ok=True)

    # 保存训练集和测试集到对应的文件夹
    train_df.to_csv(os.path.join(fold_dir, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(fold_dir, 'test.csv'), index=False)

    fold += 1

print("十折交叉验证的csv文件已生成。")

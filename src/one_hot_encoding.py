import numpy as np
import pandas as pd
import os

# 定义数据路径
DATA_FILE = "/Users/jaaasnyu/Desktop/DNA_Classification_Project/data/processed/E_coli_DNA_sequences_augmented.csv"
OUTPUT_X = "/Users/jaaasnyu/Desktop/DNA_Classification_Project/data/processed/X.npy"
OUTPUT_Y = "/Users/jaaasnyu/Desktop/DNA_Classification_Project/data/processed/y.npy"

# 定义 One-hot 映射字典
one_hot_map = {
    "A": [1, 0, 0, 0],
    "T": [0, 1, 0, 0],
    "C": [0, 0, 1, 0],
    "G": [0, 0, 0, 1]
}

def one_hot_encode_sequence(sequence, max_length):
    """
    将 DNA 序列转换为 One-hot 矩阵，填充到 max_length 长度
    """
    one_hot_seq = np.zeros((max_length, 4))  # 初始化矩阵
    for i, base in enumerate(sequence[:max_length]):  # 限制最大长度
        if base in one_hot_map:
            one_hot_seq[i] = one_hot_map[base]
    return one_hot_seq

def process_data():
    """
    读取 CSV，转换 DNA 序列为 One-hot 矩阵，并保存为 NumPy 数组
    """
    # **添加这行代码，确保 data/processed 目录存在**
    os.makedirs("/Users/jaaasnyu/Desktop/DNA_Classification_Project/data/processed", exist_ok=True)

    df = pd.read_csv(DATA_FILE)
    sequences = df["sequence"].values
    labels = df["label"].values

    # 设定最大序列长度（取 95% 分位数）
    max_length = int(np.percentile([len(seq) for seq in sequences], 95))
    print(f"📏 Max sequence length (95th percentile): {max_length}")

    # 进行 One-hot 编码
    X = np.array([one_hot_encode_sequence(seq, max_length) for seq in sequences])
    y = np.array(labels)

    # 保存 NumPy 数组
    np.save(OUTPUT_X, X)
    np.save(OUTPUT_Y, y)
    print(f"✅ One-hot encoded data saved: {OUTPUT_X}, {OUTPUT_Y}")

if __name__ == "__main__":
    process_data()


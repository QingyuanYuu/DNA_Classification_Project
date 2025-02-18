import numpy as np
import os
import matplotlib.pyplot as plt

# 定义数据路径
TRAIN_X = "/Users/jaaasnyu/Desktop/DNA_Classification_Project/data/processed/X_train.npy"
TEST_X = "/Users/jaaasnyu/Desktop/DNA_Classification_Project/data/processed/X_test.npy"
TRAIN_Y = "/Users/jaaasnyu/Desktop/DNA_Classification_Project/data/processed/y_train.npy"
TEST_Y = "/Users/jaaasnyu/Desktop/DNA_Classification_Project/data/processed/y_test.npy"

def load_data():
    """ 加载 NumPy 数据 """
    if not all(os.path.exists(f) for f in [TRAIN_X, TEST_X, TRAIN_Y, TEST_Y]):
        print("❌ 数据文件未找到，请先运行 split_data.py")
        return
    
    X_train = np.load("/Users/jaaasnyu/Desktop/DNA_Classification_Project/data/processed/X_train.npy")
    X_test = np.load(TEST_X)
    y_train = np.load(TRAIN_Y)
    y_test = np.load(TEST_Y)

    # 打印数据形状
    print("✅ 数据加载成功！")
    print(f"🔹 X_train shape: {X_train.shape}")  # (样本数, 序列长度, 4)
    print(f"🔹 X_test shape: {X_test.shape}")    # (样本数, 序列长度, 4)
    print(f"🔹 y_train shape: {y_train.shape}")  # (样本数,)
    print(f"🔹 y_test shape: {y_test.shape}")    # (样本数,)

    # 检查类别分布
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"📊 训练集类别分布: {dict(zip(unique, counts))}")
    unique, counts = np.unique(y_test, return_counts=True)
    print(f"📊 测试集类别分布: {dict(zip(unique, counts))}")

    # 可视化一个 DNA 序列的 One-hot 编码
    plt.imshow(X_train[0].T, cmap="viridis", aspect="auto")
    plt.colorbar(label="One-hot Encoding")
    plt.xlabel("DNA Sequence Position")
    plt.ylabel("A, T, C, G")
    plt.title("Example One-hot Encoding of DNA Sequence")
    plt.show()

if __name__ == "__main__":
    load_data()

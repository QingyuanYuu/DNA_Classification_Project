import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 数据路径
TRAIN_X = "/Users/jaaasnyu/Desktop/DNA_Classification_Project/data/processed/X_train.npy"
TRAIN_Y = "/Users/jaaasnyu/Desktop/DNA_Classification_Project/data/processed/y_train.npy"
TEST_X = "/Users/jaaasnyu/Desktop/DNA_Classification_Project/data/processed/X_test.npy"
TEST_Y = "/Users/jaaasnyu/Desktop/DNA_Classification_Project/data/processed/y_test.npy"

class DNADataset(Dataset):
    """ 自定义 Dataset for DNA Classification """
    def __init__(self, X_path, y_path):
        self.X = np.load(X_path)  # 形状 (样本数, 1773, 4)
        self.y = np.load(y_path)  # 形状 (样本数,)
        
        # 转换为 PyTorch 张量
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_dataloaders(batch_size=32):
    """
    返回训练 & 测试 DataLoader
    """
    train_dataset = DNADataset(TRAIN_X, TRAIN_Y)
    test_dataset = DNADataset(TEST_X, TEST_Y)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

if __name__ == "__main__":
    # 测试 DataLoader
    train_loader, test_loader = get_dataloaders()
    for X_batch, y_batch in train_loader:
        print(f"📦 Batch X shape: {X_batch.shape}, y shape: {y_batch.shape}")
        break

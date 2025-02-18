import torch
import torch.nn as nn
import torch.nn.functional as F

class DNA_CNN(nn.Module):
    def __init__(self, input_length=1773, num_classes=2):
        super(DNA_CNN, self).__init__()

        # 1D 卷积层
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=64, kernel_size=5, padding="same")
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding="same")
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding="same")
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        # 计算 Flatten 后的特征数量
        final_length = input_length // (2 * 2 * 2)  # 经过3次池化
        self.fc1 = nn.Linear(256 * final_length, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # PyTorch 的 Conv1D 需要 (batch, channels, seq_length)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        x = x.view(x.shape[0], -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

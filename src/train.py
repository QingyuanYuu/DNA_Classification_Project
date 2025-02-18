import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from model import DNA_CNN
from data_loader import get_dataloaders

# 训练参数
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.0005

# 设备选择（支持 MPS 加速）
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# 加载数据
train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)

# 初始化模型
model = DNA_CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 记录训练过程
train_losses = []
train_accuracies = []
test_accuracies = []

def train():
    """ 训练 CNN """
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # 前向传播
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 计算准确率
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

        # 计算训练损失 & 准确率
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total * 100
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)
        print(f"📌 Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.2f}%")

        # 每 5 轮评估一次
        if (epoch + 1) % 5 == 0:
            test_accuracy = evaluate()
            test_accuracies.append(test_accuracy)

def evaluate():
    """ 评估模型 """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

    accuracy = correct / total * 100
    print(f"🎯 Test Accuracy: {accuracy:.2f}%")
    return accuracy

def plot_training():
    """ 绘制训练过程 """
    plt.figure(figsize=(10, 4))

    # 绘制训练损失
    plt.subplot(1, 2, 1)
    plt.plot(range(1, EPOCHS + 1), train_losses, label="Training Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()

    # 绘制准确率
    plt.subplot(1, 2, 2)
    plt.plot(range(1, EPOCHS + 1), train_accuracies, label="Training Accuracy", marker="o")
    plt.plot(range(5, EPOCHS + 1, 5), test_accuracies, label="Test Accuracy", marker="s", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Curve")
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train()
    plot_training()
    # 保存模型
    torch.save(model.state_dict(), "/Users/jaaasnyu/Desktop/DNA_Classification_Project/results/dna_cnn_model.pth")
    print("✅ Model saved to /Users/jaaasnyu/Desktop/DNA_Classification_Project/results/dna_cnn_model.pth")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
import os
import numpy as np

# ================= 1. 加载配置与数据 =================
# 建议与之前的参数保持一致
K = 3
N = 100
Ts = 0.1
T_MAX = (N - 1) * Ts
SIGMA_H = 0.15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建模型保存目录
if not os.path.exists('models'): os.makedirs('models')


def load_data():
    if not os.path.exists('datasets/fri_train_data.pt'):
        raise FileNotFoundError("找不到数据集，请先运行 data_generation.py")

    data = torch.load('datasets/fri_train_data.pt')
    full_dataset = TensorDataset(data['input'], data['tk'], data['ak'])

    # 按照 8:2 划分训练集和验证集 (符合项目基准要求)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    return train_loader, val_loader


# ================= 2. 定义网络架构 (对应技术路线) =================
class FRINet(nn.Module):
    def __init__(self, input_dim=100, K=3):
        super(FRINet, self).__init__()
        # 编码器：提取采样信号特征
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        # 输出头：预测位置 tk (0-9.9) 和 幅度 ak (>0)
        self.tk_head = nn.Sequential(nn.Linear(256, K), nn.Sigmoid())
        self.ak_head = nn.Sequential(nn.Linear(256, K), nn.ReLU())

    def forward(self, x):
        feat = self.encoder(x)
        # 将 Sigmoid 输出映射到 [0, T_MAX]
        tk = self.tk_head(feat) * T_MAX
        ak = self.ak_head(feat) + 0.1  # 加上微小的偏置防止幅度为0
        return tk, ak


# ================= 3. 训练与基准评估逻辑 =================
def train():
    train_loader, val_loader = load_data()
    model = FRINet(input_dim=N, K=K).to(DEVICE)

    # 对应 PDF 第 5 页：使用 Adam 优化器
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()

    # 学习率调度：训练后期减小步长，使收敛更精细
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    epochs = 100
    train_losses = []
    val_rmses = []

    print(f"开始训练，设备: {DEVICE}")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_y, batch_tk, batch_ak in train_loader:
            batch_y, batch_tk, batch_ak = batch_y.to(DEVICE), batch_tk.to(DEVICE), batch_ak.to(DEVICE)

            optimizer.zero_grad()
            pred_tk, pred_ak = model(batch_y)

            # 关键：排序对齐 (解决 Permutation Invariance 问题)
            # 预测的位置和真实的脉冲位置必须排序后计算误差
            pred_tk_sorted, _ = torch.sort(pred_tk, dim=1)
            pred_ak_sorted, _ = torch.sort(pred_ak, dim=1)  # 假设 ak 随 tk 排序

            loss_tk = criterion(pred_tk_sorted, batch_tk)
            loss_ak = criterion(pred_ak_sorted, batch_ak)
            loss = loss_tk + 0.5 * loss_ak

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # 验证集基准评估 (Benchmark)
        model.eval()
        val_rmse = 0
        with torch.no_grad():
            for v_y, v_tk, _ in val_loader:
                v_y, v_tk = v_y.to(DEVICE), v_tk.to(DEVICE)
                p_tk, _ = model(v_y)
                p_tk_sorted, _ = torch.sort(p_tk, dim=1)
                val_rmse += torch.sqrt(criterion(p_tk_sorted, v_tk)).item()

        avg_train_loss = epoch_loss / len(train_loader)
        avg_val_rmse = val_rmse / len(val_loader)

        train_losses.append(avg_train_loss)
        val_rmses.append(avg_val_rmse)

        scheduler.step(avg_val_rmse)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{epochs}] | Train Loss: {avg_train_loss:.6f} | Val RMSE (tk): {avg_val_rmse:.4f}")

    # 保存最终模型
    torch.save(model.state_dict(), 'models/fri_model.pth')
    print("训练完成！模型已保存至 models/fri_model.pth")

    # 绘制训练过程曲线 (用于第14-16周报告撰写)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train MSE Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_rmses, color='orange', label='Val RMSE (tk)')
    plt.title('Validation Accuracy (tk)')
    plt.xlabel('Epoch')
    plt.legend()

    if not os.path.exists('results'): os.makedirs('results')
    plt.savefig('results/training_metrics.png')
    plt.show()


if __name__ == "__main__":
    train()
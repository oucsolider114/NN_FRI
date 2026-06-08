import sys
sys.path.insert(0, 'PythonProject1')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
import os
from data_generation import generate_fri_dataset
from model_arch import FRINet

# ================= 配置参数 =================
K = 3
N = 100
Ts = 0.1
T_MAX = (N - 1) * Ts
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = 'datasets/fri_train_data.pt'
MODEL_DIR = 'models'
RESULT_DIR = 'results'

for d in [MODEL_DIR, RESULT_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)


def load_data():
    if not os.path.exists(DATA_PATH):
        print(f"数据集不存在，自动生成中...")
        generate_fri_dataset(10000)

    data = torch.load(DATA_PATH)
    full_dataset = TensorDataset(data['input'], data['tk'], data['ak'])

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    return train_loader, val_loader


def train():
    train_loader, val_loader = load_data()
    model = FRINet(input_dim=N, K=K).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()
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

            pred_tk_sorted, _ = torch.sort(pred_tk, dim=1)
            pred_ak_sorted, _ = torch.sort(pred_ak, dim=1)

            loss_tk = criterion(pred_tk_sorted, batch_tk)
            loss_ak = criterion(pred_ak_sorted, batch_ak)
            loss = loss_tk + 0.5 * loss_ak

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

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
            print(f"Epoch [{epoch + 1}/{epochs}] | Train Loss: {avg_train_loss:.6f} | Val RMSE (tk): {avg_val_rmse:.4f}")

    torch.save(model.state_dict(), f'{MODEL_DIR}/fri_model.pth')
    print(f"训练完成！模型已保存至 {MODEL_DIR}/fri_model.pth")

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

    plt.savefig(f'{RESULT_DIR}/training_metrics.png')
    plt.show()


if __name__ == "__main__":
    train()

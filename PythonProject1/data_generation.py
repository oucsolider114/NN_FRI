import torch
import numpy as np
import os
import matplotlib.pyplot as plt

# ================= 配置参数  =================
K = 3  # 脉冲个数
N = 100  # 采样点数
Ts = 0.1  # 采样间隔
T_MAX = (N - 1) * Ts
SIGMA_H = 0.15  # 采样核宽度
NUM_SAMPLES = 10000  # 生成1万组数据用于后期训练
SNR_RANGE = (5, 30)  # 噪声范围 5dB 到 30dB (数据增强)


def generate_fri_dataset(num_samples):
    inputs = []
    labels_tk = []
    labels_ak = []

    print(f"开始生成 {num_samples} 组 FRI 信号数据...")

    for i in range(num_samples):
        # 1. 随机生成脉冲参数 (Label)
        # tk 随机分布在 [0.1T, 0.9T]，ak 在 [0.5, 1.5]
        tk = torch.sort(torch.rand(K) * T_MAX * 0.8 + T_MAX * 0.1)[0]
        ak = torch.rand(K) + 0.5

        # 2. 生成无噪声测量值 y[n] = sum ak * h(nTs - tk)
        n_grid = torch.arange(N) * Ts
        y_noiseless = torch.zeros(N)
        for k in range(K):
            diff = n_grid - tk[k]
            y_noiseless += ak[k] * torch.exp(-diff ** 2 / (2 * SIGMA_H ** 2))

        # 3. 添加随机 SNR 的噪声 (数据增强：让网络适应不同环境)
        snr = np.random.uniform(SNR_RANGE[0], SNR_RANGE[1])
        p_signal = torch.mean(y_noiseless ** 2)
        p_noise = p_signal / (10 ** (snr / 10))
        noise = torch.randn(N) * torch.sqrt(p_noise)
        y_noisy = y_noiseless + noise

        inputs.append(y_noisy)
        labels_tk.append(tk)
        labels_ak.append(ak)

        # 4. 可视化第一组数据，确认模型正确性
        if i == 0:
            plt.figure(figsize=(10, 4))
            plt.plot(n_grid.numpy(), y_noisy.numpy(), 'b.', label='Noisy Sample (Input)')
            plt.stem(tk.numpy(), ak.numpy(), 'r', markerfmt='rx', label='Ground Truth (Label)')
            plt.title(f"Sample 0 Preview (SNR: {snr:.1f}dB)")
            plt.legend()
            if not os.path.exists('results'): os.makedirs('results')
            plt.savefig('results/data_preview.png')
            print("预览图已保存至 results/data_preview.png")

    # 5. 保存数据集为 PyTorch 格式
    if not os.path.exists('datasets'): os.makedirs('datasets')
    dataset = {
        'input': torch.stack(inputs),  # 形状: [10000, 100]
        'tk': torch.stack(labels_tk),  # 形状: [10000, 3]
        'ak': torch.stack(labels_ak)  # 形状: [10000, 3]
    }
    torch.save(dataset, 'datasets/fri_train_data.pt')
    print(f"数据集已成功保存至 datasets/fri_train_data.pt")


if __name__ == "__main__":
    generate_fri_dataset(NUM_SAMPLES)
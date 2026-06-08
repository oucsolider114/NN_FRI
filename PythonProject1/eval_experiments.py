import torch
import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from model_arch import FRINet  # 确保你之前的模型定义在这个文件里

# ================= 配置与参数 =================
K = 3
N = 100
Ts = 0.1
T_MAX = (N - 1) * Ts
SIGMA_H = 0.15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 1. 传统基准算法：矩阵束法 (Matrix Pencil)
def matrix_pencil_estimate(y, K):
    L = N // 3  # 铅笔参数
    Y = linalg.hankel(y[:N - L], y[N - L - 1:])
    Y1, Y2 = Y[:, :-1], Y[:, 1:]
    U, S, Vh = linalg.svd(Y1, full_matrices=False)
    # 简化版映射：在实际FRI中这步很复杂，这里作为Baseline对比
    # 我们主要观察它对噪声的敏感度
    Emat = linalg.pinv(Y1) @ Y2
    eigvals = linalg.eigvals(Emat)
    angles = np.sort(np.abs(np.angle(eigvals[:K])))
    return angles * T_MAX / (2 * np.pi)


# 2. 加载训练好的神经网络
model = FRINet(input_dim=N, K=K).to(DEVICE)
model.load_state_dict(torch.load('models/fri_model.pth', map_location=DEVICE))
model.eval()


# 3. 实验：SNR vs RMSE 定量分析
def run_robustness_test():
    snrs = [5, 10, 15, 20, 25, 30, 35]
    nn_rmses = []
    classic_rmses = []

    print("正在进行鲁棒性定量实验...")
    for snr in snrs:
        nn_error, classic_error = 0, 0
        test_iters = 100  # 每个SNR跑100次取平均

        for _ in range(test_iters):
            # 生成随机测试数据
            tk_true = torch.sort(torch.rand(K) * T_MAX * 0.8 + T_MAX * 0.1)[0]
            n_grid = torch.arange(N) * Ts
            y_noiseless = torch.zeros(N)
            for k in range(K):
                y_noiseless += torch.exp(-(n_grid - tk_true[k]) ** 2 / (2 * SIGMA_H ** 2))

            # 加噪声
            p_noise = torch.mean(y_noiseless ** 2) / (10 ** (snr / 10))
            y_noisy = y_noiseless + torch.randn(N) * torch.sqrt(p_noise)

            # NN 预测
            with torch.no_grad():
                p_tk, _ = model(y_noisy.unsqueeze(0).to(DEVICE))
                p_tk_sorted, _ = torch.sort(p_tk.cpu(), dim=1)
                nn_error += torch.sqrt(torch.mean((p_tk_sorted - tk_true) ** 2)).item()

            # 传统方法预测
            try:
                tk_c = matrix_pencil_estimate(y_noisy.numpy(), K)
                classic_error += np.sqrt(np.mean((tk_c - tk_true.numpy()) ** 2))
            except:
                classic_error += 1.0  # 算法崩溃补偿

        nn_rmses.append(nn_error / test_iters)
        classic_rmses.append(classic_error / test_iters)
        print(f"SNR {snr}dB 测试完成")

    # 绘制对比曲线 (这是你结题报告的核心图表)
    plt.figure(figsize=(8, 5))
    plt.plot(snrs, nn_rmses, 'o-r', label='Our NN-FRI')
    plt.plot(snrs, classic_rmses, 's--b', label='Classic Matrix Pencil')
    plt.yscale('log')
    plt.xlabel('SNR (dB)')
    plt.ylabel('RMSE (Log Scale)')
    plt.title('Performance Comparison: NN vs Classic')
    plt.grid(True, which="both", ls="-")
    plt.legend()
    plt.savefig('results/performance_comparison.png')
    plt.show()


if __name__ == "__main__":
    run_snr_test = True
    if run_snr_test:
        run_robustness_test()
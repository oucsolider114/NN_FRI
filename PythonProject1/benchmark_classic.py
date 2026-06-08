import numpy as np
import scipy.linalg as linalg
from config import Config


def matrix_pencil_method(y, K, Ts):
    """
    传统基准算法：矩阵束法 (Matrix Pencil Method)
    用于从采样点 y 中直接估计脉冲的位置 tk
    """
    N = len(y)
    L = N // 2  # 铅笔参数

    # 1. 构造数据矩阵 Y1 和 Y2
    Y = linalg.hankel(y[:N - L], y[N - L - 1:])
    Y1 = Y[:, :-1]
    Y2 = Y[:, 1:]

    # 2. 奇异值分解 (SVD) 并降维到 K
    U, S, Vh = linalg.svd(Y1, full_matrices=False)
    U_k = U[:, :K]

    # 3. 求解广义特征值
    # 这一步对应 PDF 补充材料中提到的“子空间交换”易发区
    Y1_inv = linalg.pinv(Y1)
    Emat = Y1_inv @ Y2
    eigenvalues = linalg.eigvals(Emat)

    # 4. 从特征值映射回时间 tk
    # 这里是一个简化模型，实际 FRI 需要零化滤波器转换，
    # 我们取相位最接近的部分作为估计
    angles = np.angle(eigenvalues[:K])
    est_tk = np.sort(np.abs(angles) * Config.T_MAX / (2 * np.pi))

    return est_tk


# 测试一下基准算法
if __name__ == "__main__":
    from utils import generate_fri_data

    n_grid, y_meas, tk_true, ak_true = generate_fri_data(snr_db=30)
    tk_classic = matrix_pencil_method(y_meas.numpy(), Config.K, Config.Ts)
    print(f"真实位置: {tk_true.numpy()}")
    print(f"传统基准预测: {tk_classic}")
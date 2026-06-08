# traditional_methods.py
import numpy as np
from scipy.linalg import hankel, pinv, eig

def matrix_pencil(y, K, Ts=1.0, pencil_param=None):
    """
    矩阵束法估计 FRI 信号的时延和幅度
    
    参数:
        y   : 测量值向量 (N,)
        K   : 已知的脉冲个数
        Ts  : 采样间隔
        pencil_param : 束参数 L，默认 N//2
    
    返回:
        tau_est : 估计的时延 (K,)
        a_est   : 估计的幅度 (K,)
    """
    N = len(y)
    L = pencil_param if pencil_param is not None else N // 2
    
    # 构造 Hankel 矩阵
    # Y = [y[0]   y[1]   ... y[L-1]
    #      y[1]   y[2]   ... y[L]
    #      ...
    #      y[N-L] y[N-L+1] ... y[N-1]]
    Y = hankel(y[:L], y[L-1:])
    
    # SVD
    U, s, Vh = np.linalg.svd(Y, full_matrices=False)
    
    # 取前 K 个主奇异向量
    U_K = U[:, :K]
    V_K = Vh[:K, :].conj().T  # (L, K)
    s_K = s[:K]
    
    # 修正：使用标准的降秩近似
    # Y_up = U[0:N-L, :K] * s[:K] @ Vh[:K, 1:]  (去掉最后一列)
    # Y_down = U[1:N-L+1, :K] * s[:K] @ Vh[:K, :-1] (去掉第一列)
    Y_up = U_K[:-1, :] @ (np.diag(s_K) @ V_K[1:, :].conj().T) if V_K.shape[0] > 1 else U_K[:-1, :] @ np.diag(s_K)
    Y_down = U_K[1:, :] @ (np.diag(s_K) @ V_K[:-1, :].conj().T) if V_K.shape[0] > 1 else U_K[1:, :] @ np.diag(s_K)
    
    # 广义特征值分解：求解 Y_l * v = λ * Y_f * v 的广义特征值
    # 等价于 pinv(Y_f) @ Y_l 的特征值
    try:
        M = np.linalg.pinv(Y_up) @ Y_down
        eigenvalues = np.linalg.eigvals(M)
    except np.linalg.LinAlgError:
        eigenvalues = np.array([])
    
    # 取模最大的 K 个特征值（应在单位圆内）
    # 但实际中特征值可能在单位圆附近
    idx = np.argsort(np.abs(eigenvalues))[-K:]
    z_K = eigenvalues[idx]
    
    # 限制在单位圆内（数值稳定）
    z_K = z_K / np.abs(z_K)
    
    # 从特征值恢复时延
    # z_k = e^{-α Ts} ... 但我们用的是高斯脉冲等，不是复指数
    # 这里需要针对高斯脉冲做调整
    # 对于一般脉冲，矩阵束法要求脉冲是复指数形式
    # 所以其对高斯脉冲的效果有限
    # 这里给出通用版本
    tau_est = -Ts * np.angle(z_K) / (2 * np.pi)
    tau_est = np.sort(tau_est % 1.0)  # 折合到 [0, 1]
    
    # 估计幅度：最小二乘
    # 构造基矩阵
    t_samples = np.arange(N) * Ts
    G = np.zeros((N, K))
    sigma_default = 0.08  # 脉冲宽度（需已知）
    for k in range(K):
        G[:, k] = np.exp(-(t_samples - tau_est[k])**2 / (2 * sigma_default**2))
    
    try:
        a_est = np.linalg.lstsq(G, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        a_est = np.zeros(K)
    
    return tau_est, a_est


def estimate_with_known_K(y, K, Ts=1.0):
    """包装：调用矩阵束法"""
    return matrix_pencil(y, K, Ts)


def estimate_with_unknown_K(y, max_K=10, Ts=1.0):
    """
    当 K 未知时，通过奇异值衰减估计 K，再调用矩阵束法
    """
    N = len(y)
    L = N // 2
    Y = hankel(y[:L], y[L-1:])
    U, s, Vh = np.linalg.svd(Y, full_matrices=False)
    
    # 简单阈值法估计 K
    s_normalized = s / (s[0] + 1e-10)
    K_est = np.sum(s_normalized > 0.05)
    K_est = max(1, min(K_est, max_K))
    
    return matrix_pencil(y, K_est, Ts), K_est
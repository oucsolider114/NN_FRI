# fri_net.py
import torch
import torch.nn as nn
import numpy as np

class ContinuousAtomNet(nn.Module):
    """
    FRINET 核心模型：过参数化连续原子稀疏恢复
    
    参数:
        N       : 采样点数（输入长度，如 21）
        M       : 过参数化神经元个数（如 100，远大于真实脉冲数 K）
        T_end   : 观测时间窗口长度
        pulse_fn: 脉冲原子函数 (callable, 必须支持 PyTorch 张量)
    """
    
    def __init__(self, N, M, T_end=1.0, pulse_fn=None, tau_init='uniform'):
        super().__init__()
        self.N = N
        self.M = M
        self.T_end = T_end
        
        # 脉冲原子（用闭包传入，或内置高斯函数）
        if pulse_fn is None:
            self.pulse_fn = self._gaussian_pulse
        else:
            self.pulse_fn = pulse_fn
        
        # ── 可训练参数 ──
        # 时延 τ_i ∈ [0, T_end]，初始化为均匀分布
        if tau_init == 'uniform':
            tau_init_vals = torch.linspace(0.1*T_end, 0.9*T_end, M) \
                           + torch.randn(M) * 0.02  # 加点随机性
        else:
            tau_init_vals = torch.rand(M) * T_end
        # 用 sigmoid 映射保证 τ ∈ (0, T_end)
        self.tau_raw = nn.Parameter(
            self._inv_sigmoid(tau_init_vals / T_end)
        )
        
        # 幅度 a_i，初始化为小值
        a_init = torch.randn(M) * 0.01
        self.a_raw = nn.Parameter(a_init)
        
        # 采样时间点（固定的，不作为参数优化）
        t_samples = torch.linspace(0, T_end, N + 1)[:-1]  # 等价于 endpoint=False
        self.register_buffer('t_samples', t_samples)
        
        # 记录一下 τ 的初始分布（调试用）
        self.pulse_fn_name = getattr(pulse_fn, '__name__', 'custom')
    
    def _inv_sigmoid(self, x, eps=1e-6):
        """反 sigmoid，用于初始化：把 (0,1) 映射到 R"""
        x = torch.clamp(x, eps, 1-eps)
        return torch.log(x / (1 - x))
    
    def _gaussian_pulse(self, t, sigma=0.08):
        """默认高斯原子"""
        return torch.exp(-t**2 / (2 * sigma**2))
    
    def forward(self):
        """
        前向传播：根据当前 τ, a 生成重构信号
        
        返回:
            y_hat:   重构信号 (N,)
            tau:     当前时延 (M,) [0, T_end]
            a:       当前幅度 (M,)
        """
        # 将原始参数映射到有效范围
        tau = torch.sigmoid(self.tau_raw) * self.T_end  # (M,)
        a = self.a_raw                                   # (M,)  （允许负幅度）
        
        # 计算所有原子在所有采样点上的值
        # t_diff: (N, M) = 每个采样时间 - 每个时延
        t_diff = self.t_samples.unsqueeze(1) - tau.unsqueeze(0)  # (N, M)
        
        # 应用脉冲函数（元素级）
        atoms = self.pulse_fn(t_diff)  # (N, M)
        
        # 加权求和得到重构信号
        y_hat = atoms @ a  # (N,) = (N,M) @ (M,)
        
        return y_hat, tau, a
    
    def get_sparsity(self, threshold=1e-3):
        """统计有效脉冲数（|a| > threshold 的个数）"""
        with torch.no_grad():
            tau = torch.sigmoid(self.tau_raw) * self.T_end
            a = self.a_raw
            active = (a.abs() > threshold).sum().item()
        return active
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── 测试 ───
if __name__ == '__main__':
    model = ContinuousAtomNet(N=21, M=100, T_end=1.0)
    print(f"模型参数量: {model.count_parameters()} 个标量参数")
    print(f"  tau_raw 形状: {model.tau_raw.shape}")
    print(f"  a_raw 形状:   {model.a_raw.shape}")
    
    y_hat, tau, a = model.forward()
    print(f"  输出 y_hat 形状: {y_hat.shape}")  # 应为 (21,)
    print(f"  tau 范围:       [{tau.min().item():.4f}, {tau.max().item():.4f}]")
    print(f"  a 的稀疏度:     {model.get_sparsity()} / {model.M}")
    
    # 可视化初始状态
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,4))
    plt.stem(model.t_samples.numpy(), y_hat.detach().numpy(), 
             linefmt='b-', markerfmt='bo', basefmt='k-', label='Initial y_hat')
    plt.title(f'Network output before training (M={model.M})')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()
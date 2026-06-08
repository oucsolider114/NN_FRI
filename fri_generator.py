# fri_generator.py
import numpy as np
from scipy.interpolate import BSpline

class PulseLibrary:
    """脉冲原子库"""
    
    @staticmethod
    def gaussian(t, sigma=0.1):
        """高斯脉冲"""
        return np.exp(-t**2 / (2 * sigma**2))
    
    @staticmethod
    def exponential(t, tau_decay=0.05):
        """指数衰减脉冲（单边）"""
        return np.exp(-t / tau_decay) * (t >= 0)
    
    @staticmethod
    def bspline(t, order=3, support=0.2):
        """B 样条脉冲（三次）"""
        # B 样条本质上是矩形脉冲的多次卷积
        t_scaled = t / (support / (order + 1))  # 缩放到单位支撑
        result = np.zeros_like(t)
        for k in range(len(t)):
            # 递归 DE_Boor 算法（简化版：用均匀 B 样条的解析式）
            x = t_scaled[k] + (order + 1) / 2  # 移到正区间
            if 0 <= x <= order + 1:
                result[k] = UniformBSpline(x, order)
        return result

def UniformBSpline(x, order):
    """均匀 B 样条基础函数（递归）"""
    if order == 0:
        return 1.0 if 0 <= x < 1 else 0.0
    return (x / order) * UniformBSpline(x, order-1) + \
           ((order+1 - x) / order) * UniformBSpline(x-1, order-1)


class FRISignalGenerator:
    """
    FRI 信号生成器
    - K: 脉冲个数
    - T_end: 观测时长
    - N: 低速采样点数
    - snr_db: 信噪比
    - pulse_type: 'gaussian' | 'exponential' | 'bspline'
    - pulse_params: 脉冲参数 dict
    """
    
    def __init__(self, K=3, T_end=1.0, N=21, snr_db=10, 
                 pulse_type='gaussian', pulse_params=None):
        self.K = K
        self.T_end = T_end
        self.N = N
        self.snr_db = snr_db
        self.pulse_type = pulse_type
        self.pulse_params = pulse_params or {'sigma': 0.1}
        
        # 选择脉冲函数
        if pulse_type == 'gaussian':
            self.pulse_fn = lambda t: PulseLibrary.gaussian(t, **self.pulse_params)
        elif pulse_type == 'exponential':
            self.pulse_fn = lambda t: PulseLibrary.exponential(t, **self.pulse_params)
        elif pulse_type == 'bspline':
            self.pulse_fn = lambda t: PulseLibrary.bspline(t, **self.pulse_params)
        else:
            raise ValueError(f"Unknown pulse type: {pulse_type}")
    
    def generate_one(self, seed=None):
        """
        生成一个样本
        返回: {
            't_samples': 采样时间点 (N,),
            'y_noisy':   含噪采样值 (N,),
            'y_clean':   无噪采样值 (N,),
            'tau_true':  真实时延 (K,),
            'a_true':    真实幅度 (K,),
        }
        """
        if seed is not None:
            np.random.seed(seed)
        
        # 真实参数（时延在 [0.2T_end, 0.8T_end] 内避免边界效应）
        tau = np.sort(np.random.uniform(0.2*self.T_end, 0.8*self.T_end, self.K))
        a = np.random.uniform(0.5, 1.5, self.K)
        # 有时加个符号变化，测试模型能否处理
        a *= np.random.choice([-1, 1], size=self.K)
        
        # 采样时刻
        t_s = np.linspace(0, self.T_end, self.N, endpoint=False)
        
        # 计算无噪信号
        y_clean = np.zeros(self.N)
        for k in range(self.K):
            y_clean += a[k] * self.pulse_fn(t_s - tau[k])
        
        # 添加高斯白噪声
        signal_power = np.mean(y_clean**2)
        noise_power = signal_power / (10**(self.snr_db/10))
        noise = np.sqrt(noise_power) * np.random.randn(self.N)
        y_noisy = y_clean + noise
        
        return {
            't_samples': t_s,
            'y_noisy': y_noisy,
            'y_clean': y_clean,
            'tau_true': tau,
            'a_true': a,
        }
    
    def generate_batch(self, num_samples, seed_base=None):
        """批量生成，返回列表"""
        samples = []
        for i in range(num_samples):
            seed = None if seed_base is None else seed_base + i
            samples.append(self.generate_one(seed=seed))
        return samples


# ─── 测试代码 ───
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # 测试三种脉冲类型
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for ax, ptype, params in zip(
        axes,
        ['gaussian', 'exponential', 'bspline'],
        [{'sigma': 0.08}, {'tau_decay': 0.06}, {'order': 3, 'support': 0.25}]
    ):
        gen = FRISignalGenerator(K=3, N=21, snr_db=15, pulse_type=ptype, pulse_params=params)
        s = gen.generate_one(seed=42)
        ax.stem(s['t_samples'], s['y_noisy'], linefmt='r-', markerfmt='ro', basefmt='k-')
        ax.plot(s['t_samples'], s['y_clean'], 'b--')
        for tk in s['tau_true']:
            ax.axvline(tk, color='gray', ls=':', alpha=0.6)
        ax.set_title(f'{ptype} pulse')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
    
    plt.tight_layout()
    plt.savefig('fri_signal_examples.png', dpi=150)
    plt.show()
    
    print("✅ FRI 信号生成器测试完毕！")
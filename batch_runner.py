import torch
import numpy as np
import time
from tqdm import tqdm
from fri_net import ContinuousAtomNet
from fri_generator import FRISignalGenerator
import warnings

class FRINETBatchRunner:
    """
    批量运行 FRINET 优化并收集结果
    """
    
    def __init__(self, N=21, M=100, T_end=1.0, 
                 pulse_type='gaussian', pulse_params=None,
                 lam=0.01, lr=0.01, steps=1000, 
                 device='cpu'):
        self.N = N
        self.M = M
        self.T_end = T_end
        self.lam = lam
        self.lr = lr
        self.steps = steps
        self.device = device
        
        # 脉冲函数
        self.pulse_params = pulse_params or {'sigma': 0.08}
        if pulse_type == 'gaussian':
            sigma = self.pulse_params['sigma']
            self.pulse_fn = lambda t: torch.exp(-t**2 / (2 * sigma**2))
        elif pulse_type == 'exponential':
            tau_d = self.pulse_params.get('tau_decay', 0.05)
            self.pulse_fn = lambda t: torch.exp(-t / tau_d) * (t >= 0).float()
        else:
            raise ValueError(f"Unsupported pulse type: {pulse_type}")
        
        self.pulse_fn_name = pulse_type
    
    def optimize_one(self, y_noisy, verbose=False):
        """
        对单个含噪采样信号运行 FRINET 优化
        返回: {tau_pred, a_pred, losses, time_elapsed}
        """
        y_tensor = torch.tensor(y_noisy, dtype=torch.float32, device=self.device)
        
        # 创建模型（每次新建，保证随机初始化）
        model = ContinuousAtomNet(
            N=self.N, M=self.M, T_end=self.T_end, 
            pulse_fn=self.pulse_fn
        ).to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        
        losses = []
        start_time = time.time()
        
        for step in range(self.steps):
            optimizer.zero_grad()
            y_hat, tau, a = model.forward()
            
            mse = ((y_hat - y_tensor)**2).mean()
            l1_reg = self.lam * torch.abs(a).sum()
            loss = mse + l1_reg
            
            loss.backward()
            # 梯度裁剪（防止偶尔的梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            
            losses.append(loss.item())
            
            if verbose and step % 200 == 0:
                active = (torch.abs(a) > 0.05).sum().item()
                print(f"  Step {step}: loss={loss.item():.6f}, active atoms={active}")
        
        elapsed = time.time() - start_time
        
        # 提取预测
        model.eval()
        with torch.no_grad():
            _, tau, a = model.forward()
            # 合并非常接近的原子（聚类）
            tau_pred, a_pred = self._cluster_predictions(tau, a)
        
        return {
            'tau_pred': tau_pred,
            'a_pred': a_pred,
            'losses': losses,
            'time': elapsed,
            'final_active': len(tau_pred),
        }
    
    def _cluster_predictions(self, tau, a, threshold=0.05, min_dist=0.03):
        """
        聚类相近的原子，避免同一脉冲被多个原子表示
        """
        # 只保留幅度 > threshold 的原子
        mask = torch.abs(a) > threshold
        tau_active = tau[mask].detach().cpu().numpy()
        a_active = a[mask].detach().cpu().numpy()
        
        if len(tau_active) == 0:
            return np.array([]), np.array([])
        
        # 按 tau 排序
        order = np.argsort(tau_active)
        tau_sorted = tau_active[order]
        a_sorted = a_active[order]
        
        # 聚类
        clusters_tau = []
        clusters_a = []
        current_taus = [tau_sorted[0]]
        current_as = [a_sorted[0]]
        
        for i in range(1, len(tau_sorted)):
            if tau_sorted[i] - current_taus[-1] < min_dist:
                current_taus.append(tau_sorted[i])
                current_as.append(a_sorted[i])
            else:
                # 合并当前簇：幅度求和，时延取幅度加权平均
                w = np.abs(current_as) / np.sum(np.abs(current_as))
                clusters_tau.append(np.sum(w * np.array(current_taus)))
                clusters_a.append(np.sum(current_as))
                # 开始新簇
                current_taus = [tau_sorted[i]]
                current_as = [a_sorted[i]]
        
        # 最后一个簇
        if len(current_taus) > 0:
            w = np.abs(current_as) / np.sum(np.abs(current_as))
            clusters_tau.append(np.sum(w * np.array(current_taus)))
            clusters_a.append(np.sum(current_as))
        
        return np.array(clusters_tau), np.array(clusters_a)
    
    def run_on_dataset(self, generator, num_signals, seed_base=1000, verbose=False):
        """
        在生成的数据集上批量运行
        返回: results 列表
        """
        results = []
        for i in tqdm(range(num_signals), desc="FRINET optimization"):
            sample = generator.generate_one(seed=seed_base + i)
            
            res = self.optimize_one(sample['y_noisy'], verbose=False)
            res['tau_true'] = sample['tau_true']
            res['a_true'] = sample['a_true']
            res['K_true'] = len(sample['tau_true'])
            res['snr_db'] = generator.snr_db
            res['y_noisy'] = sample['y_noisy']
            res['y_clean'] = sample['y_clean']
            res['t_samples'] = sample['t_samples']
            results.append(res)
        
        return results
# sensitivity_analysis.py灵敏度分析 — 对初始化、对 λ 的敏感性
import numpy as np
import matplotlib.pyplot as plt

def test_initialization_sensitivity(K=3, SNR=15, n_runs=20):
    """同一条信号，不同随机初始化，看结果方差"""
    from fri_net import ContinuousAtomNet
    from fri_generator import FRISignalGenerator
    from batch_runner import FRINETBatchRunner
    import torch
    
    gen = FRISignalGenerator(K=K, N=21, T_end=1.0, snr_db=SNR,
                             pulse_type='gaussian', pulse_params={'sigma': 0.08})
    sample = gen.generate_one(seed=999)
    
    all_tau_preds = []
    all_loss_curves = []
    
    for run in range(n_runs):
        y_tensor = torch.tensor(sample['y_noisy'], dtype=torch.float32)
        # 不做 seed 固定，每次初始化不同
        model = ContinuousAtomNet(N=21, M=100, T_end=1.0,
                                  pulse_fn=lambda t: torch.exp(-t**2/(2*0.08**2)))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        losses = []
        for step in range(800):
            optimizer.zero_grad()
            y_hat, tau, a = model.forward()
            loss = ((y_hat - y_tensor)**2).mean() + 0.01 * torch.abs(a).sum()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        with torch.no_grad():
            _, tau, a = model.forward()
            mask = torch.abs(a) > 0.05
            all_tau_preds.append(tau[mask].cpu().numpy())
            all_loss_curves.append(losses)
    
    # 统计
    tau_stds = []
    for i in range(K):
        pos_i = [pred[i] for pred in all_tau_preds if len(pred) >= i+1]
        if pos_i:
            tau_stds.append(np.std(pos_i))
    
    print(f"初始化灵敏度：{n_runs}次运行，时延标准差: {np.mean(tau_stds):.5f}")
    
    # 画所有损失曲线
    plt.figure(figsize=(8,4))
    for lc in all_loss_curves:
        plt.plot(lc, alpha=0.3)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(f'Loss curves from {n_runs} random initializations')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.savefig('init_sensitivity.png', dpi=150)
    plt.show()
    
    return tau_stds, all_loss_curves

test_initialization_sensitivity()
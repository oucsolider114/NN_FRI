import numpy as np
from batch_runner import FRINETBatchRunner
from fri_generator import FRISignalGenerator
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def run_grid_search(M_values, lam_values, lr_values=None,
                    K=3, SNR=15, N_signals=30, steps=600):
    """
    对 M 和 λ 做网格搜索
    返回: results_3d[M_idx, lam_idx] 的字典
    """
    if lr_values is None:
        lr_values = [0.01]
    
    gen = FRISignalGenerator(K=K, N=21, T_end=1.0, snr_db=SNR,
                             pulse_type='gaussian', pulse_params={'sigma': 0.08})
    
    results = np.zeros((len(M_values), len(lam_values), len(lr_values)))
    
    for i, M in enumerate(M_values):
        for j, lam in enumerate(lam_values):
            for k, lr in enumerate(lr_values):
                print(f"M={M}, λ={lam}, lr={lr} ...", end=" ")
                runner = FRINETBatchRunner(
                    N=21, M=M, T_end=1.0,
                    pulse_type='gaussian', pulse_params={'sigma': 0.08},
                    lam=lam, lr=lr, steps=steps
                )
                res = runner.run_on_dataset(gen, N_signals, seed_base=1000+i*100+j*10+k, verbose=False)
                
                # 计算平均时延 RMSE（只评估找到 K 个脉冲的信号）
                errors = []
                for r in res:
                    if len(r['tau_pred']) == K:
                        from scipy.optimize import linear_sum_assignment
                        D = np.abs(r['tau_pred'].reshape(-1,1) - r['tau_true'].reshape(1,-1))
                        ri, ci = linear_sum_assignment(D)
                        errors.append(np.sqrt(np.mean((r['tau_pred'][ri] - r['tau_true'][ci])**2)))
                
                avg_error = np.mean(errors) if len(errors) > 0 else np.nan
                results[i, j, k] = avg_error
                print(f"RMSE={avg_error:.5f}" if not np.isnan(avg_error) else "✗ 无合格结果")
    
    return results

# 运行搜索
M_vals = [20, 50, 100, 200, 500]
lam_vals = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1]
lr_vals = [0.01]

print("开始网格搜索...")
results = run_grid_search(M_vals, lam_vals, lr_vals, K=3, SNR=15, N_signals=30, steps=600)

# 可视化热力图
fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(results[:, :, 0].T, origin='lower', aspect='auto',
               extent=[M_vals[0], M_vals[-1], lam_vals[0], lam_vals[-1]],
               norm=LogNorm())
ax.set_xlabel('Network width M')
ax.set_ylabel('Regularization λ')
ax.set_title(f'Tau RMSE (K=3, SNR=15dB)')
plt.colorbar(im, label='RMSE')
# 标出最优
best_idx = np.unravel_index(np.nanargmin(results[:, :, 0]), results[:, :, 0].shape)
ax.plot(M_vals[best_idx[0]], lam_vals[best_idx[1]], 'r*', markersize=20)
ax.text(M_vals[best_idx[0]], lam_vals[best_idx[1]], 
        f'  Best: M={M_vals[best_idx[0]]}, λ={lam_vals[best_idx[1]]}', 
        color='red', fontweight='bold')
plt.tight_layout()
plt.savefig('grid_search_heatmap.png', dpi=150)
plt.show()

print(f"\n最优参数: M={M_vals[best_idx[0]]}, λ={lam_vals[best_idx[1]]}")
print(f"最优 RMSE: {results[best_idx[0], best_idx[1], 0]:.5f}")
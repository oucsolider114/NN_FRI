# robustness_tests.py泛化测试矩阵
import numpy as np
from batch_runner import FRINETBatchRunner
from fri_generator import FRISignalGenerator, PulseLibrary
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ─── 测试矩阵设计 ───
tests = [
    # (测试名称, 脉冲类型, 脉冲参数, K, SNR, 备注)
    ('基准高斯', 'gaussian', {'sigma': 0.08}, 3, 20, '训练同分布'),
    ('宽高斯', 'gaussian', {'sigma': 0.15}, 3, 20, '脉冲变宽'),
    ('窄高斯', 'gaussian', {'sigma': 0.04}, 3, 20, '脉冲变窄'),
    ('指数衰减', 'exponential', {'tau_decay': 0.06}, 3, 20, '换脉冲类型'),
    ('K=2', 'gaussian', {'sigma': 0.08}, 2, 20, '脉冲数不同'),
    ('K=5', 'gaussian', {'sigma': 0.08}, 5, 20, '更多脉冲'),
    ('K=2低SNR', 'gaussian', {'sigma': 0.08}, 2, 5, '极端噪声+K少'),
    ('K=5低SNR', 'gaussian', {'sigma': 0.08}, 5, 5, '极端噪声+K多'),
    ('密集脉冲', 'gaussian', {'sigma': 0.06}, 3, 15, '脉冲间隔 < 0.05'),
]

results_summary = []

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

for idx, (name, ptype, pparams, K, snr, note) in enumerate(tests):
    print(f"测试: {name} ...")
    
    gen = FRISignalGenerator(K=K, N=21, T_end=1.0, snr_db=snr,
                             pulse_type=ptype, pulse_params=pparams)
    
    # 密集脉冲特殊处理
    if '密集' in name:
        # 手动控制最小间隔
        pass
    
    runner = FRINETBatchRunner(
        N=21, M=100, T_end=1.0,
        pulse_type=ptype, pulse_params=pparams,
        lam=0.01, lr=0.01, steps=800
    )
    res = runner.run_on_dataset(gen, 30, seed_base=idx*1000, verbose=False)
    
    # 计算成功率和 RMSE
    from scipy.optimize import linear_sum_assignment
    errors = []
    succ = 0
    for r in res:
        if len(r['tau_pred']) == K:
            D = np.abs(r['tau_pred'].reshape(-1,1) - r['tau_true'].reshape(1,-1))
            ri, ci = linear_sum_assignment(D)
            err = np.sqrt(np.mean((r['tau_pred'][ri] - r['tau_true'][ci])**2))
            if err < 0.05:
                succ += 1
            errors.append(err)
    
    avg_err = np.mean(errors) if errors else np.nan
    succ_rate = succ / 30 * 100
    results_summary.append({'test': name, 'rmse': avg_err, 'success': succ_rate})
    
    # 画一个代表性样本
    if idx < 9:
        sample = gen.generate_one(seed=idx*1000)
        best_res = min(res, key=lambda r: r['losses'][-1]) if res else None
        if best_res:
            ax = axes[idx]
            t_s = best_res['t_samples']
            ax.stem(t_s, best_res['y_noisy'], linefmt='r-', markerfmt='ro', basefmt='k-')
            ax.plot(t_s, best_res['y_clean'], 'b--')
            for tk in best_res['tau_true']:
                ax.axvline(tk, color='green', ls=':', alpha=0.7)
            for tk in best_res['tau_pred']:
                ax.axvline(tk, color='orange', ls='-', alpha=0.6)
            ax.set_title(f'{name}\nErr={avg_err:.4f}, Succ={succ_rate:.0f}%')

plt.tight_layout()
plt.savefig('robustness_all.png', dpi=150)
plt.show()

# 打印总结表
print(f"\n{'='*60}")
print(f"{'测试名称':<15} {'RMSE':>10} {'成功率':>10}")
print(f"{'-'*35}")
for r in results_summary:
    print(f"{r['test']:<15} {r['rmse']:>10.5f} {r['success']:>9.1f}%")
print(f"{'='*60}")
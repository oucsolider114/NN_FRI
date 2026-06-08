import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from fri_generator import FRISignalGenerator
from batch_runner import FRINETBatchRunner
from traditional_methods import matrix_pencil

# ─── 实验设置 ───
N_SIGNALS = 50  # 先用少量快速验证
K_TRUE = 3
SNR_DB = 25

# 生成固定参数的测试信号
gen = FRISignalGenerator(K=K_TRUE, N=21, T_end=1.0, snr_db=SNR_DB,
                         pulse_type='gaussian', pulse_params={'sigma': 0.08})

# 批量运行 FRINET
runner = FRINETBatchRunner(N=21, M=100, T_end=1.0, 
                           pulse_type='gaussian', pulse_params={'sigma': 0.08},
                           lam=0.01, lr=0.01, steps=800)
frinet_results = runner.run_on_dataset(gen, N_SIGNALS, seed_base=500, verbose=False)

# 运行矩阵束法
mp_results = []
for i in range(N_SIGNALS):
    sample = gen.generate_one(seed=500 + i)
    tau_mp, a_mp = matrix_pencil(sample['y_noisy'], K_TRUE)
    mp_results.append({
        'tau_pred': tau_mp,
        'a_pred': a_mp,
        'tau_true': sample['tau_true'],
        'a_true': sample['a_true'],
    })

# ─── 评估指标计算 ───
def compute_rmse(pred, true, normalize=True):
    """计算 RMSE，先做最优匹配"""
    from scipy.optimize import linear_sum_assignment
    if len(pred) == 0 or len(true) == 0:
        return np.nan
    # 距离矩阵
    D = np.abs(pred.reshape(-1, 1) - true.reshape(1, -1))
    row_ind, col_ind = linear_sum_assignment(D)
    errors = pred[row_ind] - true[col_ind]
    return np.sqrt(np.mean(errors**2))

frinet_tau_rmse = []
mp_tau_rmse = []
frinet_success = []  # 找到刚好 K 个脉冲就算成功

for fr, mp in zip(frinet_results, mp_results):
    # 匹配计算
    tau_p = fr['tau_pred']
    tau_t = fr['tau_true']
    if len(tau_p) >= len(tau_t):
        D = np.abs(tau_p.reshape(-1,1) - tau_t.reshape(1,-1))
        ri, ci = linear_sum_assignment(D)
        err = np.sqrt(np.mean((tau_p[ri] - tau_t[ci])**2))
    else:
        err = np.nan
    frinet_tau_rmse.append(err)
    frinet_success.append(len(tau_p) == K_TRUE)
    
    tau_p = mp['tau_pred']
    if len(tau_p) >= len(tau_t):
        D = np.abs(tau_p.reshape(-1,1) - tau_t.reshape(1,-1))
        ri, ci = linear_sum_assignment(D)
        err = np.sqrt(np.mean((tau_p[ri] - tau_t[ci])**2))
    else:
        err = np.nan
    mp_tau_rmse.append(err)

# 打印
print(f"\n{'='*50}")
print(f"对比实验: K={K_TRUE}, SNR={SNR_DB}dB, {N_SIGNALS}个信号")
print(f"{'='*50}")
print(f"FRINET 时延 RMSE: {np.nanmean(frinet_tau_rmse):.5f}")
print(f"矩阵束法 时延 RMSE: {np.nanmean(mp_tau_rmse):.5f}")
print(f"FRINET 成功率 (找到恰好{K_TRUE}个脉冲): {np.mean(frinet_success)*100:.1f}%")

# 画箱线图
fig, ax = plt.subplots(figsize=(8, 4))
data = [frinet_tau_rmse, mp_tau_rmse]
ax.boxplot(data, labels=['FRINET', 'Matrix Pencil'])
ax.set_ylabel('Tau RMSE')
ax.set_title(f'Performance comparison (SNR={SNR_DB}dB)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('baseline_compare.png', dpi=150)
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from batch_runner import FRINETBatchRunner
from fri_generator import FRISignalGenerator

M_VALUES = [10, 20, 30, 50, 75, 100, 150, 200, 300, 500]
K_TRUE = 3
SNR_VALUES = [5, 10, 15, 20, 25]
N_SIGNALS = 40  # 每个点用40个信号统计

gen = FRISignalGenerator(K=K_TRUE, N=21, T_end=1.0, snr_db=10,
                         pulse_type='gaussian', pulse_params={'sigma': 0.08})

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for snr in SNR_VALUES:
    success_rates = []
    tau_rmse_list = []
    
    gen.snr_db = snr  # 修改 SNR
    
    for M in M_VALUES:
        runner = FRINETBatchRunner(
            N=21, M=M, T_end=1.0,
            pulse_type='gaussian', pulse_params={'sigma': 0.08},
            lam=0.01, lr=0.01, steps=800
        )
        res = runner.run_on_dataset(gen, N_SIGNALS, seed_base=int(snr*100), verbose=False)
        
        # 成功率：找到恰好 K 个脉冲，且时延误差 < 0.05
        successes = 0
        errors = []
        for r in res:
            if len(r['tau_pred']) == K_TRUE:
                from scipy.optimize import linear_sum_assignment
                D = np.abs(r['tau_pred'].reshape(-1,1) - r['tau_true'].reshape(1,-1))
                ri, ci = linear_sum_assignment(D)
                err = np.sqrt(np.mean((r['tau_pred'][ri] - r['tau_true'][ci])**2))
                if err < 0.05:
                    successes += 1
                errors.append(err)
        
        success_rates.append(successes / N_SIGNALS * 100)
        tau_rmse_list.append(np.mean(errors) if errors else np.nan)
    
    axes[0].plot(M_VALUES, success_rates, 'o-', label=f'SNR={snr}dB')
    axes[1].plot(M_VALUES, tau_rmse_list, 'o-', label=f'SNR={snr}dB')

axes[0].set_xlabel('Network width M')
axes[0].set_ylabel('Success rate (%)')
axes[0].set_title(f'Success rate vs M (K={K_TRUE})')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_xscale('log')

axes[1].set_xlabel('Network width M')
axes[1].set_ylabel('Tau RMSE')
axes[1].set_title(f'RMSE vs M (K={K_TRUE})')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_xscale('log')

plt.tight_layout()
plt.savefig('overparam_effect.png', dpi=150)
plt.show()
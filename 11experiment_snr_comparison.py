# experiment_snr_comparison.py
import numpy as np
import matplotlib.pyplot as plt
from batch_runner import FRINETBatchRunner
from fri_generator import FRISignalGenerator
from traditional_methods import matrix_pencil
from scipy.optimize import linear_sum_assignment

SNR_RANGE = np.arange(-5, 31, 2)  # -5 到 30 dB，步长 2dB
K_TRUE = 3
N_SIGNALS_PER_SNR = 50
M_OPTIMAL = 100  # 从第10周搜索得到的最优值

gen = FRISignalGenerator(K=K_TRUE, N=21, T_end=1.0, snr_db=0,
                         pulse_type='gaussian', pulse_params={'sigma': 0.08})

frinet_rmse = []
frinet_std = []
mp_rmse = []
mp_std = []
frinet_success = []
mp_success = []
all_frinet_results = []
snr_list = []

for snr in SNR_RANGE:
    gen.snr_db = snr
    
    # ── FRINET ──
    runner = FRINETBatchRunner(
        N=21, M=M_OPTIMAL, T_end=1.0,
        pulse_type='gaussian', pulse_params={'sigma': 0.08},
        lam=0.01, lr=0.01, steps=800
    )
    fr_res = runner.run_on_dataset(gen, N_SIGNALS_PER_SNR, seed_base=int(snr*100) + 10000, verbose=False)
    all_frinet_results.append(fr_res)
    snr_list.append(snr)

    fr_errors = []
    fr_succ = 0
    for r in fr_res:
        if len(r['tau_pred']) == K_TRUE:
            D = np.abs(r['tau_pred'].reshape(-1,1) - r['tau_true'].reshape(1,-1))
            ri, ci = linear_sum_assignment(D)
            err = np.sqrt(np.mean((r['tau_pred'][ri] - r['tau_true'][ci])**2))
            if err < 0.05:
                fr_succ += 1
            fr_errors.append(err)
    frinet_rmse.append(np.mean(fr_errors) if fr_errors else np.nan)
    frinet_std.append(np.std(fr_errors) if fr_errors else np.nan)
    frinet_success.append(fr_succ / N_SIGNALS_PER_SNR * 100)
    
    # ── 矩阵束法 ──
    mp_errors = []
    mp_succ = 0
    for i in range(N_SIGNALS_PER_SNR):
        sample = gen.generate_one(seed=int(snr*1000) + i + 10000)
        try:
            tau_mp, a_mp = matrix_pencil(sample['y_noisy'], K_TRUE)
            if len(tau_mp) >= K_TRUE:
                D = np.abs(tau_mp[:K_TRUE].reshape(-1,1) - sample['tau_true'].reshape(1,-1))
                ri, ci = linear_sum_assignment(D)
                err = np.sqrt(np.mean((tau_mp[ri] - sample['tau_true'][ci])**2))
                if err < 0.05:
                    mp_succ += 1
                mp_errors.append(err)
        except:
            pass
    
    mp_rmse.append(np.mean(mp_errors) if mp_errors else np.nan)
    mp_std.append(np.std(mp_errors) if mp_errors else np.nan)
    mp_success.append(mp_succ / N_SIGNALS_PER_SNR * 100)
    
    print(f"SNR={snr:3d}dB | FRINET RMSE={frinet_rmse[-1]:.5f} succ={frinet_success[-1]:.0f}% | "
          f"MP RMSE={mp_rmse[-1]:.5f} succ={mp_success[-1]:.0f}%")

# ─── 画大图 ───
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

ax = axes[0]
ax.errorbar(SNR_RANGE, frinet_rmse, yerr=frinet_std, fmt='o-', capsize=3, label='FRINET (M=100)')
ax.errorbar(SNR_RANGE, mp_rmse, yerr=mp_std, fmt='s--', capsize=3, label='Matrix Pencil')
ax.set_xlabel('SNR (dB)')
ax.set_ylabel('Tau RMSE')
ax.set_title('Parameter estimation error vs SNR')
ax.set_yscale('log')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(SNR_RANGE, frinet_success, 'o-', label='FRINET (M=100)')
ax.plot(SNR_RANGE, mp_success, 's--', label='Matrix Pencil')
ax.set_xlabel('SNR (dB)')
ax.set_ylabel('Success rate (%)')
ax.set_title('Recovery success rate vs SNR')
ax.axhline(y=100, color='gray', ls=':', alpha=0.5)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('snr_comparison_full.png', dpi=150)
plt.show()

# ─── 失败案例分析 (SNR=0dB & 10dB) ───
from failure_analysis import analyze_failure_cases

for target_snr in [0, 10]:
    if target_snr in snr_list:
        idx = snr_list.index(target_snr)
        snr_results = all_frinet_results[idx]
        print(f"\n===== 失败分析: SNR={target_snr}dB =====")
        failures = analyze_failure_cases(snr_results, snr_label=f"{target_snr}dB")
    else:
        print(f"SNR={target_snr}dB 不在扫描范围内，跳过")
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

def analyze_failure_cases(results, n_examples=3, snr_label=''):
    """
    找出 FRINET 失败的案例并分析原因
    """
    failures = []
    for r in results:
        if len(r['tau_pred']) != r['K_true']:
            failures.append(r)
        elif len(r['tau_pred']) == r['K_true']:
            # 检查时延误差是否过大
            D = np.abs(r['tau_pred'].reshape(-1,1) - r['tau_true'].reshape(1,-1))
            ri, ci = linear_sum_assignment(D)
            err = np.sqrt(np.mean((r['tau_pred'][ri] - r['tau_true'][ci])**2))
            if err > 0.1:
                failures.append(r)

    fail_rate = len(failures) / len(results) * 100
    print(f"失败案例数: {len(failures)} / {len(results)} ({fail_rate:.1f}%)")

    if len(failures) == 0:
        print("没有失败案例，跳过可视化")
        return failures

    # 画前几个失败案例
    n_plot = min(n_examples, len(failures))
    fig, axes = plt.subplots(1, n_plot, figsize=(5 * n_plot, 4))
    if n_plot == 1:
        axes = [axes]

    for i, f in enumerate(failures[:n_examples]):
        ax = axes[i]
        t_s = f['t_samples']
        ax.stem(t_s, f['y_noisy'], linefmt='r-', markerfmt='ro', basefmt='k-', label='Noisy')
        ax.plot(t_s, f['y_clean'], 'b--', label='Clean')

        # 真实脉冲位置
        for tk in f['tau_true']:
            ax.axvline(tk, color='green', ls=':', alpha=0.8, label='True τ' if i == 0 else '')
        # 预测脉冲位置
        for tk in f['tau_pred']:
            ax.axvline(tk, color='orange', ls='-', alpha=0.6, label='Pred τ' if i == 0 else '')
        ax.set_title(f'Failure case: K_true={f["K_true"]}, K_pred={len(f["tau_pred"])}')
        ax.legend()

    plt.tight_layout()
    fname = f'failure_cases_{snr_label}.png' if snr_label else 'failure_cases.png'
    plt.savefig(fname, dpi=150)
    plt.show()

    return failures

if __name__ == '__main__':
    from batch_runner import FRINETBatchRunner
    from fri_generator import FRISignalGenerator

    for snr in [0, 10]:
        print(f"\n===== 失败分析: SNR={snr}dB =====")
        gen = FRISignalGenerator(K=3, N=21, T_end=1.0, snr_db=snr,
                                 pulse_type='gaussian', pulse_params={'sigma': 0.08})
        runner = FRINETBatchRunner(
            N=21, M=100, T_end=1.0,
            pulse_type='gaussian', pulse_params={'sigma': 0.08},
            lam=0.01, lr=0.01, steps=800
        )
        results = runner.run_on_dataset(gen, 50, seed_base=10000 + snr * 100, verbose=True)
        analyze_failure_cases(results, snr_label=f"{snr}dB")
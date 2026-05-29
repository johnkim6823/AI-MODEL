"""
[메타러닝] MAML: 1차(FOMAML) vs 2차(정식) 비교 ⭐
==================================================

▶ MAML의 메타그래디언트 (이너 1스텝)
      g_meta = (I − inner_lr·H) · g_query        (H = 서포트 손실의 헤시안)
  - 2차 MAML: 위 식을 그대로 사용 → 이너 갱신을 '거슬러 미분'(헤시안 필요).
  - 1차 MAML(FOMAML): (I − inner_lr·H) 항을 생략하고 g_query 만 사용.
    → 헤시안 계산이 없어 훨씬 단순하고 빠르다.

▶ 잘 알려진 사실 (Finn et al. 2017)
  - FOMAML은 2차 MAML과 '거의 같은' 성능을 낸다.
  - 그래서 실무에선 보통 1차 근사(FOMAML)나 Reptile을 쓴다.

이 스크립트는 둘을 같은 조건으로 학습해 (학습 시간, 적응 성능)을 비교한다.
  * 2차 항의 헤시안-벡터곱은 유한차분으로 근사한다(meta_core.py 참고).
"""

import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import meta_core as mc

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def main():
    print("=" * 55)
    print(" MAML 1차(FOMAML) vs 2차 비교 (학습 중...)")
    print("=" * 55)

    settings = dict(n_iters=6000, meta_batch=10, k_shot=10,
                    inner_lr=0.01, meta_lr=0.01, seed=0)

    t0 = time.time()
    theta_fo, hist_fo = mc.train_maml_variant(second_order=False, **settings)
    t_fo = time.time() - t0

    t0 = time.time()
    theta_so, hist_so = mc.train_maml_variant(second_order=True, **settings)
    t_so = time.time() - t0

    # 적응 곡선 평가
    test_tasks = [mc.SineTask(np.random.default_rng(2000 + i)) for i in range(80)]
    c_fo = mc.adaptation_curve(theta_fo, test_tasks, 10, 0.01, 10, seed=7)
    c_so = mc.adaptation_curve(theta_so, test_tasks, 10, 0.01, 10, seed=7)

    print(f"  {'방법':<14}{'학습시간':>10}{'적응후 MSE':>12}")
    print("  " + "-" * 38)
    print(f"  {'1차 (FOMAML)':<14}{t_fo:>9.1f}s{c_fo[-1]:>12.3f}")
    print(f"  {'2차 (정식)':<14}{t_so:>9.1f}s{c_so[-1]:>12.3f}")
    print(f"  → 성능은 비슷하지만 2차가 약 {t_so/max(t_fo,1e-9):.1f}배 느리다.")

    # 시각화 --------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sm_fo = mc.moving_average(hist_fo, 100)
    sm_so = mc.moving_average(hist_so, 100)
    axes[0].plot(range(len(sm_fo)), sm_fo, color="#4C72B0", linewidth=2,
                 label="First-order (FOMAML)")
    axes[0].plot(range(len(sm_so)), sm_so, color="#C44E52", linewidth=2,
                 label="Second-order (full)")
    axes[0].set_title("Meta-training loss (almost the same)")
    axes[0].set_xlabel("Meta-iteration")
    axes[0].set_ylabel("Loss (smoothed)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(range(11), c_fo, "o-", color="#4C72B0", linewidth=2,
                 label=f"First-order ({t_fo:.0f}s)")
    axes[1].plot(range(11), c_so, "o-", color="#C44E52", linewidth=2,
                 label=f"Second-order ({t_so:.0f}s)")
    axes[1].set_title("Adaptation curve (comparable performance)")
    axes[1].set_xlabel("Adaptation steps")
    axes[1].set_ylabel("MSE on new task")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("MAML: First-order vs Second-order", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = os.path.join(OUTPUT_DIR, "23_maml_first_vs_second_order.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 그래프 저장됨 -> {save_path}")
    print("\n  💡 해석:")
    print("     - 1차(FOMAML)와 2차 MAML의 성능은 비슷하다.")
    print("     - 2차는 헤시안 계산 때문에 더 느리다 → 실무에선 보통 1차를 쓴다.")


if __name__ == "__main__":
    main()

"""
[메타러닝] Meta-SGD — 학습률까지 학습하기 ⭐
=============================================

Meta-SGD (Li et al. 2017)

▶ MAML 과의 차이
  - MAML: '좋은 초기값 θ' 만 학습한다 (이너 학습률은 사람이 고정).
  - Meta-SGD: '초기값 θ' + '파라미터마다 다른 학습률 α' 를 함께 학습한다.
    이너 갱신:  φ = θ − α ⊙ ∇L_support(θ)
    → 어떤 파라미터는 크게, 어떤 건 작게 적응하도록 '적응 방법' 자체를 배운다.

▶ 결과
  (왼쪽) 학습된 학습률 α의 분포 — 0 근처부터 큰 값까지 다양하게 학습됨
  (오른쪽) 적응 곡선: Meta-SGD vs 고정 학습률 MAML(FOMAML)
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import meta_core as mc

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def main():
    print("=" * 55)
    print(" Meta-SGD: 초기값 + 파라미터별 학습률 함께 학습 (학습 중...)")
    print("=" * 55)

    # Meta-SGD 학습 (초기값 theta + 학습률 alpha)
    theta_msgd, alpha, _ = mc.train_meta_sgd(
        n_iters=6000, meta_batch=10, k_shot=10,
        init_inner_lr=0.01, meta_lr=0.01, alpha_lr=0.02, seed=0)

    # 비교용: 고정 학습률 MAML(FOMAML)
    theta_fomaml, _ = mc.train_maml(
        n_iters=6000, meta_batch=10, k_shot=10,
        inner_lr=0.01, meta_lr=0.01, inner_steps=1, seed=0)

    # 적응 곡선 평가
    test_tasks = [mc.SineTask(np.random.default_rng(2000 + i)) for i in range(80)]
    c_msgd = mc.adaptation_curve(theta_msgd, test_tasks, k_shot=10,
                                 inner_lr=alpha, max_steps=10, seed=7)
    c_fomaml = mc.adaptation_curve(theta_fomaml, test_tasks, k_shot=10,
                                   inner_lr=0.01, max_steps=10, seed=7)

    # 학습된 학습률 모으기
    all_alpha = np.concatenate([a[0].ravel() for a in alpha] +
                               [a[1].ravel() for a in alpha])

    print(f"  학습된 학습률 α: 평균 {all_alpha.mean():.4f}, "
          f"범위 [{all_alpha.min():.4f}, {all_alpha.max():.4f}]")
    print(f"  적응 후(10스텝) MSE — Meta-SGD {c_msgd[-1]:.3f}  vs  FOMAML {c_fomaml[-1]:.3f}")

    # 시각화 --------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].hist(all_alpha, bins=40, color="#8172B3", alpha=0.8)
    axes[0].axvline(0.01, color="#C44E52", linestyle="--",
                    label="initial lr = 0.01 (MAML uses fixed)")
    axes[0].set_title("Learned per-parameter learning rates (Meta-SGD)")
    axes[0].set_xlabel("learning rate value")
    axes[0].set_ylabel("count")
    axes[0].legend()

    axes[1].plot(range(11), c_fomaml, "o-", color="#4C72B0", linewidth=2,
                 label="MAML (fixed lr)")
    axes[1].plot(range(11), c_msgd, "o-", color="#8172B3", linewidth=2,
                 label="Meta-SGD (learned lr)")
    axes[1].set_title("Adaptation curve: Meta-SGD vs fixed-lr MAML")
    axes[1].set_xlabel("Adaptation steps")
    axes[1].set_ylabel("MSE on new task (lower = better)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Meta-SGD: learning the learning rate", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = os.path.join(OUTPUT_DIR, "22_meta_sgd.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 그래프 저장됨 -> {save_path}")
    print("\n  💡 해석:")
    print("     - Meta-SGD는 파라미터마다 다른 학습률을 '스스로' 학습한다(분포가 넓음).")
    print("     - 적응 방법까지 배우므로 고정 학습률 MAML과 비슷하거나 더 잘 적응한다.")


if __name__ == "__main__":
    main()

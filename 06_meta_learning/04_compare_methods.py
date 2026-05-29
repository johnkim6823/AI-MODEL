"""
[메타러닝 - 실험] 방법 비교 + 데이터 양(K-shot)에 따른 성능 ⭐
================================================================

메타러닝의 효과를 한눈에 비교한다.

▶ 비교 대상 (모두 같은 사인파 문제, 같은 적응 방식)
  - Random  : 학습 안 한 무작위 초기값 (하한선)
  - Joint   : 메타러닝 없이 여러 태스크를 그냥 합쳐 학습 (평균 곡선만 배움)
  - Reptile : 메타러닝
  - MAML    : 메타러닝

▶ 두 가지 그래프
  (왼쪽) 적응 곡선: 적응(경사하강) 단계를 늘릴수록 새 태스크 오차(MSE)가
        얼마나 빨리 줄어드는가. 메타학습된 초기값일수록 낮게 시작해 빨리 준다.
  (오른쪽) K-shot 효과: 주어지는 예시 개수(K)가 많을수록 적응 후 오차가 준다.
        (지도학습에서 '데이터가 많을수록 좋다'와 같은 직관)
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import meta_core as mc

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def main():
    print("=" * 60)
    print(" 메타러닝 방법 비교 실험 (학습 중...)")
    print("=" * 60)

    # 1) 각 방법으로 초기값 준비 -----------------------------------------
    theta_rand = mc.init_params(np.random.default_rng(123), n_hidden=40)
    theta_joint, _ = mc.train_joint(n_iters=10000, k_shot=10, lr=0.01, seed=0)
    theta_reptile, _ = mc.train_reptile(n_iters=12000, k_shot=10, inner_lr=0.02,
                                        meta_lr=0.1, inner_steps=5, seed=0)
    theta_maml, _ = mc.train_maml(n_iters=8000, meta_batch=10, k_shot=10,
                                  inner_lr=0.01, meta_lr=0.01, inner_steps=1, seed=0)

    methods = {
        "Random": (theta_rand, "#999999"),
        "Joint": (theta_joint, "#CCB974"),
        "Reptile": (theta_reptile, "#55A868"),
        "MAML": (theta_maml, "#C44E52"),
    }

    # 공통 테스트 태스크
    test_tasks = [mc.SineTask(np.random.default_rng(2000 + i)) for i in range(80)]
    inner_lr, max_steps = 0.01, 10

    # 2) (왼쪽) 적응 곡선: MSE vs 적응 단계 -------------------------------
    print(f"\n  [적응 곡선] K=10, 적응 {max_steps}스텝까지")
    print(f"  {'방법':>8} | {'적응전':>7} | {'10스텝후':>8}")
    print("  " + "-" * 30)
    curves = {}
    for name, (theta, _) in methods.items():
        c = mc.adaptation_curve(theta, test_tasks, k_shot=10,
                                inner_lr=inner_lr, max_steps=max_steps, seed=7)
        curves[name] = c
        print(f"  {name:>8} | {c[0]:>7.3f} | {c[-1]:>8.3f}")

    # 3) (오른쪽) K-shot 효과 (가장 강한 메타학습 초기값으로) -------------
    k_values = [2, 5, 10, 20]
    print(f"\n  [K-shot 효과] Reptile 초기값, {max_steps}스텝 적응 후 MSE")
    k_final = []
    for k in k_values:
        c = mc.adaptation_curve(theta_reptile, test_tasks, k_shot=k,
                                inner_lr=inner_lr, max_steps=max_steps, seed=7)
        k_final.append(c[-1])
        print(f"     K={k:<3} -> 적응 후 MSE {c[-1]:.3f}")

    # 4) 시각화 -----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for name, (_, color) in methods.items():
        axes[0].plot(range(max_steps + 1), curves[name], "o-", color=color,
                     linewidth=2, markersize=4, label=name)
    axes[0].set_title("(1) Adaptation Curve: MSE vs adaptation steps")
    axes[0].set_xlabel("Adaptation steps (gradient steps on K points)")
    axes[0].set_ylabel("MSE on new task (lower = better)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    bars = axes[1].bar([str(k) for k in k_values], k_final, color="#55A868")
    for bar, val in zip(bars, k_final):
        axes[1].text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    axes[1].set_title("(2) Effect of K (number of examples) — Reptile")
    axes[1].set_xlabel("K (shots / examples per task)")
    axes[1].set_ylabel("MSE after adaptation")
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.suptitle("Meta-Learning Comparison: faster adaptation from a learned init",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = os.path.join(OUTPUT_DIR, "18_meta_learning_comparison.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 그래프 저장됨 -> {save_path}")
    print("\n  💡 해석:")
    print("     - 메타학습(Reptile/MAML) 초기값은 낮은 오차에서 시작해 빨리 적응한다.")
    print("     - Random/Joint는 적은 데이터로 새 태스크에 잘 적응하지 못한다.")
    print("     - K(예시 수)가 많을수록 적응 후 오차가 줄어든다.")


if __name__ == "__main__":
    main()

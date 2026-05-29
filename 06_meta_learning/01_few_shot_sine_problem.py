"""
[메타러닝 - 문제 소개] few-shot 사인파 회귀란?
================================================

메타러닝이 푸는 문제 상황을 먼저 눈으로 본다.

▶ 태스크 분포
  - 태스크마다 진폭/위상이 다른 사인파가 무한히 많다.
  - 메타러닝은 "이 사인파들 전반"을 경험하며 적응 능력을 기른다.

▶ few-shot(소량 학습)의 어려움
  - 새 사인파의 점 K개(예: 10개)만 보고 전체 곡선을 맞혀야 한다.
  - '무작위 초기값'에서 시작하면, 10개 점에 몇 번 갱신해 봐도
    엉뚱한 곡선이 된다(정보가 부족). → 그래서 '좋은 초기값'이 필요!
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import meta_core as mc

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def main():
    rng = np.random.default_rng(0)
    x_dense = np.linspace(-5, 5, 200).reshape(-1, 1)

    print("=" * 55)
    print(" few-shot 사인파 회귀 문제 소개")
    print("=" * 55)
    print("  - 태스크마다 다른 사인파. 점 K개만 보고 전체 곡선 맞히기.")
    print("  - 무작위 초기값으로는 소량 데이터에 잘 적응하지 못한다.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # (왼쪽) 여러 태스크(사인파) 예시 -----------------------------------
    for _ in range(6):
        task = mc.SineTask(rng)
        axes[0].plot(x_dense, task.true_curve(x_dense), linewidth=2, alpha=0.8)
    axes[0].set_title("Task distribution: many different sine waves")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].grid(True, alpha=0.3)

    # (오른쪽) 한 태스크의 K-shot + 무작위 초기값 적응 시도 ---------------
    task = mc.SineTask(np.random.default_rng(42))
    k = 10
    x_s, y_s = task.sample(k, np.random.default_rng(7))

    # 무작위 초기값으로 서포트 점에 적응 시도 (몇 스텝 학습)
    theta = mc.init_params(np.random.default_rng(123), n_hidden=40)
    adapted = mc.inner_adapt(theta, x_s, y_s, inner_lr=0.01, inner_steps=10)
    pred, _ = mc.forward(adapted, x_dense)

    axes[1].plot(x_dense, task.true_curve(x_dense), "g-", linewidth=2.5,
                 label="true sine (goal)")
    axes[1].scatter(x_s, y_s, color="black", zorder=5, s=50,
                    label=f"{k} given points (support)")
    axes[1].plot(x_dense, pred, "--", color="#C44E52", linewidth=2,
                 label="random-init fit (10 steps)")
    axes[1].set_title("Few-shot challenge: random init fits poorly")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    save_path = os.path.join(OUTPUT_DIR, "15_few_shot_sine_problem.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 그래프 저장됨 -> {save_path}")
    print("\n  💡 다음 단계: MAML(02), Reptile(03)로 '좋은 초기값'을 학습하면")
    print("     같은 10개 점만으로도 곡선을 훨씬 잘 맞춘다!")


if __name__ == "__main__":
    main()

"""
[메타러닝] Reptile — 더 단순한 메타러닝 ⭐
============================================

Reptile (Nichol et al. 2018, OpenAI)

▶ 핵심 아이디어 (MAML보다 훨씬 단순)
  1) 한 태스크에서 여러 번 경사하강으로 학습해 φ 를 얻는다.
  2) 초기값 θ 를 φ 쪽으로 조금 당긴다:   θ ← θ + ε·(φ − θ)
  반복하면 "여러 태스크의 좋은 해들에 두루 가까운" 초기값이 만들어진다.

▶ MAML 과의 차이
  - MAML: 적응 후 쿼리 성능의 기울기로 초기값을 갱신 (원래는 2차 미분 필요).
  - Reptile: 기울기를 거슬러 미분할 필요가 전혀 없음. 적응된 파라미터 쪽으로
             초기값을 이동시키기만 한다 → 구현이 매우 간단하고 빠르다.

▶ 결과 (MAML 예제와 동일한 구성으로 비교)
  (왼쪽) 메타 학습 곡선
  (오른쪽) 새 사인파에 점 10개로 적응한 결과
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import meta_core as mc

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def main():
    # 1) Reptile 메타 학습 ------------------------------------------------
    print("=" * 55)
    print(" Reptile 메타 학습 중... (사인파 태스크들)")
    print("=" * 55)
    inner_lr, k_shot, adapt_steps = 0.01, 10, 5
    theta, history = mc.train_reptile(
        n_iters=12000, k_shot=k_shot, inner_lr=0.02,
        meta_lr=0.1, inner_steps=5, seed=0,
    )

    # 2) 새 태스크에 few-shot 적응 (MAML 예제와 같은 태스크/세팅) ---------
    task = mc.SineTask(np.random.default_rng(42))
    x_s, y_s = task.sample(k_shot, np.random.default_rng(7))
    x_dense = np.linspace(-5, 5, 200).reshape(-1, 1)
    y_true = task.true_curve(x_dense)

    def mse_to_true(params):
        pred, _ = mc.forward(params, x_dense)
        return float(np.mean((pred - y_true) ** 2))

    pred_before, _ = mc.forward(theta, x_dense)
    adapted = mc.inner_adapt(theta, x_s, y_s, inner_lr, adapt_steps)
    pred_after, _ = mc.forward(adapted, x_dense)

    theta_rand = mc.init_params(np.random.default_rng(123), n_hidden=40)
    rand_adapted = mc.inner_adapt(theta_rand, x_s, y_s, inner_lr, adapt_steps)
    pred_rand, _ = mc.forward(rand_adapted, x_dense)

    print(f"  새 태스크 적응 (점 {k_shot}개, {adapt_steps}스텝):")
    print(f"   - 메타 초기값, 적응 전 MSE : {mse_to_true(theta):.3f}")
    print(f"   - 메타 초기값, 적응 후 MSE : {mse_to_true(adapted):.3f}  <- 확 줄어듦!")
    print(f"   - 무작위 초기값, 적응 후 MSE: {mse_to_true(rand_adapted):.3f}  (대조군)")

    # 3) 시각화 -----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sm = mc.moving_average(history, 200)
    axes[0].plot(range(len(sm)), sm, color="#55A868", linewidth=2)
    axes[0].set_title("Reptile meta-training loss")
    axes[0].set_xlabel("Meta-iteration")
    axes[0].set_ylabel("Loss (smoothed)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x_dense, y_true, "g-", linewidth=2.5, label="true sine")
    axes[1].scatter(x_s, y_s, color="black", s=50, zorder=5, label=f"{k_shot} points")
    axes[1].plot(x_dense, pred_before, ":", color="gray", linewidth=2,
                 label="meta-init (0 steps)")
    axes[1].plot(x_dense, pred_after, "-", color="#C44E52", linewidth=2.2,
                 label=f"Reptile adapted ({adapt_steps} steps)")
    axes[1].plot(x_dense, pred_rand, "--", color="#CCB974", linewidth=2,
                 label=f"random-init adapted ({adapt_steps} steps)")
    axes[1].set_title("Few-shot adaptation on a NEW sine task")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    save_path = os.path.join(OUTPUT_DIR, "17_reptile_sine.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 그래프 저장됨 -> {save_path}")
    print("\n  💡 해석: Reptile은 MAML보다 단순하지만, 마찬가지로 좋은 초기값을 배워")
    print("     적은 데이터로 새 사인파에 빠르게 적응한다.")


if __name__ == "__main__":
    main()

"""
[메타러닝] MAML — 빠르게 적응하는 초기값 배우기 ⭐
===================================================

MAML (Model-Agnostic Meta-Learning, Finn et al. 2017)

▶ 핵심 아이디어
  "어떤 새 태스크든, 몇 번의 경사하강만으로 잘 풀리는 '초기값 θ'를 찾자."
    - 이너 루프: 한 태스크의 서포트셋으로 θ를 임시 적응 → φ
    - 아우터 루프: 적응된 φ가 그 태스크의 쿼리셋에서 잘 맞도록 θ를 갱신
  즉, "적응한 뒤의 성능"이 좋아지도록 출발점 자체를 학습한다.

▶ 1차 근사(FOMAML) 사용
  진짜 MAML은 이너 갱신을 거슬러 2차 미분을 하지만, 여기서는 그 항을
  생략한 1차 근사를 쓴다(구현 단순·성능 비슷; meta_core.py 참고).

▶ 결과
  (왼쪽) 메타 학습 곡선 (쿼리 손실이 줄어드는 모습)
  (오른쪽) 학습한 적이 없는 '새 사인파'에, 점 10개만으로 적응한 결과
          - 적응 전(메타 초기값) vs 적응 후 vs 무작위 초기값 적응
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import meta_core as mc

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def main():
    # 1) MAML 메타 학습 ---------------------------------------------------
    print("=" * 55)
    print(" MAML 메타 학습 중... (사인파 태스크들)")
    print("=" * 55)
    inner_lr, k_shot, adapt_steps = 0.01, 10, 5
    theta, history = mc.train_maml(
        n_iters=8000, meta_batch=10, k_shot=k_shot,
        inner_lr=inner_lr, meta_lr=0.01, inner_steps=1, seed=0,
    )

    # 2) 새 태스크에 few-shot 적응 ---------------------------------------
    task = mc.SineTask(np.random.default_rng(42))      # 학습에 안 쓰인 새 사인파
    x_s, y_s = task.sample(k_shot, np.random.default_rng(7))
    x_dense = np.linspace(-5, 5, 200).reshape(-1, 1)
    y_true = task.true_curve(x_dense)

    def mse_to_true(params):
        pred, _ = mc.forward(params, x_dense)
        return float(np.mean((pred - y_true) ** 2))

    # 메타 초기값에서 적응
    pred_before, _ = mc.forward(theta, x_dense)         # 0스텝
    adapted = mc.inner_adapt(theta, x_s, y_s, inner_lr, adapt_steps)
    pred_after, _ = mc.forward(adapted, x_dense)

    # 무작위 초기값에서 같은 적응 (대조군)
    theta_rand = mc.init_params(np.random.default_rng(123), n_hidden=40)
    rand_adapted = mc.inner_adapt(theta_rand, x_s, y_s, inner_lr, adapt_steps)
    pred_rand, _ = mc.forward(rand_adapted, x_dense)

    print(f"  새 태스크 적응 (점 {k_shot}개, {adapt_steps}스텝):")
    print(f"   - 메타 초기값, 적응 전 MSE : {mse_to_true(theta):.3f}")
    print(f"   - 메타 초기값, 적응 후 MSE : {mse_to_true(adapted):.3f}  <- 확 줄어듦!")
    print(f"   - 무작위 초기값, 적응 후 MSE: {mse_to_true(rand_adapted):.3f}  (대조군)")

    # 3) 시각화 -----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sm = mc.moving_average(history, 100)
    axes[0].plot(range(len(sm)), sm, color="#4C72B0", linewidth=2)
    axes[0].set_title("MAML meta-training loss (query loss)")
    axes[0].set_xlabel("Meta-iteration")
    axes[0].set_ylabel("Loss (smoothed)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x_dense, y_true, "g-", linewidth=2.5, label="true sine")
    axes[1].scatter(x_s, y_s, color="black", s=50, zorder=5, label=f"{k_shot} points")
    axes[1].plot(x_dense, pred_before, ":", color="gray", linewidth=2,
                 label="meta-init (0 steps)")
    axes[1].plot(x_dense, pred_after, "-", color="#C44E52", linewidth=2.2,
                 label=f"MAML adapted ({adapt_steps} steps)")
    axes[1].plot(x_dense, pred_rand, "--", color="#CCB974", linewidth=2,
                 label=f"random-init adapted ({adapt_steps} steps)")
    axes[1].set_title("Few-shot adaptation on a NEW sine task")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    save_path = os.path.join(OUTPUT_DIR, "16_maml_sine.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 그래프 저장됨 -> {save_path}")
    print("\n  💡 해석: MAML 초기값에서 시작하면 점 10개·5스텝만으로 새 사인파에")
    print("     빠르게 적응한다(빨강). 무작위 초기값(노랑)은 잘 적응하지 못한다.")


if __name__ == "__main__":
    main()

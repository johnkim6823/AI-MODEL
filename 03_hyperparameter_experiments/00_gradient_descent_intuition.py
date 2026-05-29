"""
[직관] 학습률(Learning Rate)이 학습에 주는 영향 — 그릇 모델 ⭐
================================================================

"학습률이 왜 중요한가?"를 가장 단순한 예로 이해한다.

▶ 상황: 손실 함수가 f(w) = w² 인 '그릇' 모양이라고 하자.
  - 우리의 목표: 그릇의 바닥(w=0, 손실 최소)으로 내려가기.
  - 경사하강법: 현재 위치의 기울기(=2w)를 보고 반대로 이동.
        w <- w - 학습률 * 2w
  - 즉 한 걸음의 크기를 "학습률"이 정한다.

▶ 학습률에 따른 4가지 모습 (이 예에서 0<lr<1 이면 수렴, lr>=1 이면 발산)
  1) 너무 작음  : 한 걸음이 너무 작아 바닥까지 너무 느리게 감.
  2) 적당함     : 바닥으로 부드럽게 수렴. ✅
  3) 약간 큼    : 좌우로 튕기며(진동) 그래도 바닥으로 수렴.
  4) 너무 큼    : 걸음이 과해 바닥을 지나쳐 점점 멀어짐 → 발산(학습 실패).

이 데모는 실제 분류 정확도가 아니라 "학습률의 원리"를 보여주기 위한 것이다.
실제 데이터의 정확도 변화는 01_learning_rate_experiment.py 에서 확인한다.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def gradient_descent_on_bowl(lr, start=4.0, n_steps=30):
    """f(w)=w² 위에서 경사하강법 수행. 거쳐간 모든 w 위치를 반환."""
    w = start
    path = [w]
    for _ in range(n_steps):
        grad = 2 * w            # f(w)=w² 의 기울기
        w = w - lr * grad       # 학습률만큼 반대 방향으로 이동
        path.append(w)
    return np.array(path)


def main():
    # 각 학습률과 그 성격(설명) -----------------------------------------
    settings = [
        (0.03, "Too small (slow)"),
        (0.30, "Good (smooth)"),
        (0.90, "Large (oscillates)"),
        (1.05, "Too large (diverges)"),
    ]

    print("=" * 60)
    print(" 학습률 직관 데모: f(w)=w² 그릇 바닥(w=0)으로 내려가기")
    print("=" * 60)
    paths = []
    for lr, desc in settings:
        path = gradient_descent_on_bowl(lr)
        paths.append((lr, desc, path))
        final_w = path[-1]
        status = "발산!" if abs(final_w) > 10 else f"w={final_w:.4f}"
        print(f"  lr={lr:<5} {desc:22s} 30걸음 후 -> {status}")

    # 시각화: 위 2칸 = 손실 곡선, 아래 4칸 = 그릇 위 이동 경로 ----------
    fig = plt.figure(figsize=(14, 10))

    # (위) 학습률별 손실 f(w)=w² 이 줄어드는 모습 -------------------------
    ax_top = fig.add_subplot(2, 1, 1)
    for lr, desc, path in paths:
        losses = path ** 2
        ax_top.plot(losses, "o-", markersize=3, label=f"lr={lr} ({desc})")
    ax_top.set_title("Loss f(w)=w^2 over steps (per learning rate)")
    ax_top.set_xlabel("Step")
    ax_top.set_ylabel("Loss")
    ax_top.set_ylim(0, 25)   # 발산 케이스는 위로 뚫고 나가도록 (=학습 실패 표시)
    ax_top.legend(loc="upper right")
    ax_top.grid(True, alpha=0.3)
    ax_top.text(15, 22, "diverges -> infinity (off the top)",
                color="#C44E52", fontsize=9)

    # (아래) 각 학습률이 그릇 위에서 어떻게 움직이는지 ---------------------
    w_curve = np.linspace(-5, 5, 200)
    for idx, (lr, desc, path) in enumerate(paths):
        ax = fig.add_subplot(2, 4, 5 + idx)
        ax.plot(w_curve, w_curve ** 2, color="lightgray", linewidth=2)  # 그릇
        # 거쳐간 위치(점)와 이동(선)
        clipped = np.clip(path, -5, 5)  # 그래프 밖으로 나간 점은 가장자리에 표시
        ax.plot(clipped, clipped ** 2, "o-", color="#4C72B0",
                markersize=4, linewidth=1)
        ax.plot(path[0], path[0] ** 2, "o", color="green", markersize=9,
                label="start")
        ax.plot(0, 0, "*", color="red", markersize=14, label="goal (min)")
        ax.set_title(f"lr={lr}\n{desc}", fontsize=10)
        ax.set_xlim(-5.5, 5.5)
        ax.set_ylim(-1, 26)
        ax.set_xlabel("w")
        if idx == 0:
            ax.set_ylabel("f(w)")
            ax.legend(fontsize=8, loc="upper center")

    fig.suptitle("How Learning Rate Affects Gradient Descent", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    save_path = os.path.join(OUTPUT_DIR, "00_gradient_descent_intuition.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 그래프 저장됨 -> {save_path}")
    print("\n  💡 핵심: 학습률은 '한 걸음의 크기'다.")
    print("     작으면 느리고, 적당하면 잘 수렴하고, 너무 크면 발산한다.")


if __name__ == "__main__":
    main()

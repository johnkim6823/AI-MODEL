"""
[딥러닝] CNN 기초 — 합성곱이 하는 일 + 학습된 필터 ⭐
======================================================

▶ 합성곱(Convolution)이란?
  - 작은 필터(예: 3x3)를 이미지 위로 미끄러뜨리며 곱-합을 계산.
  - 필터 모양에 따라 '세로 모서리', '가로 모서리' 등 특정 패턴이 강조된다.

▶ 이 스크립트가 보여주는 것 (손글씨 숫자 8x8)
  (윗줄) 사람이 손으로 만든 필터들을 적용한 결과 → 합성곱의 직관
        (세로 모서리/가로 모서리/흐림/또렷하게)
  (아랫줄) CNN을 직접 학습시킨 뒤, 학습된 필터들이 만들어내는 특징맵
        → 신경망이 '스스로' 유용한 필터를 배운다는 것을 확인
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from cnn import SimpleCNN, conv2d

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def main():
    # 1) 데이터 준비 ------------------------------------------------------
    digits = load_digits()
    X = (digits.data / 16.0).reshape(-1, 1, 8, 8)   # (N,1,8,8)
    y = digits.target
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    # 시각화에 쓸 숫자 하나 (label=0 인 첫 샘플)
    sample = X[np.where(y == 0)[0][0]]               # (1,8,8)
    sample_batch = sample[None]                      # (1,1,8,8)

    # 2) 손으로 만든 필터들 (합성곱 직관) ---------------------------------
    hand_filters = {
        "Vertical edge": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
        "Horizontal edge": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
        "Blur": np.ones((3, 3)) / 9.0,
        "Sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
    }

    # 3) CNN 직접 학습 ----------------------------------------------------
    print("=" * 55)
    print(" CNN 기초 — 합성곱 + 학습된 필터")
    print("=" * 55)
    cnn = SimpleCNN(n_filters=8, kernel=3, learning_rate=0.01, seed=0)
    cnn.fit(X_tr, y_tr, n_epochs=20, batch_size=64, X_val=X_te, y_val=y_te, seed=0)
    test_acc = cnn.score(X_te, y_te)
    print(f"  CNN 구조: Conv(8필터,3x3) -> ReLU -> MaxPool(2x2) -> Dense(10)")
    print(f"  손실: {cnn.loss_history_[0]:.3f} -> {cnn.loss_history_[-1]:.3f}")
    print(f"  테스트 정확도: {test_acc:.3f}")

    # 학습된 필터가 만드는 특징맵 (conv+relu 결과)
    learned = np.maximum(0, conv2d(sample_batch, cnn.Wc, cnn.bc))  # (1,8,6,6)
    # 이 입력에서 가장 활성화가 큰 필터 5개를 골라 보여준다 (죽은 필터 제외)
    activity = learned[0].sum(axis=(1, 2))
    top_filters = np.argsort(activity)[::-1][:5]

    # 4) 시각화 -----------------------------------------------------------
    fig, axes = plt.subplots(2, 5, figsize=(16, 7))

    # (윗줄) 원본 + 손으로 만든 필터 결과
    axes[0, 0].imshow(sample[0], cmap="gray")
    axes[0, 0].set_title("Input digit", fontsize=11)
    for ax_idx, (name, kernel) in enumerate(hand_filters.items(), start=1):
        W = kernel.reshape(1, 1, 3, 3)
        out = conv2d(sample_batch, W, np.zeros(1))[0, 0]
        axes[0, ax_idx].imshow(out, cmap="gray")
        axes[0, ax_idx].set_title(name, fontsize=11)

    # (아랫줄) 학습된 CNN의 특징맵 (활성도 상위 5개)
    for col, k in enumerate(top_filters):
        axes[1, col].imshow(learned[0, k], cmap="viridis")
        axes[1, col].set_title(f"learned filter #{k}", fontsize=10)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    # 줄별 설명
    axes[0, 0].set_ylabel("Hand-crafted\nfilters", fontsize=12)
    axes[1, 0].set_ylabel("Learned filters\n(trained CNN)", fontsize=12)

    fig.suptitle(f"CNN basics: convolution extracts features "
                 f"(trained CNN test accuracy = {test_acc:.2f})", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = os.path.join(OUTPUT_DIR, "21_cnn_basics.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 그래프 저장됨 -> {save_path}")
    print("\n  💡 해석:")
    print("     - 윗줄: 필터 모양에 따라 모서리·흐림 등 다른 특징이 추출된다.")
    print("     - 아랫줄: CNN은 이런 필터를 '학습으로 스스로' 찾아내 숫자를 분류한다.")


if __name__ == "__main__":
    main()

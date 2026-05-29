"""
[실험 3] 학습률 × 반복 횟수 조합 탐색 (그리드 서치 + 히트맵) ⭐
==============================================================

질문: "학습률과 반복 횟수를 '동시에' 바꾸면 어떤 조합이 최고일까?"

실험 1, 2에서는 하나씩만 바꿨다. 이번에는 두 값을 모두 바꿔가며
모든 조합의 테스트 정확도를 표(히트맵)로 한눈에 본다.
이렇게 여러 하이퍼파라미터 조합을 전부 시도하는 것을
"그리드 서치(Grid Search)"라고 한다.

▶ 결과 해석
  - 히트맵에서 밝은(높은) 칸이 좋은 조합.
  - 보통 "학습률이 작으면 → 반복을 많이" 해야 하고,
         "학습률이 크면 → 적은 반복"으로도 수렴한다.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from model import GradientDescentClassifier

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def main():
    # 1) 데이터 준비 ------------------------------------------------------
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 2) 탐색할 하이퍼파라미터 격자(grid) 정의 ----------------------------
    learning_rates = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
    iteration_counts = [10, 50, 100, 300, 600, 1000]

    # 결과를 담을 2차원 표 (행=학습률, 열=반복횟수)
    acc_grid = np.zeros((len(learning_rates), len(iteration_counts)))

    print("=" * 60)
    print(" 실험 3: 학습률 × 반복 횟수 그리드 서치")
    print("=" * 60)
    for i, lr in enumerate(learning_rates):
        for j, n_iter in enumerate(iteration_counts):
            clf = GradientDescentClassifier(learning_rate=lr, n_iterations=n_iter)
            clf.fit(X_train, y_train)
            acc_grid[i, j] = clf.score(X_test, y_test)

    # 최고 조합 찾기
    best_idx = np.unravel_index(np.argmax(acc_grid), acc_grid.shape)
    best_lr = learning_rates[best_idx[0]]
    best_iter = iteration_counts[best_idx[1]]
    best_acc = acc_grid[best_idx]
    print(f"  ✅ 최고 조합: 학습률={best_lr}, 반복={best_iter}회 → 정확도 {best_acc:.4f}")

    # 3) 히트맵 시각화 ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(acc_grid, cmap="viridis", aspect="auto")

    # 축 눈금 라벨 달기
    ax.set_xticks(range(len(iteration_counts)))
    ax.set_xticklabels(iteration_counts)
    ax.set_yticks(range(len(learning_rates)))
    ax.set_yticklabels(learning_rates)
    ax.set_xlabel("Number of Iterations (Epochs)")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Test Accuracy Heatmap (Learning Rate x Iterations)", fontsize=13)

    # 각 칸에 정확도 숫자 표시
    for i in range(len(learning_rates)):
        for j in range(len(iteration_counts)):
            val = acc_grid[i, j]
            # 배경이 밝으면 검은 글씨, 어두우면 흰 글씨
            color = "white" if val < acc_grid.max() - 0.15 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=color, fontsize=9)

    # 최고 조합 칸에 빨간 테두리 표시
    ax.add_patch(plt.Rectangle((best_idx[1] - 0.5, best_idx[0] - 0.5), 1, 1,
                               fill=False, edgecolor="red", linewidth=3))

    fig.colorbar(im, ax=ax, label="Test Accuracy")

    save_path = os.path.join(OUTPUT_DIR, "08_grid_search_heatmap.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 그래프 저장됨 -> {save_path}")
    print("\n  💡 해석:")
    print("     - 빨간 테두리 칸이 최고 조합.")
    print("     - 학습률이 작을수록(위쪽) 같은 정확도를 내려면 반복을 더 많이 해야 한다.")
    print("     - 학습률이 충분하면(아래쪽) 적은 반복으로도 높은 정확도에 도달한다.")


if __name__ == "__main__":
    main()

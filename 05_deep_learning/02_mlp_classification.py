"""
[딥러닝] 신경망으로 비선형 분류 + 학습 곡선
=============================================

은닉층을 가진 신경망(MLP)이 곡선 모양의 복잡한 경계를 학습하는 과정을 본다.

▶ 사용 데이터: 초승달(moons) 두 개가 맞물린 모양
  - 직선으로는 못 나누는 전형적인 비선형 데이터.

▶ 보는 것
  (왼쪽) 학습 곡선: 에폭이 지날수록 손실↓, 정확도↑
  (오른쪽) 학습된 결정 경계: 신경망이 만든 곡선 경계
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from neural_network import MLPClassifier

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def main():
    # 1) 데이터 + 학습 ----------------------------------------------------
    X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = MLPClassifier(hidden_layers=(16, 16), activation="relu",
                          learning_rate=0.2, n_epochs=400, seed=0)
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    print("=" * 55)
    print(" 신경망(MLP) 비선형 분류 — moons 데이터")
    print("=" * 55)
    print(f"  구조: 입력2 -> 은닉(16) -> 은닉(16) -> 출력2, 활성화=ReLU")
    print(f"  학습 정확도  : {train_acc:.3f}")
    print(f"  테스트 정확도: {test_acc:.3f}")
    print(f"  최종 손실    : {model.loss_history_[-1]:.4f}")

    # 2) 시각화 -----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # (왼쪽) 학습 곡선 (손실 + 정확도, 축 2개)
    ax1 = axes[0]
    ax1.plot(model.loss_history_, color="#C44E52", label="loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="#C44E52")
    ax1.tick_params(axis="y", labelcolor="#C44E52")
    ax2 = ax1.twinx()
    ax2.plot(model.acc_history_, color="#4C72B0", label="train accuracy")
    ax2.set_ylabel("Accuracy", color="#4C72B0")
    ax2.tick_params(axis="y", labelcolor="#4C72B0")
    ax1.set_title("Training Curve (loss down, accuracy up)")
    ax1.grid(True, alpha=0.3)

    # (오른쪽) 결정 경계
    ax = axes[1]
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.25, cmap="coolwarm")
    ax.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1],
               color="#4C72B0", edgecolor="k", s=25, label="class 0")
    ax.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1],
               color="#C44E52", edgecolor="k", s=25, label="class 1")
    ax.set_title(f"Learned Decision Boundary (test acc={test_acc:.2f})")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend()

    save_path = os.path.join(OUTPUT_DIR, "13_mlp_classification.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 그래프 저장됨 -> {save_path}")
    print("\n  💡 해석: 신경망은 직선이 아닌 '곡선' 경계를 학습해 초승달을 잘 나눈다.")


if __name__ == "__main__":
    main()

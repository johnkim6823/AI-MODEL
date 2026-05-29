"""
[딥러닝 - 기초] 왜 은닉층이 필요한가? — XOR 문제
==================================================

딥러닝이 왜 "층을 깊게 쌓는지"를 가장 단순한 예로 이해한다.

▶ XOR 문제
  - 두 입력이 서로 '다르면' 1, '같으면' 0 인 패턴.
        (0,0)->0   (0,1)->1   (1,0)->1   (1,1)->0
  - 이 네 점은 '직선 하나'로는 두 그룹으로 나눌 수 없다! (선형 분리 불가)

▶ 핵심 교훈
  - 은닉층 없는 모델(=선형 분류기)은 XOR을 못 푼다 → 정확도 ≈ 50%(찍기 수준).
  - 은닉층 + 비선형 활성화 함수를 넣으면 곡선 경계를 만들 수 있어 → 100% 해결.
  - 즉 '은닉층의 비선형성'이 딥러닝의 표현력의 핵심이다.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from neural_network import MLPClassifier

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def make_xor(n_per_corner=75, noise=0.12, seed=0):
    """네 모서리 주변에 점을 뿌린 '균형 잡힌' XOR 데이터."""
    rng = np.random.default_rng(seed)
    corners = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]  # (x1, x2, label)
    Xs, ys = [], []
    for cx, cy, lab in corners:
        Xs.append(rng.normal([cx, cy], noise, (n_per_corner, 2)))
        ys.append(np.full(n_per_corner, lab))
    return np.vstack(Xs), np.concatenate(ys)


def plot_boundary(ax, model, X, y, title):
    """결정 경계 + 데이터 점을 그린다."""
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.25, cmap="coolwarm")
    ax.scatter(X[y == 0, 0], X[y == 0, 1], color="#4C72B0", edgecolor="k", s=25, label="0")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], color="#C44E52", edgecolor="k", s=25, label="1")
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend()


def main():
    X, y = make_xor()

    # 은닉층 없음(=선형) vs 은닉층 2개
    linear = MLPClassifier(hidden_layers=(), n_epochs=200,
                           learning_rate=0.1, seed=0).fit(X, y)
    mlp = MLPClassifier(hidden_layers=(8, 8), activation="relu",
                        n_epochs=400, learning_rate=0.3, seed=0).fit(X, y)

    acc_lin = linear.score(X, y)
    acc_mlp = mlp.score(X, y)

    print("=" * 55)
    print(" XOR 문제: 은닉층의 필요성")
    print("=" * 55)
    print(f"  선형 모델(은닉층 없음) 정확도: {acc_lin:.3f}  <- 직선으론 못 풀어 ≈50%")
    print(f"  MLP(은닉층 8,8)        정확도: {acc_mlp:.3f}  <- 곡선 경계로 해결!")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    plot_boundary(axes[0], linear, X, y,
                  f"No hidden layer (linear)\naccuracy = {acc_lin:.2f}")
    plot_boundary(axes[1], mlp, X, y,
                  f"MLP with hidden layers (8,8)\naccuracy = {acc_mlp:.2f}")
    fig.suptitle("Why we need hidden layers: the XOR problem", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = os.path.join(OUTPUT_DIR, "12_perceptron_vs_mlp.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 그래프 저장됨 -> {save_path}")
    print("\n  💡 해석: 왼쪽(직선)은 XOR을 못 나눈다. 오른쪽(은닉층)은 곡선으로 완벽히 나눈다.")


if __name__ == "__main__":
    main()

"""
[딥러닝 - 실험] 정규화로 과적합 줄이기 (L2 · 드롭아웃) ⭐
=========================================================

▶ 과적합(Overfitting) 복습
  - 망이 크고 데이터가 적으면, 모델이 학습 데이터를 '통째로 외워'
    학습 정확도는 100%지만 새 데이터(테스트)에는 약해진다.

▶ 정규화(Regularization): 과적합을 억제하는 기법
  - L2(가중치 감쇠): 가중치가 너무 커지지 않게 페널티를 줘 모델을 '단순'하게.
  - 드롭아웃(Dropout): 학습 중 은닉 뉴런을 무작위로 꺼서 특정 뉴런에
    과하게 의존하지 않게 만든다.

▶ 실험 설정 (과적합이 잘 일어나도록)
  - 특성 50개인데 학습 데이터는 80개뿐 (적은 데이터 + 큰 망 128x128)
  - 무작위성을 줄이려 여러 seed로 학습해 평균낸다.

  (왼쪽) 규제 없을 때: 학습 정확도↑(거의 100%)인데 테스트는 낮음 → 과적합
  (오른쪽) 규제 없음 vs L2 vs 드롭아웃의 테스트 정확도 → 규제가 테스트를 끌어올림
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from neural_network import MLPClassifier

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")

N_SEEDS = 5
N_EPOCHS = 300


def train_avg(X_tr, y_tr, X_te, y_te, **kw):
    """여러 seed로 학습해 (평균 train 곡선, 평균 test 곡선)을 반환."""
    train_curves, test_curves = [], []
    for sd in range(N_SEEDS):
        m = MLPClassifier(hidden_layers=(128, 128), learning_rate=0.1,
                          n_epochs=N_EPOCHS, seed=sd, **kw)
        m.fit(X_tr, y_tr, X_val=X_te, y_val=y_te)
        train_curves.append(m.acc_history_)
        test_curves.append(m.val_acc_history_)
    return np.mean(train_curves, axis=0), np.mean(test_curves, axis=0)


def main():
    X, y = make_classification(n_samples=400, n_features=50, n_informative=10,
                               n_redundant=10, flip_y=0.05, random_state=0)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.8, random_state=0, stratify=y)

    print("=" * 60)
    print(" 정규화(L2 · 드롭아웃)로 과적합 줄이기")
    print(f"   학습 {len(y_tr)}개 / 테스트 {len(y_te)}개, 특성 {X.shape[1]}개, 망(128,128)")
    print("=" * 60)

    configs = [
        ("No regularization", {}, "#C44E52"),
        ("L2 (l2=0.1)", {"l2": 0.1}, "#55A868"),
        ("Dropout (0.3)", {"dropout": 0.3}, "#4C72B0"),
    ]
    results = {}
    for name, kw, _ in configs:
        tr, te = train_avg(X_tr, y_tr, X_te, y_te, **kw)
        results[name] = (tr, te)
        print(f"  {name:20s}: train={tr[-1]:.3f}  test(최종)={te[-1]:.3f}  test(최고)={te.max():.3f}")

    # 시각화 --------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # (왼쪽) 규제 없음: 학습 vs 테스트 (과적합 갭)
    tr0, te0 = results["No regularization"]
    axes[0].plot(tr0, color="#4C72B0", linewidth=2, label="train accuracy")
    axes[0].plot(te0, color="#C44E52", linewidth=2, label="test accuracy")
    axes[0].fill_between(range(len(tr0)), te0, tr0, color="orange", alpha=0.15)
    axes[0].text(len(tr0) * 0.5, (tr0[-1] + te0[-1]) / 2, "overfitting gap",
                 color="darkorange", fontsize=11, ha="center")
    axes[0].set_title("No regularization: big train-test gap (overfitting)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # (오른쪽) 규제별 테스트 정확도 비교
    for name, _, color in configs:
        _, te = results[name]
        axes[1].plot(te, color=color, linewidth=2, label=name)
    axes[1].set_title("Test accuracy: regularization helps generalization")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Test Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Regularization (L2 & Dropout) reduces overfitting", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = os.path.join(OUTPUT_DIR, "19_regularization.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 그래프 저장됨 -> {save_path}")
    print("\n  💡 해석:")
    print("     - 규제가 없으면 train≈100%인데 test는 낮다(과적합 갭이 큼).")
    print("     - L2·드롭아웃은 모델을 단순하게 만들어 테스트 정확도를 끌어올린다.")
    print("     - 단, 규제가 너무 강하면 거꾸로 과소적합(둘 다 낮아짐)이 된다.")


if __name__ == "__main__":
    main()

"""
[실험 2] 반복 횟수(Epochs/Iterations)에 따른 정확도 변화 ⭐
============================================================

질문: "더 오래(많이) 학습시키면 항상 더 좋아질까?"

학습률은 고정하고, 반복 횟수에 따라
  - 학습(train) 정확도
  - 테스트(test) 정확도
가 어떻게 변하는지 함께 그려본다.

▶ 관찰 포인트
  - 처음엔 둘 다 빠르게 오른다 (학습이 진행됨).
  - 어느 순간부터 train은 계속 올라 1.0(100%)에 가까워지지만
    test는 정체하거나 오히려 떨어진다.
    → 이것이 "과적합(Overfitting)": 학습 데이터(정답+잡음)를 통째로
       외워버려서, 처음 보는 데이터엔 약해지는 현상.
  - 그래서 무조건 오래 학습한다고 좋은 게 아니다.
    test 정확도가 가장 높은 시점에 멈추는 것을 "조기 종료(Early Stopping)"라 한다.

▶ 과적합을 잘 보이게 하려고:
  - 특성(80개)이 샘플 수에 비해 많고 라벨 잡음이 섞인 데이터를 쓴다.
    특성이 많으면 모델이 학습 데이터를 "통째로 외우기" 쉬워진다.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from model import GradientDescentClassifier

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def main():
    # 1) 데이터 준비: 특성은 많고 라벨 잡음 섞기 → 과적합이 잘 일어나는 환경 ----
    X, y = make_classification(
        n_samples=250,
        n_features=80,      # 특성이 많음 (샘플 수에 비해)
        n_informative=12,   # 실제로 유용한 특성은 12개 (나머지는 잡음에 가까움)
        n_redundant=8,
        flip_y=0.08,        # 8%의 라벨에 잡음 → 과적합 유도
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 2) 학습률 고정 + 충분히 많은 반복으로 학습 (단 한 번의 fit으로 곡선 기록) --
    learning_rate = 0.1
    n_iterations = 3000

    clf = GradientDescentClassifier(
        learning_rate=learning_rate,
        n_iterations=n_iterations,
        record_history=True,
    )
    clf.fit(X_train, y_train, X_val=X_test, y_val=y_test)

    train_acc_history = clf.train_acc_history
    test_acc_history = clf.val_acc_history

    # 테스트 정확도가 가장 높았던 시점 (조기 종료에 적합한 지점)
    best_iter = int(np.argmax(test_acc_history))
    best_test_acc = test_acc_history[best_iter]

    print("=" * 60)
    print(" 실험 2: 반복 횟수(Epochs)에 따른 정확도")
    print(f"         (학습률 {learning_rate} 고정, 특성 80개 / 샘플 250개)")
    print("=" * 60)
    checkpoints = [1, 10, 50, 100, 500, 1000, 3000]
    print(f"  {'반복 횟수':>10} | {'학습 정확도':>10} | {'테스트 정확도':>12}")
    print("  " + "-" * 40)
    for it in checkpoints:
        idx = min(it, n_iterations) - 1
        print(f"  {it:>10} | {train_acc_history[idx]:>10.4f} | {test_acc_history[idx]:>12.4f}")
    print("  " + "-" * 40)
    print(f"  학습 정확도 최종값  : {train_acc_history[-1]:.4f}  (1.0에 근접 = 학습데이터 암기)")
    print(f"  ✅ 테스트 정확도 최고: {best_test_acc:.4f}  ({best_iter+1}회 시점)")
    print(f"     이후 반복을 늘려도 테스트 정확도는 더 좋아지지 않는다(과적합).")

    # 3) 시각화 -----------------------------------------------------------
    plt.figure(figsize=(10, 6))
    iters = range(1, n_iterations + 1)
    plt.plot(iters, train_acc_history, label="Train Accuracy",
             color="#4C72B0", linewidth=2)
    plt.plot(iters, test_acc_history, label="Test Accuracy",
             color="#C44E52", linewidth=2)
    plt.axvline(best_iter + 1, color="green", linestyle="--",
                label=f"Best test point (iter={best_iter+1})")

    # 과적합 구간을 음영으로 표시 (테스트 최고점 이후)
    plt.axvspan(best_iter + 1, n_iterations, color="orange", alpha=0.08)
    plt.text((best_iter + 1 + n_iterations) / 2, 0.55, "Overfitting zone\n(train UP but test DOWN)",
             ha="center", va="center", fontsize=10, color="darkorange")

    plt.title(f"Accuracy vs Iterations (learning rate = {learning_rate})", fontsize=13)
    plt.xlabel("Number of Iterations (Epochs)")
    plt.ylabel("Accuracy")
    plt.ylim(0.45, 1.02)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(OUTPUT_DIR, "07_epochs_experiment.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 그래프 저장됨 -> {save_path}")
    print("\n  💡 해석:")
    print("     - 초반: train/test 정확도가 함께 오른다 (학습 진행).")
    print("     - 후반: train은 1.0(100%)까지 오르지만 test는 정체/하락 (과적합).")
    print("     - 따라서 '적당한 시점에 멈추기(Early Stopping)'가 중요하다.")


if __name__ == "__main__":
    main()

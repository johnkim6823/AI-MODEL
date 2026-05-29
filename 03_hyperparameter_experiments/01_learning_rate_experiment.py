"""
[실험 1] 학습률(Learning Rate)에 따른 정확도 변화 ⭐
=====================================================

질문: "학습률을 다르게 주면 학습 결과(정확도)가 어떻게 달라질까?"

같은 모델·같은 반복 횟수(300회 고정)로, 학습률만 바꿔가며 학습시켜 본다.
  - 너무 작은 학습률 (0.0001) : 한 걸음이 작아 300회 안에 충분히 못 배움 → 정확도 낮음
  - 적당한 학습률   (0.1~1.0) : 같은 횟수로 빠르게 수렴 → 정확도 높음 ✅

핵심 메시지:
  "반복 횟수가 정해져 있을 때, 학습률이 너무 작으면 '시간이 모자라'
   덜 배운 상태로 끝나 정확도가 낮다."

※ "학습률이 너무 크면 발산한다"는 개념은 00_gradient_descent_intuition.py
   에서 확인하자. (여기 쓰는 로지스틱 회귀 + 표준화 데이터는 볼록·안정적이라
   학습률이 꽤 커도 발산하지 않는다. 모델/데이터에 따라 다르다.)

그래프 2개를 그린다.
  (왼쪽) 반복에 따른 손실(loss) 곡선 - 학습률별로 비교
  (오른쪽) 학습률별 최종 테스트 정확도 - 막대그래프
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from model import GradientDescentClassifier

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def main():
    # 1) 데이터 준비 (유방암 진단, 이진 분류) ----------------------------
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 경사하강법은 특성 크기에 민감하므로 반드시 표준화한다.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 2) 여러 학습률로 각각 학습 ------------------------------------------
    learning_rates = [0.0001, 0.001, 0.01, 0.1, 1.0]
    n_iterations = 300  # 반복 횟수는 고정! (학습률 효과만 비교하기 위해)

    results = []  # (lr, 최종 테스트 정확도, 손실 기록)
    print("=" * 60)
    print(" 실험 1: 학습률(Learning Rate)에 따른 정확도")
    print(f"         (반복 횟수 {n_iterations}회 고정)")
    print("=" * 60)
    print(f"  {'학습률':>10} | {'테스트 정확도':>12}")
    print("  " + "-" * 28)

    for lr in learning_rates:
        clf = GradientDescentClassifier(
            learning_rate=lr, n_iterations=n_iterations, record_history=True
        )
        clf.fit(X_train, y_train)
        test_acc = clf.score(X_test, y_test)
        results.append((lr, test_acc, clf.loss_history))
        print(f"  {lr:>10} | {test_acc:>12.4f}")

    # 가장 좋은 학습률 찾기
    best_lr, best_acc, _ = max(results, key=lambda r: r[1])
    print("  " + "-" * 28)
    print(f"  ✅ 최적 학습률: {best_lr}  (정확도 {best_acc:.4f})")

    # 3) 시각화 -----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # (왼쪽) 학습률별 손실 곡선
    for lr, _, loss_hist in results:
        axes[0].plot(loss_hist, label=f"lr={lr}", linewidth=2)
    axes[0].set_title("Loss curve per Learning Rate")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Loss (lower = better)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    # 손실이 발산하는 경우가 있어 y축을 보기 좋게 제한
    axes[0].set_ylim(0, 1.5)

    # (오른쪽) 학습률별 최종 테스트 정확도
    lrs = [str(r[0]) for r in results]
    accs = [r[1] for r in results]
    colors = ["#C44E52" if a == best_acc else "#4C72B0" for a in accs]
    bars = axes[1].bar(lrs, accs, color=colors)
    for bar, acc in zip(bars, accs):
        axes[1].text(bar.get_x() + bar.get_width() / 2, acc + 0.01,
                     f"{acc:.3f}", ha="center", va="bottom", fontsize=9)
    axes[1].set_title("Final Test Accuracy per Learning Rate")
    axes[1].set_xlabel("Learning Rate")
    axes[1].set_ylabel("Test Accuracy")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, axis="y", alpha=0.3)

    save_path = os.path.join(OUTPUT_DIR, "06_learning_rate_experiment.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 그래프 저장됨 -> {save_path}")
    print("\n  💡 해석:")
    print("     - 학습률이 너무 작으면(0.0001) 300회 안에 손실이 거의 안 줄어 정확도가 낮다.")
    print("     - 학습률이 커질수록 같은 횟수로 더 빨리 수렴해 정확도가 높아진다.")
    print("     - (너무 큰 학습률의 '발산'은 00_gradient_descent_intuition.py 참고)")


if __name__ == "__main__":
    main()

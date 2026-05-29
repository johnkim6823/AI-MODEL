"""
[딥러닝 - 실험] 옵티마이저 비교: SGD vs Momentum vs Adam ⭐
============================================================

옵티마이저(optimizer)는 "기울기로 가중치를 어떻게 갱신할지"를 정하는 규칙이다.
같은 신경망·데이터라도 옵티마이저에 따라 학습 속도와 안정성이 달라진다.

▶ 세 가지 옵티마이저
  - SGD       : 가장 기본. w ← w - lr·g  (기울기 방향으로 그냥 한 걸음)
  - Momentum  : 이전 갱신 방향을 관성처럼 누적 → 골짜기를 더 빠르게 내려감
  - Adam      : 각 파라미터마다 보폭을 자동 조절(기울기 1·2차 모멘트 사용)
                → 보통 가장 빠르고 튜닝에 덜 민감

▶ 결과
  (왼쪽) 학습 손실 곡선: 어떤 옵티마이저가 더 빨리 손실을 줄이는가
  (오른쪽) 테스트 정확도 곡선: 더 빨리 좋은 성능에 도달하는가
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from neural_network import MLPClassifier

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def main():
    # 손글씨 숫자(8x8) 데이터 — 64차원 입력, 10개 클래스
    digits = load_digits()
    X, y = digits.data / 16.0, digits.target  # 0~1로 정규화
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    # 각 옵티마이저에 맞는 일반적인 학습률 사용
    configs = [
        ("SGD", "sgd", 0.3, "#4C72B0"),
        ("Momentum", "momentum", 0.1, "#55A868"),
        ("Adam", "adam", 0.01, "#C44E52"),
    ]

    print("=" * 60)
    print(" 옵티마이저 비교 (손글씨 숫자 분류, 망(64,))")
    print("=" * 60)

    results = {}
    for name, opt, lr, _ in configs:
        m = MLPClassifier(hidden_layers=(64,), optimizer=opt, learning_rate=lr,
                          n_epochs=50, batch_size=64, seed=0)
        m.fit(X_tr, y_tr, X_val=X_te, y_val=y_te)
        results[name] = (m.loss_history_, m.val_acc_history_)
        # 손실이 0.1 아래로 처음 내려간 에폭 (수렴 속도 지표)
        reached = next((i + 1 for i, l in enumerate(m.loss_history_) if l < 0.1), None)
        print(f"  {name:9s}: 최종 손실 {m.loss_history_[-1]:.4f}, "
              f"테스트 정확도 {m.val_acc_history_[-1]:.3f}, "
              f"손실<0.1 도달 에폭 {reached}")

    # 시각화 --------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for name, _, _, color in configs:
        loss, acc = results[name]
        axes[0].plot(loss, color=color, linewidth=2, label=name)
        axes[1].plot(acc, color=color, linewidth=2, label=name)

    axes[0].set_title("Training Loss (faster down = better optimizer)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Test Accuracy over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Test Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Optimizer Comparison: SGD vs Momentum vs Adam", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = os.path.join(OUTPUT_DIR, "20_optimizer_comparison.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 그래프 저장됨 -> {save_path}")
    print("\n  💡 해석:")
    print("     - Momentum/Adam은 SGD보다 손실을 더 빨리 줄이는 경향.")
    print("     - Adam은 파라미터마다 보폭을 자동 조절해 보통 가장 빠르게 수렴한다.")


if __name__ == "__main__":
    main()

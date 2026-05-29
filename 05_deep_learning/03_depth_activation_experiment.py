"""
[딥러닝 - 실험] 망의 깊이·너비·활성화 함수에 따른 성능 ⭐
==========================================================

지도학습에서 학습률·반복 횟수를 바꿔봤듯, 딥러닝에서는 신경망의
'구조(깊이/너비)'와 '활성화 함수'가 중요한 하이퍼파라미터다.
이 값들을 바꿔가며 정확도가 어떻게 변하는지 본다.

  ① 망 구조(은닉층 크기) → 테스트 정확도
       작으면(또는 은닉층 없음) 표현력이 부족(과소적합),
       적당히 키우면 복잡한 패턴을 학습.
  ② 활성화 함수(ReLU/tanh/sigmoid) → 학습 곡선(수렴 속도)
       sigmoid는 깊은 망에서 학습이 느려지기 쉽고(기울기 소실),
       ReLU가 보통 빠르고 안정적이다.
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
    X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print("=" * 60)
    print(" 딥러닝 하이퍼파라미터 실험 (moons 데이터)")
    print("=" * 60)

    # ① 망 구조(깊이/너비) 비교 -----------------------------------------
    architectures = [
        ((), "no hidden\n(linear)"),
        ((4,), "(4)"),
        ((16,), "(16)"),
        ((16, 16), "(16,16)"),
        ((64, 64), "(64,64)"),
    ]
    arch_labels, arch_accs = [], []
    print("\n  [① 망 구조] 활성화=ReLU, lr=0.2, 400 epoch")
    for hidden, label in architectures:
        m = MLPClassifier(hidden_layers=hidden, activation="relu",
                          learning_rate=0.2, n_epochs=400, seed=0).fit(X_train, y_train)
        acc = m.score(X_test, y_test)
        arch_labels.append(label)
        arch_accs.append(acc)
        print(f"     hidden={str(hidden):10s} -> 테스트 정확도 {acc:.3f}")

    # ② 활성화 함수 비교 (학습 곡선) -------------------------------------
    activations = ["relu", "tanh", "sigmoid"]
    colors = {"relu": "#4C72B0", "tanh": "#55A868", "sigmoid": "#C44E52"}
    curves = {}
    print("\n  [② 활성화 함수] 구조=(16,16), lr=0.2, 400 epoch")
    for act in activations:
        m = MLPClassifier(hidden_layers=(16, 16), activation=act,
                          learning_rate=0.2, n_epochs=400, seed=0).fit(X_train, y_train)
        curves[act] = m.loss_history_
        print(f"     {act:8s} -> 테스트 정확도 {m.score(X_test, y_test):.3f}, "
              f"최종 손실 {m.loss_history_[-1]:.4f}")

    # 시각화 --------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    best_i = int(np.argmax(arch_accs))
    bar_colors = ["#C44E52" if i == best_i else "#4C72B0" for i in range(len(arch_accs))]
    bars = axes[0].bar(arch_labels, arch_accs, color=bar_colors)
    for bar, acc in zip(bars, arch_accs):
        axes[0].text(bar.get_x() + bar.get_width() / 2, acc + 0.005,
                     f"{acc:.3f}", ha="center", va="bottom", fontsize=9)
    axes[0].set_title("(1) Effect of Network Size on Accuracy")
    axes[0].set_xlabel("Hidden layers")
    axes[0].set_ylabel("Test Accuracy")
    axes[0].set_ylim(min(arch_accs) - 0.05, 1.02)
    axes[0].grid(True, axis="y", alpha=0.3)

    for act in activations:
        axes[1].plot(curves[act], color=colors[act], linewidth=2, label=act)
    axes[1].set_title("(2) Effect of Activation Function on Training Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Training Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Deep Learning: Architecture & Activation Effects", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = os.path.join(OUTPUT_DIR, "14_depth_activation_experiment.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 그래프 저장됨 -> {save_path}")
    print("\n  💡 해석:")
    print("     - 은닉층이 없으면(선형) moons를 못 풀어 정확도가 낮다.")
    print("     - 은닉층을 키울수록 표현력이 늘어 정확도가 올라간다(어느 선에서 포화).")
    print("     - 활성화 함수에 따라 수렴 속도가 다르다 (보통 ReLU가 빠르고 안정적).")


if __name__ == "__main__":
    main()

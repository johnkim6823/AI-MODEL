"""
[강화학습 - 실험] 하이퍼파라미터에 따른 학습 성능 변화 ⭐
==========================================================

지도학습에서 학습률·반복 횟수를 바꿔봤듯이, 강화학습에서도
하이퍼파라미터를 바꾸면 학습이 어떻게 달라지는지 본다.

같은 GridWorld + Q러닝에서 아래 값들을 바꿔가며 학습 곡선을 비교한다.
  ① 학습률 α   : Q값을 한 번에 얼마나 갱신할지
  ② 탐험율 ε   : 얼마나 자주 무작위로 탐험할지
  ③ 할인율 γ   : 미래 보상을 얼마나 중요하게 볼지
  ④ (요약) 학습률별 최종 성능 비교

* 무작위성이 크므로 여러 번(seed) 돌려 평균을 낸 학습 곡선을 그린다.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gridworld import GridWorld, moving_average, q_learning

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")

N_EPISODES = 400
N_SEEDS = 20  # 여러 번 돌려 평균 (곡선을 안정적으로)


def avg_curve(alpha, gamma, epsilon):
    """여러 seed로 학습해 '에피소드별 평균 보상' 곡선을 반환."""
    all_rewards = []
    for sd in range(N_SEEDS):
        env = GridWorld()
        _, rewards, _ = q_learning(
            env, n_episodes=N_EPISODES, alpha=alpha, gamma=gamma,
            epsilon=epsilon, seed=sd,
        )
        all_rewards.append(rewards)
    return np.mean(all_rewards, axis=0)


def main():
    # 기본값 (한 가지만 바꾸고 나머지는 고정)
    base = dict(alpha=0.1, gamma=0.95, epsilon=0.1)

    print("=" * 60)
    print(" 강화학습 하이퍼파라미터 실험 (GridWorld + Q러닝)")
    print(f"   에피소드 {N_EPISODES}, seed {N_SEEDS}회 평균")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # ① 학습률 α 비교 ----------------------------------------------------
    alphas = [0.01, 0.1, 0.5, 0.9]
    final_perf_alpha = []
    print("\n  [① 학습률 α]  (γ=0.95, ε=0.1 고정)")
    for a in alphas:
        curve = avg_curve(alpha=a, gamma=base["gamma"], epsilon=base["epsilon"])
        sm = moving_average(curve, 20)
        axes[0, 0].plot(range(len(sm)), sm, linewidth=2, label=f"α={a}")
        final = curve[-50:].mean()
        final_perf_alpha.append(final)
        print(f"     α={a:<5} -> 후반 평균 보상 {final:7.2f}")
    axes[0, 0].set_title("(1) Effect of Learning Rate α")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Avg Reward (smoothed)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # ② 탐험율 ε 비교 ----------------------------------------------------
    epsilons = [0.0, 0.05, 0.1, 0.3]
    print("\n  [② 탐험율 ε]  (α=0.1, γ=0.95 고정)")
    for e in epsilons:
        curve = avg_curve(alpha=base["alpha"], gamma=base["gamma"], epsilon=e)
        sm = moving_average(curve, 20)
        axes[0, 1].plot(range(len(sm)), sm, linewidth=2, label=f"ε={e}")
        print(f"     ε={e:<5} -> 후반 평균 보상 {curve[-50:].mean():7.2f}")
    axes[0, 1].set_title("(2) Effect of Exploration ε")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Avg Reward (smoothed)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # ③ 할인율 γ 비교 ----------------------------------------------------
    gammas = [0.5, 0.9, 0.95, 0.99]
    print("\n  [③ 할인율 γ]  (α=0.1, ε=0.1 고정)")
    for g in gammas:
        curve = avg_curve(alpha=base["alpha"], gamma=g, epsilon=base["epsilon"])
        sm = moving_average(curve, 20)
        axes[1, 0].plot(range(len(sm)), sm, linewidth=2, label=f"γ={g}")
        print(f"     γ={g:<5} -> 후반 평균 보상 {curve[-50:].mean():7.2f}")
    axes[1, 0].set_title("(3) Effect of Discount Factor γ")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Avg Reward (smoothed)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # ④ 학습률별 최종 성능 요약 막대 -------------------------------------
    best_i = int(np.argmax(final_perf_alpha))
    bar_colors = ["#C44E52" if i == best_i else "#4C72B0" for i in range(len(alphas))]
    bars = axes[1, 1].bar([str(a) for a in alphas], final_perf_alpha, color=bar_colors)
    for bar, val in zip(bars, final_perf_alpha):
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2, val + 0.1,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=10)
    axes[1, 1].set_title("(4) Final Performance vs Learning Rate α")
    axes[1, 1].set_xlabel("Learning Rate α")
    axes[1, 1].set_ylabel("Final Avg Reward (last 50)")
    axes[1, 1].grid(True, axis="y", alpha=0.3)

    fig.suptitle("Reinforcement Learning: Hyperparameter Effects", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    save_path = os.path.join(OUTPUT_DIR, "11_rl_hyperparameter_experiments.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 그래프 저장됨 -> {save_path}")
    print(f"  ✅ 최적 학습률 α = {alphas[best_i]} (후반 평균 보상 {final_perf_alpha[best_i]:.2f})")
    print("\n  💡 해석:")
    print("     - 학습률 α: 너무 작으면(0.01) 학습이 매우 느려 400 에피소드로도 모자란다.")
    print("                적당히 크면(0.1~0.9) 빠르게 수렴한다. (지도학습 학습률과 같은 직관)")
    print("     - 탐험율 ε: 너무 크면(0.3) 학습 후에도 무작위 행동을 자주 해 보상이 낮다.")
    print("                이 환경은 걸음마다 -1이라 미탐험 행동(Q=0)이 매력적으로 보여")
    print("                ε=0이어도 자연스레 탐험이 일어나 잘 배운다(낙관적 초기화 효과).")
    print("                ※ 보상이 드문 더 어려운 환경에서는 탐험(ε>0)이 꼭 필요하다.")
    print("     - 할인율 γ: 이 작은 문제에선 영향이 작다. 목표가 멀거나 보상이 드문")
    print("                장기 문제에서 γ(미래를 얼마나 중시할지)가 중요해진다.")


if __name__ == "__main__":
    main()

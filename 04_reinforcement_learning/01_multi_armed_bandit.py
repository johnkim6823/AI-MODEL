"""
[강화학습 - 기초] 멀티암드 밴딧 (Multi-Armed Bandit)
=====================================================

강화학습의 가장 단순한 문제로 "탐험 vs 활용"을 이해한다.

▶ 상황
  - 슬롯머신(arm)이 10대 있다. 각 머신은 평균 보상이 다르지만 우리는 모른다.
  - 한 번에 한 대를 당겨 보상을 받는다.
  - 1000번 당기는 동안 총 보상을 최대로 하고 싶다.

▶ 딜레마: 탐험(Exploration) vs 활용(Exploitation)
  - 활용: 지금까지 제일 좋아 보이는 머신만 계속 당긴다 → 더 좋은 걸 놓칠 수 있음
  - 탐험: 가끔 다른 머신도 시도해 본다 → 더 좋은 머신을 찾을 기회

▶ ε-greedy 전략
  - 확률 ε 로 무작위 머신(탐험), 확률 (1-ε) 로 현재 최선 머신(활용).
  - ε 값에 따라 결과가 어떻게 달라지는지 그래프로 비교한다.
      ε=0    : 순수 활용 (탐험 안 함) → 운 나쁘면 나쁜 머신에 갇힘
      ε=0.01 : 아주 가끔 탐험
      ε=0.1  : 적당히 탐험 ✅
      ε=0.3  : 너무 자주 탐험 → 좋은 걸 알면서도 딴 데를 자주 당김
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def run_bandit(epsilon, k=10, steps=1000, runs=2000, seed=0):
    """ε-greedy 밴딧 실험. 여러 번(runs) 반복해 평균을 낸다.

    속도를 위해 모든 run을 numpy 배열로 동시에 계산한다.
    반환: 각 step의 (평균 보상, 최적 머신 선택 비율)
    """
    rng = np.random.default_rng(seed)
    # 각 run마다 머신들의 '진짜' 평균 보상 (우리는 모른다고 가정)
    q_true = rng.normal(0, 1, size=(runs, k))
    optimal_arm = np.argmax(q_true, axis=1)  # run별 정답 머신

    Q = np.zeros((runs, k))   # 머신별 추정 가치
    N = np.zeros((runs, k))   # 머신별 당긴 횟수
    avg_reward = np.zeros(steps)
    pct_optimal = np.zeros(steps)
    idx = np.arange(runs)

    for t in range(steps):
        # ε-greedy 행동 선택 (run별로 동시에)
        explore = rng.random(runs) < epsilon
        greedy_a = np.argmax(Q, axis=1)
        random_a = rng.integers(0, k, size=runs)
        a = np.where(explore, random_a, greedy_a)

        # 보상 받기: 진짜 평균 주변의 잡음 섞인 값
        rewards = rng.normal(q_true[idx, a], 1.0)

        # 추정 가치 갱신 (표본 평균을 점진적으로)
        N[idx, a] += 1
        Q[idx, a] += (rewards - Q[idx, a]) / N[idx, a]

        avg_reward[t] = rewards.mean()
        pct_optimal[t] = np.mean(a == optimal_arm)

    return avg_reward, pct_optimal


def main():
    epsilons = [0.0, 0.01, 0.1, 0.3]
    colors = ["#8172B3", "#CCB974", "#4C72B0", "#C44E52"]

    print("=" * 60)
    print(" 멀티암드 밴딧: 탐험율(ε)에 따른 성능 (머신 10대, 1000번)")
    print("=" * 60)

    results = {}
    for eps in epsilons:
        avg_reward, pct_optimal = run_bandit(eps)
        results[eps] = (avg_reward, pct_optimal)
        print(f"  ε={eps:<5} -> 마지막 평균보상 {avg_reward[-100:].mean():.3f}, "
              f"최적 선택 비율 {pct_optimal[-100:].mean()*100:.1f}%")

    # 시각화: (왼쪽) 평균 보상, (오른쪽) 최적 머신 선택 비율 ----------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for eps, color in zip(epsilons, colors):
        avg_reward, pct_optimal = results[eps]
        axes[0].plot(avg_reward, color=color, label=f"ε={eps}", linewidth=1.5)
        axes[1].plot(pct_optimal * 100, color=color, label=f"ε={eps}", linewidth=1.5)

    axes[0].set_title("Average Reward over Time")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Average Reward")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("% Optimal Action over Time")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("% Optimal Action")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    save_path = os.path.join(OUTPUT_DIR, "09_multi_armed_bandit.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 그래프 저장됨 -> {save_path}")
    print("\n  💡 해석:")
    print("     - ε=0 (탐험 안 함): 초반에 잘못 찍으면 그 머신에 갇혀 성능이 낮다.")
    print("     - ε=0.1 (적당한 탐험): 좋은 머신을 잘 찾아 최적 선택 비율이 가장 높다.")
    print("     - ε=0.3 (과한 탐험): 좋은 걸 알면서도 자주 딴 데를 당겨 평균 보상이 낮다.")


if __name__ == "__main__":
    main()

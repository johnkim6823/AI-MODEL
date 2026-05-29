"""
[심층강화학습] DQN vs Double DQN (CartPole)
=============================================

가치 기반 DRL 두 가지를 같은 환경(CartPole, 막대 균형 잡기)에서 비교한다.
  - DQN  : 신경망으로 Q(s,a)를 근사 (리플레이 버퍼 + 타깃 네트워크)
  - DDQN : Q값 과대평가를 줄인 개선판

그래프: 에피소드별 총 보상(=버틴 스텝 수, 최대 200)이 학습으로 오르는 모습.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from drl_nn import moving_average
from dqn import DQNAgent, train
from envs import CartPole

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def main():
    n_episodes = 300
    print("=" * 55)
    print(" DQN vs Double DQN (CartPole, 최대 200점)")
    print("=" * 55)

    results = {}
    for name, double, color in [("DQN", False, "#4C72B0"), ("Double DQN", True, "#C44E52")]:
        env = CartPole(max_steps=200, seed=0)
        agent = DQNAgent(env.obs_dim, env.n_actions, double=double, seed=0)
        rewards = train(env, agent, n_episodes=n_episodes)
        results[name] = (rewards, color)
        print(f"  {name:11s}: 초기 평균 {np.mean(rewards[:20]):.1f} -> "
              f"마지막 평균 {np.mean(rewards[-20:]):.1f} (최대 {max(rewards):.0f})")

    plt.figure(figsize=(10, 6))
    for name, (rewards, color) in results.items():
        plt.plot(rewards, color=color, alpha=0.25, linewidth=0.8)
        sm = moving_average(rewards, 20)
        plt.plot(range(len(sm)), sm, color=color, linewidth=2.2, label=name)
    plt.axhline(200, color="gray", linestyle="--", alpha=0.6, label="max (200)")
    plt.title("DQN vs Double DQN on CartPole")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward (steps survived)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(OUTPUT_DIR, "26_dqn_ddqn_cartpole.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 그래프 저장됨 -> {save_path}")
    print("\n  💡 해석: 두 방법 모두 학습으로 보상이 오른다. DDQN은 과대평가를 줄여")
    print("     보통 더 안정적이다(환경/시드에 따라 차이는 있음).")


if __name__ == "__main__":
    main()

"""
[심층강화학습] DDPG vs SAC (Pendulum, 연속 행동)
=================================================

연속 행동(토크) 제어 DRL 두 가지를 비교한다.
  - DDPG : 결정적 정책 + 트윈 없는 단일 크리틱(여기선 기본형). 탐험은 잡음으로.
  - SAC  : 확률적 정책 + 엔트로피 보상 + 트윈 크리틱 → 보통 더 안정적.

Pendulum 보상은 음수(비용). 무작위는 ~ -1200~-1600, 잘 세우면 0에 가까워진다.
그래프: 에피소드별 총 보상이 (아래에서 위로) 오르는 모습.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ddpg import DDPGAgent
from ddpg import train as ddpg_train
from drl_nn import moving_average
from envs import Pendulum
from sac import SACAgent
from sac import train as sac_train

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def main():
    n_episodes = 80
    print("=" * 55)
    print(" DDPG vs SAC (Pendulum, 연속 제어)")
    print("=" * 55)

    env = Pendulum(max_steps=200, seed=0)
    ddpg = DDPGAgent(env.obs_dim, env.action_dim, env.action_high, seed=0)
    r_ddpg = ddpg_train(env, ddpg, n_episodes=n_episodes)
    print(f"  DDPG: 초기 {np.mean(r_ddpg[:10]):.0f} -> 마지막 {np.mean(r_ddpg[-10:]):.0f} (최고 {max(r_ddpg):.0f})")

    env = Pendulum(max_steps=200, seed=0)
    sac = SACAgent(env.obs_dim, env.action_dim, env.action_high, seed=0)
    r_sac = sac_train(env, sac, n_episodes=n_episodes)
    print(f"  SAC : 초기 {np.mean(r_sac[:10]):.0f} -> 마지막 {np.mean(r_sac[-10:]):.0f} (최고 {max(r_sac):.0f})")

    plt.figure(figsize=(10, 6))
    for rewards, color, name in [(r_ddpg, "#4C72B0", "DDPG"), (r_sac, "#C44E52", "SAC")]:
        plt.plot(rewards, color=color, alpha=0.25, linewidth=0.9)
        sm = moving_average(rewards, 10)
        plt.plot(range(len(sm)), sm, color=color, linewidth=2.2, label=name)
    plt.title("DDPG vs SAC on Pendulum (continuous control)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward (higher = better, max ~0)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(OUTPUT_DIR, "28_ddpg_sac_pendulum.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 그래프 저장됨 -> {save_path}")
    print("\n  💡 해석: 둘 다 진자를 세우는 연속 제어를 학습한다. SAC는 엔트로피")
    print("     보상과 트윈 크리틱 덕분에 보통 더 안정적으로 수렴한다.")


if __name__ == "__main__":
    main()

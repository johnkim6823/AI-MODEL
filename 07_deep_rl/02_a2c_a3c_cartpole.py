"""
[심층강화학습] A2C vs A3C (CartPole)
=====================================

정책 기반(액터-크리틱) DRL 두 가지를 비교한다.
  - A2C : 한 줄(단일 워커)로 어드밴티지 액터-크리틱 학습
  - A3C : 여러 워커가 각자 환경에서 경험을 모아 공유망을 비동기로 갱신
          (서로 다른 경험이 섞여 탈상관 → 안정적)

그래프: 에피소드별 총 보상의 학습 곡선.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from a2c import A2CAgent
from a2c import train as a2c_train
from a3c import A3C
from drl_nn import moving_average
from envs import CartPole

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def main():
    print("=" * 55)
    print(" A2C vs A3C (CartPole, 최대 200점)")
    print("=" * 55)

    # A2C
    env = CartPole(max_steps=200, seed=0)
    a2c = A2CAgent(env.obs_dim, env.n_actions, seed=0)
    r_a2c = a2c_train(env, a2c, n_episodes=400)
    print(f"  A2C: 초기 {np.mean(r_a2c[:20]):.1f} -> 마지막 {np.mean(r_a2c[-20:]):.1f} (최대 {max(r_a2c):.0f})")

    # A3C (멀티 워커)
    a3c = A3C(4, 2, n_workers=8, n_step=20, seed=0)
    r_a3c = a3c.train(lambda wid: CartPole(max_steps=200, seed=wid), n_episodes=600)
    print(f"  A3C: 초기 {np.mean(r_a3c[:30]):.1f} -> 마지막 {np.mean(r_a3c[-30:]):.1f} (최대 {max(r_a3c):.0f})")

    plt.figure(figsize=(10, 6))
    for rewards, color, name in [(r_a2c, "#55A868", "A2C"), (r_a3c, "#8172B3", "A3C (8 workers)")]:
        plt.plot(rewards, color=color, alpha=0.2, linewidth=0.8)
        sm = moving_average(rewards, 20)
        plt.plot(range(len(sm)), sm, color=color, linewidth=2.2, label=name)
    plt.axhline(200, color="gray", linestyle="--", alpha=0.6, label="max (200)")
    plt.title("A2C vs A3C on CartPole")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(OUTPUT_DIR, "27_a2c_a3c_cartpole.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 그래프 저장됨 -> {save_path}")
    print("\n  💡 해석: 액터-크리틱은 정책을 직접 학습한다. A3C는 여러 워커의")
    print("     다양한 경험으로 더 빠르고 안정적으로 학습하는 경향이 있다.")


if __name__ == "__main__":
    main()

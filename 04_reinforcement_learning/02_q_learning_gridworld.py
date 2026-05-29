"""
[강화학습 - Q러닝] 격자 세계 길 찾기 (Q-Learning on GridWorld)
==============================================================

강화학습의 대표 알고리즘 'Q-러닝'으로 에이전트가 미로 같은 격자에서
출발점(S)에서 목표점(G)까지 가는 길을 "스스로" 배우게 한다.

▶ 어떻게 배우나?
  - 처음엔 아무것도 모르니 거의 무작위로 헤맨다(보상 낮음).
  - 목표에 도착해 +10을 받으면, 그 정보가 Q값을 통해 주변 상태로 퍼진다.
  - 에피소드를 반복할수록 "어느 칸에서 어느 방향이 좋은지"를 알게 되어
    점점 빠르게 목표에 도달한다(보상 높아짐).

▶ 결과로 보는 것
  (왼쪽) 학습 곡선: 에피소드가 지날수록 받는 총 보상이 오르는 모습
  (오른쪽) 학습된 정책: 각 칸에서 최선의 행동(화살표)과 최종 경로(S→G)
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gridworld import GridWorld, greedy_path, moving_average, q_learning

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def main():
    # 1) 환경 생성 + Q러닝 학습 ------------------------------------------
    env = GridWorld()
    alpha, gamma, epsilon, n_episodes = 0.1, 0.95, 0.1, 500

    Q, rewards, steps = q_learning(
        env, n_episodes=n_episodes, alpha=alpha, gamma=gamma,
        epsilon=epsilon, seed=0,
    )

    print("=" * 60)
    print(" Q-러닝: 격자 세계 길 찾기")
    print(f"   (학습률 α={alpha}, 할인율 γ={gamma}, 탐험율 ε={epsilon}, {n_episodes} 에피소드)")
    print("=" * 60)
    print(f"  초반 50 에피소드 평균 보상: {np.mean(rewards[:50]):7.2f}")
    print(f"  후반 50 에피소드 평균 보상: {np.mean(rewards[-50:]):7.2f}  <- 학습됨!")
    print(f"  후반 50 에피소드 평균 걸음: {np.mean(steps[-50:]):7.2f}  (짧을수록 효율적)")

    path = greedy_path(env, Q)
    print(f"  학습된 최종 경로 길이: {len(path)-1} 걸음  (이론상 최단 8걸음)")

    # 2) 시각화 -----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # (왼쪽) 학습 곡선: 에피소드별 총 보상
    axes[0].plot(rewards, color="#bcbcbc", alpha=0.6, linewidth=0.8, label="raw")
    smooth = moving_average(rewards, window=20)
    axes[0].plot(range(len(smooth)), smooth, color="#C44E52", linewidth=2.2,
                 label="smoothed (20)")
    axes[0].set_title("Learning Curve: Reward per Episode")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Reward")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # (오른쪽) 학습된 정책 + 경로
    ax = axes[1]
    # 칸마다 가치 V = 최선 Q값. 벽은 마스킹해 검게 표시.
    V = Q.max(axis=1).reshape(env.rows, env.cols).astype(float)
    mask = np.zeros_like(V, dtype=bool)
    for (wr, wc) in env.walls:
        mask[wr, wc] = True
    V_masked = np.ma.array(V, mask=mask)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("#333333")
    im = ax.imshow(V_masked, cmap=cmap, origin="upper")
    fig.colorbar(im, ax=ax, label="State Value (max Q)", fraction=0.046)

    # 각 칸에 최선 행동 화살표 표시
    for r in range(env.rows):
        for c in range(env.cols):
            if (r, c) in env.walls:
                continue
            if (r, c) == env.goal:
                ax.text(c, r, "G", ha="center", va="center",
                        fontsize=14, fontweight="bold", color="white")
                continue
            if (r, c) == env.start:
                ax.text(c, r, "S", ha="center", va="center",
                        fontsize=12, fontweight="bold", color="yellow")
            best_a = int(np.argmax(Q[env.state_id((r, c))]))
            ax.text(c, r + 0.32, env.action_arrows[best_a], ha="center",
                    va="center", fontsize=13, color="white")

    # 학습된 최종 경로(S→G)를 빨간 선으로
    ys = [p[0] for p in path]
    xs = [p[1] for p in path]
    ax.plot(xs, ys, "o-", color="red", linewidth=2, markersize=6, alpha=0.8,
            label="greedy path")
    ax.set_title("Learned Policy (arrows) and Path")
    ax.set_xticks(range(env.cols))
    ax.set_yticks(range(env.rows))
    ax.legend(loc="upper left")

    save_path = os.path.join(OUTPUT_DIR, "10_q_learning_gridworld.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 그래프 저장됨 -> {save_path}")
    print("\n  💡 해석:")
    print("     - 학습 곡선: 처음엔 헤매서 보상이 낮지만 점점 올라 안정된다.")
    print("     - 정책 그림: 각 칸의 화살표가 목표(G)를 향하도록 학습되었다.")
    print("     - 밝은 칸일수록 가치가 높다(목표에 가까움).")


if __name__ == "__main__":
    main()

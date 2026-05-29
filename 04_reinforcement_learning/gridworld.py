"""
GridWorld 환경 + Q-러닝 (강화학습 공용 모듈)
=============================================

강화학습 예제(02, 03)에서 함께 쓰는 환경과 학습 알고리즘을 모아둔 파일.
(지도학습의 model.py 와 같은 역할)

▶ GridWorld 란?
  - 격자(grid) 위에서 에이전트가 출발점(S)에서 목표점(G)까지 가는 환경.
  - 에이전트는 매 순간 상/우/하/좌 중 하나로 한 칸 움직인다.
  - 한 걸음마다 -1점(빨리 가도록 유도), 목표 도착 시 +10점.
  - 벽(#)으로는 못 지나간다. (제자리)

  지도(5x5):  S = 출발, G = 목표, # = 벽
      S  .  .  .  .
      .  .  #  .  .
      .  #  #  .  .
      .  #  .  .  .
      .  .  .  .  G

▶ 강화학습 핵심 용어
  - 상태(State)   : 에이전트의 현재 위치
  - 행동(Action)  : 상/우/하/좌
  - 보상(Reward)  : 행동의 결과로 받는 점수
  - 정책(Policy)  : 각 상태에서 어떤 행동을 할지에 대한 규칙
  - Q값(Q-value)  : "이 상태에서 이 행동을 하면 앞으로 받을 총 보상의 기대치"
"""

import numpy as np


class GridWorld:
    """5x5 격자 환경. 출발점에서 목표점까지 가는 문제."""

    def __init__(self):
        self.rows = 5
        self.cols = 5
        self.start = (0, 0)
        self.goal = (4, 4)
        self.walls = {(1, 2), (2, 1), (2, 2), (3, 1)}  # 못 지나가는 칸
        self.step_reward = -1.0    # 한 걸음마다 (빨리 도착하도록 유도)
        self.goal_reward = 10.0    # 목표 도착 보상
        self.max_steps = 100       # 한 에피소드 최대 걸음 수 (무한 방황 방지)

        # 행동: 0=상, 1=우, 2=하, 3=좌
        self.n_actions = 4
        self.action_deltas = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        self.action_arrows = ["^", ">", "v", "<"]  # 정책 시각화용 (ASCII)

        self.reset()

    @property
    def n_states(self):
        return self.rows * self.cols

    def state_id(self, pos):
        """(행, 열) 위치를 0~24 사이의 숫자 하나로 변환."""
        return pos[0] * self.cols + pos[1]

    def reset(self):
        """에이전트를 출발점으로 되돌리고 시작 상태를 반환."""
        self.agent = self.start
        self.steps = 0
        return self.state_id(self.agent)

    def step(self, action):
        """행동을 받아 (다음 상태, 보상, 종료여부)를 반환."""
        self.steps += 1
        dr, dc = self.action_deltas[action]
        r, c = self.agent
        nr, nc = r + dr, c + dc

        # 격자 안이고 벽이 아니면 이동, 아니면 제자리
        if 0 <= nr < self.rows and 0 <= nc < self.cols and (nr, nc) not in self.walls:
            self.agent = (nr, nc)

        if self.agent == self.goal:
            return self.state_id(self.agent), self.goal_reward, True
        if self.steps >= self.max_steps:
            return self.state_id(self.agent), self.step_reward, True
        return self.state_id(self.agent), self.step_reward, False


def q_learning(env, n_episodes=500, alpha=0.1, gamma=0.95, epsilon=0.1, seed=0):
    """Q-러닝으로 환경을 학습한다.

    핵심 갱신식 (벨만 방정식 기반):
        Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]

    Parameters
    ----------
    alpha   : 학습률  - Q값을 한 번에 얼마나 갱신할지 (지도학습의 학습률과 같은 역할)
    gamma   : 할인율  - 미래 보상을 얼마나 중요하게 볼지 (0~1, 클수록 멀리 봄)
    epsilon : 탐험율  - 이 확률로 무작위 행동(탐험), 아니면 최선 행동(활용)
    n_episodes : 에피소드(시도) 횟수 - 출발~종료를 몇 번 반복 학습할지

    Returns
    -------
    Q                  : 학습된 Q 테이블 (상태 x 행동)
    rewards_per_episode: 에피소드별 받은 총 보상 (학습 곡선용)
    steps_per_episode  : 에피소드별 걸린 걸음 수
    """
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n_states, env.n_actions))
    rewards_per_episode = []
    steps_per_episode = []

    for _ in range(n_episodes):
        s = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            # ε-greedy: 탐험(무작위) vs 활용(현재 최선)
            if rng.random() < epsilon:
                a = rng.integers(env.n_actions)
            else:
                # 최댓값이 여러 개면 그 중 무작위 선택 (한 방향 쏠림 방지)
                q_row = Q[s]
                a = rng.choice(np.flatnonzero(q_row == q_row.max()))

            s2, reward, done = env.step(a)

            # Q-러닝 갱신: 미래 최선 보상을 반영해 현재 Q를 조금 수정
            td_target = reward + gamma * np.max(Q[s2])
            Q[s, a] += alpha * (td_target - Q[s, a])

            s = s2
            total_reward += reward

        rewards_per_episode.append(total_reward)
        steps_per_episode.append(env.steps)

    return Q, rewards_per_episode, steps_per_episode


def greedy_path(env, Q, max_len=50):
    """학습된 Q를 따라 (탐험 없이) 출발점에서 목표까지 가는 경로를 반환."""
    pos = env.start
    path = [pos]
    visited = {pos}
    for _ in range(max_len):
        if pos == env.goal:
            break
        s = env.state_id(pos)
        a = int(np.argmax(Q[s]))
        dr, dc = env.action_deltas[a]
        nr, nc = pos[0] + dr, pos[1] + dc
        if not (0 <= nr < env.rows and 0 <= nc < env.cols) or (nr, nc) in env.walls:
            nr, nc = pos  # 벽/밖이면 제자리
        pos = (nr, nc)
        if pos in visited:   # 같은 칸 반복(무한 루프) 방지
            break
        visited.add(pos)
        path.append(pos)
    return path


def moving_average(x, window=20):
    """학습 곡선을 부드럽게 보기 위한 이동 평균."""
    x = np.asarray(x, dtype=float)
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")

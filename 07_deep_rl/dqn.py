"""
DQN / Double DQN (가치 기반 심층 강화학습)
===========================================

▶ DQN (Deep Q-Network)
  - 표 대신 '신경망'으로 Q(s,a)를 근사한다 (상태가 연속이라 표가 불가능).
  - 안정적 학습을 위한 두 핵심 장치:
      · 리플레이 버퍼: 과거 경험을 저장해 무작위로 뽑아 학습 (상관성↓)
      · 타깃 네트워크: 목표값 계산용 별도 네트워크(천천히 갱신)로 진동↓
  - 목표값:  y = r + γ · max_a' Q_target(s', a')

▶ Double DQN (DDQN)
  - DQN은 max 연산 때문에 Q값을 과대평가하는 경향이 있다.
  - DDQN은 '행동 선택'은 온라인망으로, '값 평가'는 타깃망으로 분리:
      a* = argmax_a Q_online(s', a)
      y  = r + γ · Q_target(s', a*)
  - 과대평가를 줄여 더 안정적.
"""

import numpy as np

from drl_nn import MLP, ReplayBuffer


class DQNAgent:
    def __init__(self, obs_dim, n_actions, hidden=64, lr=1e-3, gamma=0.99,
                 buffer_size=20000, batch_size=64, tau=0.01,
                 eps_start=1.0, eps_end=0.05, eps_decay=0.995, double=False, seed=0):
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.double = double
        self.rng = np.random.default_rng(seed)

        self.q = MLP([obs_dim, hidden, hidden, n_actions], lr=lr, seed=seed)
        self.q_target = MLP([obs_dim, hidden, hidden, n_actions], lr=lr, seed=seed)
        self.q_target.hard_update_from(self.q)
        self.buffer = ReplayBuffer(buffer_size, obs_dim, action_dim=1, seed=seed)

    def act(self, state, explore=True):
        if explore and self.rng.random() < self.eps:
            return int(self.rng.integers(self.n_actions))
        q = self.q.forward(state[None])[0]
        return int(np.argmax(q))

    def train_step(self):
        if self.buffer.size < self.batch_size:
            return
        s, a, r, s2, done = self.buffer.sample(self.batch_size)
        a = a.astype(int).ravel()

        # 목표값 계산 (타깃 네트워크 사용)
        q_next_target = self.q_target.forward(s2)         # (B, n_actions)
        if self.double:
            q_next_online = self.q.forward(s2)
            a_star = np.argmax(q_next_online, axis=1)
            max_q_next = q_next_target[np.arange(len(a_star)), a_star]
        else:
            max_q_next = q_next_target.max(axis=1)
        y = r.ravel() + self.gamma * (1 - done.ravel()) * max_q_next

        # 현재 Q값 예측 후 '취한 행동'에 대해서만 손실 기울기
        q_pred = self.q.forward(s)                         # (B, n_actions)
        d_out = np.zeros_like(q_pred)
        idx = np.arange(self.batch_size)
        d_out[idx, a] = (q_pred[idx, a] - y) / self.batch_size

        grads, _ = self.q.backward(d_out)
        self.q.apply_grads(grads, max_norm=10.0)
        self.q_target.soft_update_from(self.q, self.tau)

    def decay_epsilon(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)


def train(env, agent, n_episodes=300, train_per_step=1):
    """에피소드별 총 보상 리스트를 반환."""
    rewards = []
    for _ in range(n_episodes):
        s = env.reset()
        total, done = 0.0, False
        while not done:
            a = agent.act(s, explore=True)
            s2, r, done, _ = env.step(a)
            agent.buffer.add(s, a, r, s2, done)
            for _ in range(train_per_step):
                agent.train_step()
            s = s2
            total += r
        agent.decay_epsilon()
        rewards.append(total)
    return rewards

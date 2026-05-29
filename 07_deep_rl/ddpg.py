"""
DDPG (Deep Deterministic Policy Gradient, 연속 행동 제어)
==========================================================

▶ 연속 행동(예: 토크 -2~2)을 다루는 off-policy 액터-크리틱.
  - 액터 μ(s): 상태 -> '하나의 결정적 행동' (tanh로 범위 제한).
  - 크리틱 Q(s,a): 상태+행동 -> 가치.
  - 타깃 네트워크 + 리플레이 버퍼로 안정화 (DQN의 연속판).

▶ 학습
  - 크리틱:  y = r + γ·Q'(s', μ'(s')),  손실 = (Q(s,a) − y)²
  - 액터:    Q(s, μ(s)) 를 최대화 → ∇_a Q 를 액터로 흘려보낸다
             (이를 위해 신경망이 '입력에 대한 기울기'를 돌려줘야 함; drl_nn.py)
  - 탐험:    행동에 가우시안 잡음을 더한다.
"""

import numpy as np

from drl_nn import MLP, ReplayBuffer


class DDPGAgent:
    def __init__(self, obs_dim, action_dim, action_high, hidden=128,
                 actor_lr=1e-3, critic_lr=1e-3, gamma=0.99, tau=0.005,
                 buffer_size=100000, batch_size=128, noise_std=0.2, seed=0):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_high = action_high
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

        self.actor = MLP([obs_dim, hidden, hidden, action_dim], out_activation="tanh",
                         lr=actor_lr, seed=seed)
        self.actor_t = MLP([obs_dim, hidden, hidden, action_dim], out_activation="tanh",
                           lr=actor_lr, seed=seed)
        self.actor_t.hard_update_from(self.actor)

        self.critic = MLP([obs_dim + action_dim, hidden, hidden, 1], lr=critic_lr, seed=seed + 1)
        self.critic_t = MLP([obs_dim + action_dim, hidden, hidden, 1], lr=critic_lr, seed=seed + 1)
        self.critic_t.hard_update_from(self.critic)

        self.buffer = ReplayBuffer(buffer_size, obs_dim, action_dim, seed=seed)

    def act(self, state, explore=True):
        a = self.actor.forward(state[None])[0] * self.action_high
        if explore:
            a = a + self.rng.normal(0, self.noise_std * self.action_high, size=self.action_dim)
        return np.clip(a, -self.action_high, self.action_high)

    def train_step(self):
        if self.buffer.size < self.batch_size:
            return
        s, a, r, s2, done = self.buffer.sample(self.batch_size)

        # ---- 크리틱 갱신 ----
        a2 = self.actor_t.forward(s2) * self.action_high           # μ'(s')
        q_next = self.critic_t.forward(np.concatenate([s2, a2], axis=1)).ravel()
        y = r.ravel() + self.gamma * (1 - done.ravel()) * q_next

        q_pred = self.critic.forward(np.concatenate([s, a], axis=1)).ravel()
        d_q = ((q_pred - y) / self.batch_size)[:, None]
        gc, _ = self.critic.backward(d_q)
        self.critic.apply_grads(gc, max_norm=10.0)

        # ---- 액터 갱신: Q(s, μ(s)) 최대화 ----
        a_pred = self.actor.forward(s) * self.action_high          # 스케일 전 tanh 저장됨
        q = self.critic.forward(np.concatenate([s, a_pred], axis=1))
        # -Q 를 최소화 => d(-Q)/dQ = -1
        _, d_in = self.critic.backward(-np.ones((self.batch_size, 1)) / self.batch_size)
        d_action = d_in[:, self.obs_dim:]                          # ∇_a Q (행동 부분)
        # a_pred = action_high * tanh(...) 이므로 스케일을 곱해 액터로 전파
        ga, _ = self.actor.backward(d_action * self.action_high)
        self.actor.apply_grads(ga, max_norm=10.0)

        # ---- 타깃 소프트 갱신 ----
        self.actor_t.soft_update_from(self.actor, self.tau)
        self.critic_t.soft_update_from(self.critic, self.tau)


def train(env, agent, n_episodes=120, warmup=1000):
    rewards = []
    steps = 0
    for _ in range(n_episodes):
        s = env.reset()
        total, done = 0.0, False
        while not done:
            if steps < warmup:
                a = agent.rng.uniform(-agent.action_high, agent.action_high, size=agent.action_dim)
            else:
                a = agent.act(s, explore=True)
            s2, r, done, _ = env.step(a)
            agent.buffer.add(s, a, r, s2, done)
            agent.train_step()
            s = s2
            total += r
            steps += 1
        rewards.append(total)
    return rewards

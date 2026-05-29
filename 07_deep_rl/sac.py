"""
SAC (Soft Actor-Critic, 연속 행동 제어)
========================================

현대 연속제어의 대표 알고리즘. DDPG와 달리 '확률적 정책'과 '엔트로피 보상'을 쓴다.

▶ 핵심 아이디어
  - 최대 엔트로피 RL: 보상뿐 아니라 '정책의 무작위성(엔트로피)'도 함께 최대화
    → 탐험이 잘 되고 학습이 안정적.
  - 확률적 액터: 상태 -> 가우시안(평균 μ, 표준편차 σ). 행동을 샘플 후 tanh로 제한.
  - 트윈 크리틱(Q1,Q2): 둘 중 작은 값을 써서 과대평가를 억제.
  - 재매개화(reparameterization): a = tanh(μ + σ·ε),  ε~N(0,1)
    → 샘플링을 거쳐도 액터로 기울기를 흘려보낼 수 있다.

▶ 학습 목표
  - 크리틱: y = r + γ·( min(Q1',Q2')(s',a') − α·logπ(a'|s') )
  - 액터:   E[ α·logπ(a|s) − min(Q1,Q2)(s,a) ] 최소화
  - α(온도)는 고정값 사용(엔트로피 가중치).

* 재매개화/타nh 보정의 기울기를 직접 유도해 적용한다(아래 train_step 참고).
"""

import numpy as np

from drl_nn import MLP, ReplayBuffer

LOG2PI = np.log(2 * np.pi)
EPS = 1e-6


class SACAgent:
    def __init__(self, obs_dim, action_dim, action_high, hidden=128,
                 lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2,
                 buffer_size=100000, batch_size=128, seed=0):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_high = action_high
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

        # 액터: 출력 = [평균(mu), log표준편차(log_std)]
        self.actor = MLP([obs_dim, hidden, hidden, 2 * action_dim], lr=lr, seed=seed)
        # 트윈 크리틱 + 타깃
        self.q1 = MLP([obs_dim + action_dim, hidden, hidden, 1], lr=lr, seed=seed + 1)
        self.q2 = MLP([obs_dim + action_dim, hidden, hidden, 1], lr=lr, seed=seed + 2)
        self.q1_t = MLP([obs_dim + action_dim, hidden, hidden, 1], lr=lr, seed=seed + 1)
        self.q2_t = MLP([obs_dim + action_dim, hidden, hidden, 1], lr=lr, seed=seed + 2)
        self.q1_t.hard_update_from(self.q1)
        self.q2_t.hard_update_from(self.q2)
        self.buffer = ReplayBuffer(buffer_size, obs_dim, action_dim, seed=seed)

    def _policy(self, s, rng):
        """액터 forward 후 행동 샘플. (a_scaled, logπ, 캐시) 반환."""
        out = self.actor.forward(s)
        mu = out[:, :self.action_dim]
        log_std_raw = out[:, self.action_dim:]
        log_std = np.clip(log_std_raw, -20, 2)
        std = np.exp(log_std)
        eps = rng.standard_normal(mu.shape)
        a_raw = mu + std * eps
        u = np.tanh(a_raw)
        a_scaled = self.action_high * u
        # log π(a|s) = 가우시안 로그확률 − tanh 보정
        logp = (-0.5 * eps ** 2 - log_std - 0.5 * LOG2PI).sum(axis=1)
        logp -= np.log(1 - u ** 2 + EPS).sum(axis=1)
        cache = dict(mu=mu, log_std=log_std, log_std_raw=log_std_raw,
                     std=std, eps=eps, u=u)
        return a_scaled, logp, cache

    def act(self, state, explore=True):
        out = self.actor.forward(state[None])
        mu = out[:, :self.action_dim]
        if not explore:
            return np.clip(self.action_high * np.tanh(mu)[0], -self.action_high, self.action_high)
        log_std = np.clip(out[:, self.action_dim:], -20, 2)
        a_raw = mu + np.exp(log_std) * self.rng.standard_normal(mu.shape)
        return np.clip(self.action_high * np.tanh(a_raw)[0], -self.action_high, self.action_high)

    def train_step(self):
        if self.buffer.size < self.batch_size:
            return
        s, a, r, s2, done = self.buffer.sample(self.batch_size)
        N = self.batch_size

        # ---- 크리틱 갱신 ----
        a2, logp2, _ = self._policy(s2, self.rng)
        q1t = self.q1_t.forward(np.concatenate([s2, a2], axis=1)).ravel()
        q2t = self.q2_t.forward(np.concatenate([s2, a2], axis=1)).ravel()
        min_qt = np.minimum(q1t, q2t)
        y = r.ravel() + self.gamma * (1 - done.ravel()) * (min_qt - self.alpha * logp2)

        for critic in (self.q1, self.q2):
            q = critic.forward(np.concatenate([s, a], axis=1)).ravel()
            d = ((q - y) / N)[:, None]
            g, _ = critic.backward(d)
            critic.apply_grads(g, max_norm=10.0)

        # ---- 액터 갱신 (재매개화) ----
        out = self.actor.forward(s)
        mu = out[:, :self.action_dim]
        log_std_raw = out[:, self.action_dim:]
        log_std = np.clip(log_std_raw, -20, 2)
        std = np.exp(log_std)
        eps = self.rng.standard_normal(mu.shape)
        a_raw = mu + std * eps
        u = np.tanh(a_raw)
        a_pred = self.action_high * u

        # 두 크리틱의 ∇_a Q 중 '작은 Q' 쪽을 사용
        q1p = self.q1.forward(np.concatenate([s, a_pred], axis=1))
        _, din1 = self.q1.backward(np.ones((N, 1)))
        q2p = self.q2.forward(np.concatenate([s, a_pred], axis=1))
        _, din2 = self.q2.backward(np.ones((N, 1)))
        use1 = (q1p.ravel() < q2p.ravel())[:, None]
        dQ_da_scaled = np.where(use1, din1[:, self.obs_dim:], din2[:, self.obs_dim:])
        dQ_da_raw = dQ_da_scaled * self.action_high * (1 - u ** 2)

        # 손실 L = α·logπ − Q.  유도한 기울기:
        #   d logπ/dμ = 2u,   d logπ/dlog_std = 2u·σ·ε − 1
        d_mu = (self.alpha * (2 * u) - dQ_da_raw) / N
        d_log_std = (self.alpha * (2 * u * std * eps - 1) - dQ_da_raw * std * eps) / N
        d_log_std *= ((log_std_raw > -20) & (log_std_raw < 2))   # 클램프된 곳은 기울기 0
        d_out = np.concatenate([d_mu, d_log_std], axis=1)
        ga, _ = self.actor.backward(d_out)
        self.actor.apply_grads(ga, max_norm=10.0)

        # ---- 타깃 소프트 갱신 ----
        self.q1_t.soft_update_from(self.q1, self.tau)
        self.q2_t.soft_update_from(self.q2, self.tau)


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

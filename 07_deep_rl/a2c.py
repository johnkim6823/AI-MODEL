"""
A2C (Advantage Actor-Critic, 정책 기반 심층 강화학습)
======================================================

▶ 액터-크리틱 구조
  - 액터(정책망): 상태 -> 행동 확률 π(a|s) (softmax). '무엇을 할지' 결정.
  - 크리틱(가치망): 상태 -> V(s). 그 상태가 '얼마나 좋은지' 평가.

▶ 어드밴티지(advantage)
  - A(s,a) = G_t − V(s_t)   (실제로 받은 이득이 기대보다 얼마나 나았나)
  - 정책 갱신: A가 양수인 행동의 확률을 높이고, 음수면 낮춘다.
      액터 손실:  −Σ logπ(a_t|s_t) · A_t   (− 엔트로피 보너스로 탐험 유지)
      크리틱 손실: Σ (V(s_t) − G_t)²

이 구현은 한 에피소드를 모은 뒤(몬테카를로 리턴) 갱신하는 단순한 형태다.
"""

import numpy as np

from drl_nn import MLP, softmax


class A2CAgent:
    def __init__(self, obs_dim, n_actions, hidden=64, lr=2e-3, gamma=0.99,
                 entropy_coef=0.01, seed=0):
        self.n_actions = n_actions
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.rng = np.random.default_rng(seed)
        self.actor = MLP([obs_dim, hidden, hidden, n_actions], lr=lr, seed=seed)
        self.critic = MLP([obs_dim, hidden, hidden, 1], lr=lr, seed=seed + 1)

    def act(self, state, explore=True):
        logits = self.actor.forward(state[None])
        p = softmax(logits)[0]
        if explore:
            return int(self.rng.choice(self.n_actions, p=p))
        return int(np.argmax(p))

    def _returns(self, rewards):
        G = np.zeros(len(rewards))
        running = 0.0
        for t in reversed(range(len(rewards))):
            running = rewards[t] + self.gamma * running
            G[t] = running
        return G

    def update(self, states, actions, rewards):
        S = np.array(states)
        A = np.array(actions, dtype=int)
        G = self._returns(rewards)
        N = len(A)

        # 크리틱: V(s) 가 리턴 G 를 맞추도록
        V = self.critic.forward(S).ravel()
        adv = G - V
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)   # 어드밴티지 표준화(안정화)

        # 액터: 정책 경사 + 엔트로피 보너스
        logits = self.actor.forward(S)
        p = softmax(logits)
        onehot = np.zeros_like(p)
        onehot[np.arange(N), A] = 1
        d_logits = (p - onehot) * adv[:, None]          # 정책 경사
        logp = np.log(p + 1e-12)
        H = -(p * logp).sum(axis=1, keepdims=True)      # 행별 엔트로피
        d_logits += self.entropy_coef * p * (logp + H)  # 엔트로피 보너스(탐험 유지)
        d_logits /= N
        ga, _ = self.actor.backward(d_logits)
        self.actor.apply_grads(ga, max_norm=5.0)

        # 크리틱 갱신 (MSE)
        d_value = ((V - G) / N)[:, None]
        gc, _ = self.critic.backward(d_value)
        self.critic.apply_grads(gc, max_norm=5.0)


def train(env, agent, n_episodes=400):
    rewards_hist = []
    for _ in range(n_episodes):
        s = env.reset()
        states, actions, rewards = [], [], []
        done = False
        while not done:
            a = agent.act(s, explore=True)
            s2, r, done, _ = env.step(a)
            states.append(s); actions.append(a); rewards.append(r)
            s = s2
        agent.update(states, actions, rewards)
        rewards_hist.append(sum(rewards))
    return rewards_hist

"""
A3C (Asynchronous Advantage Actor-Critic)
==========================================

▶ A2C 와의 차이 = '여러 워커 + 비동기 갱신'
  - 여러 워커(worker)가 '각자의 환경'에서 경험을 모은다(서로 다른 경험 → 탈상관).
  - 각 워커는 공유(global) 네트워크를 사본으로 떠서 기울기를 계산한 뒤,
    준비되는 대로 '공유 네트워크에 자기 기울기를 반영'한다(비동기 갱신).

▶ 구현 메모 (안정적 재현을 위해 단일 프로세스로)
  - 진짜 A3C는 OS 스레드/프로세스로 워커를 '동시에' 돌리지만, 파이썬 GIL과
    공유상태 경쟁 때문에 재현이 불안정하다. 여기서는 워커들을 한 프로세스에서
    '번갈아(round-robin)' 돌려 각자 자기 기울기를 공유망에 차례로 반영한다.
    → A3C의 핵심(여러 환경의 탈상관 경험 + 워커별 비동기 갱신)을 안정적으로 재현.
"""

import numpy as np

from drl_nn import MLP, softmax


class A3C:
    def __init__(self, obs_dim, n_actions, hidden=64, lr=7e-4, gamma=0.99,
                 entropy_coef=0.01, n_workers=8, n_step=20, seed=0):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden = hidden
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.n_workers = n_workers
        self.n_step = n_step
        self.seed = seed
        self.g_actor = MLP([obs_dim, hidden, hidden, n_actions], lr=lr, seed=seed)
        self.g_critic = MLP([obs_dim, hidden, hidden, 1], lr=lr, seed=seed + 1)

    def _new_local(self, wid):
        return (MLP([self.obs_dim, self.hidden, self.hidden, self.n_actions], seed=wid),
                MLP([self.obs_dim, self.hidden, self.hidden, 1], seed=wid + 100))

    def _worker_update(self, w):
        """워커 w가 n_step 경험을 모아 자기 기울기를 공유망에 반영. (끝난 에피소드 보상 리스트 반환)"""
        l_actor, l_critic = w["actor"], w["critic"]
        l_actor.hard_update_from(self.g_actor)        # 공유망 -> 로컬 동기화
        l_critic.hard_update_from(self.g_critic)

        env, s, rng = w["env"], w["s"], w["rng"]
        states, actions, rewards = [], [], []
        done = False
        finished = []
        for _ in range(self.n_step):
            p = softmax(l_actor.forward(s[None]))[0]
            a = int(rng.choice(self.n_actions, p=p))
            s2, r, done, _ = env.step(a)
            states.append(s); actions.append(a); rewards.append(r)
            w["ep_reward"] += r
            s = s2
            if done:
                finished.append(w["ep_reward"])
                w["ep_reward"] = 0.0
                s = env.reset()
                break
        w["s"] = s

        # n-step 리턴 (끝이 아니면 가치로 부트스트랩)
        R = 0.0 if done else float(l_critic.forward(s[None])[0, 0])
        G = np.zeros(len(rewards))
        for t in reversed(range(len(rewards))):
            R = rewards[t] + self.gamma * R
            G[t] = R

        S = np.array(states); A = np.array(actions, dtype=int); N = len(A)
        V = l_critic.forward(S).ravel()
        adv = G - V
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        logits = l_actor.forward(S)
        p = softmax(logits)
        onehot = np.zeros_like(p); onehot[np.arange(N), A] = 1
        d_logits = (p - onehot) * adv[:, None]
        logp = np.log(p + 1e-12)
        H = -(p * logp).sum(axis=1, keepdims=True)
        d_logits += self.entropy_coef * p * (logp + H)
        d_logits /= N
        ga, _ = l_actor.backward(d_logits)
        d_value = ((V - G) / N)[:, None]
        gc, _ = l_critic.backward(d_value)

        # 공유망에 워커 기울기 반영 (비동기 방식)
        self.g_actor.apply_grads(ga, max_norm=5.0)
        self.g_critic.apply_grads(gc, max_norm=5.0)
        return finished

    def train(self, env_factory, n_episodes=600):
        workers = []
        for wid in range(self.n_workers):
            a, c = self._new_local(wid)
            env = env_factory(wid)
            workers.append({"env": env, "actor": a, "critic": c,
                            "s": env.reset(), "ep_reward": 0.0,
                            "rng": np.random.default_rng(1000 + wid)})
        rewards_hist = []
        while len(rewards_hist) < n_episodes:
            for w in workers:                  # 워커들이 번갈아 공유망을 갱신
                rewards_hist.extend(self._worker_update(w))
        return rewards_hist[:n_episodes]

    def act(self, state, explore=False):
        p = softmax(self.g_actor.forward(state[None]))[0]
        return int(np.argmax(p))

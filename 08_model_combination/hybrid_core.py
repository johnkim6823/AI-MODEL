"""
메타러닝 + 강화학습 하이브리드 코어 (Reptile + 정책경사)
=========================================================

▶ 아이디어 ("reptile+SAC" 같은 결합의 단순화 버전)
  - 여러 RL 태스크 분포에서 'RL 정책의 좋은 초기값'을 Reptile로 메타학습한다.
  - 그러면 새로운 RL 태스크를 만났을 때, 그 초기값에서 시작해 RL로 몇 번만
    적응하면 빠르게 잘하게 된다. (메타러닝으로 RL을 가속)

▶ 구성
  - 태스크: 컨텍스추얼 밴딧. 맥락 x∈R^d 가 주어지면 숨은 최적 행동
    a*(x)=tanh(w_task·x). 보상은 r = −(a − a*(x))²  (보상만 관측 = RL 피드백).
  - 정책: 작은 신경망 μ(x)=tanh(...) (행동 범위를 [-1,1]로 제한). a ~ N(μ(x), σ).
  - 이너(RL): REINFORCE(정책경사)로 보상을 높이도록 정책 갱신.
  - 아우터(메타): Reptile 로 정책 초기값을 적응된 정책 쪽으로 이동.

* RL 이너로는 가벼운 REINFORCE를 쓴다. 같은 아우터(Reptile)에 SAC/DDPG를
  끼울 수도 있다(원리는 동일, 구현만 무거워짐).
"""

import numpy as np


def init_params(rng, d, hidden=32):
    sizes = [d, hidden, 1]
    return [[rng.normal(0, np.sqrt(2.0 / sizes[i]), (sizes[i], sizes[i + 1])),
             np.zeros(sizes[i + 1])] for i in range(len(sizes) - 1)]


def clone(p):
    return [[W.copy(), b.copy()] for W, b in p]


def forward(params, X):
    a = X
    caches = []
    for i, (W, b) in enumerate(params):
        z = a @ W + b
        caches.append((a, z))
        a = np.maximum(0, z) if i < len(params) - 1 else np.tanh(z)   # 출력 μ∈[-1,1]
    return a, caches


def backward(params, caches, d_out):
    grads = [None] * len(params)
    ga = d_out
    last = len(params) - 1
    for i in reversed(range(len(params))):
        a_prev, z = caches[i]
        W, _ = params[i]
        if i == last:
            dz = ga * (1 - np.tanh(z) ** 2)          # tanh 출력 미분
        else:
            dz = ga * (z > 0)                         # ReLU 미분
        grads[i] = [a_prev.T @ dz, dz.sum(axis=0)]
        ga = dz @ W.T
    return grads


def _clip(grads, max_norm=5.0):
    total = np.sqrt(sum((dW ** 2).sum() + (db ** 2).sum() for dW, db in grads))
    if total > max_norm and total > 0:
        s = max_norm / total
        return [[dW * s, db * s] for dW, db in grads]
    return grads


def sgd(params, grads, lr):
    grads = _clip(grads)
    return [[W - lr * dW, b - lr * db] for (W, b), (dW, db) in zip(params, grads)]


class BanditTask:
    """컨텍스추얼 밴딧: 맥락 x -> 숨은 최적 행동 a*(x)=tanh(w·x)."""

    def __init__(self, rng, d):
        self.w = rng.normal(0, 1, (d, 1))

    def optimal(self, X):
        return np.tanh(X @ self.w)


def _policy_grad(p, task, d, lr, batch, sigma, rng):
    """한 번의 REINFORCE 갱신을 적용한 새 파라미터 반환."""
    X = rng.normal(0, 1, (batch, d))
    mu, caches = forward(p, X)
    a = mu + sigma * rng.standard_normal(mu.shape)     # 행동 샘플
    r = -((a - task.optimal(X)) ** 2)                  # 보상
    adv = (r - r.mean()) / (r.std() + 1e-8)            # 어드밴티지 표준화(안정화)
    # E[r] 최대화 → 손실 기울기 d_mu = -(adv)·(a - μ)/σ²
    d_mu = -(adv * (a - mu) / (sigma ** 2)) / batch
    return sgd(p, backward(p, caches, d_mu), lr)


def reinforce_adapt(params, task, d, steps, lr, batch, sigma, rng):
    p = clone(params)
    for _ in range(steps):
        p = _policy_grad(p, task, d, lr, batch, sigma, rng)
    return p


def eval_return(params, task, d, rng, n=500):
    """그리디 정책(μ)의 성능 = −평균제곱오차. 높을수록(0에 가까울수록) 좋음."""
    X = rng.normal(0, 1, (n, d))
    mu, _ = forward(params, X)
    return -float(np.mean((mu - task.optimal(X)) ** 2))


def train_reptile_rl(d, n_iters, inner_steps, inner_lr, meta_lr, batch, sigma,
                     hidden=32, seed=0):
    """Reptile 아우터 + REINFORCE 이너로 정책 초기값을 메타학습."""
    rng = np.random.default_rng(seed)
    theta = init_params(rng, d, hidden)
    for _ in range(n_iters):
        task = BanditTask(rng, d)
        phi = reinforce_adapt(theta, task, d, inner_steps, inner_lr, batch, sigma, rng)
        theta = [[W + meta_lr * (Wp - W), b + meta_lr * (bp - b)]
                 for (W, b), (Wp, bp) in zip(theta, phi)]
    return theta


def train_joint_rl(d, n_iters, inner_lr, batch, sigma, hidden=32, seed=0):
    """베이스라인: 메타 없이 여러 태스크를 한 정책으로 그냥 RL 학습(평균만 배움)."""
    rng = np.random.default_rng(seed)
    theta = init_params(rng, d, hidden)
    for _ in range(n_iters):
        theta = _policy_grad(theta, BanditTask(rng, d), d, inner_lr, batch, sigma, rng)
    return theta


def adaptation_curve(init_params_, test_tasks, d, max_steps, inner_lr, batch,
                     sigma, seed=0):
    """테스트 태스크들에 대해, 적응 단계 수에 따른 평균 성능(−MSE) 곡선."""
    rng = np.random.default_rng(seed)
    curve = np.zeros(max_steps + 1)
    for task in test_tasks:
        p = clone(init_params_)
        for step in range(max_steps + 1):
            curve[step] += eval_return(p, task, d, rng)
            p = reinforce_adapt(p, task, d, 1, inner_lr, batch, sigma, rng)
    return curve / len(test_tasks)

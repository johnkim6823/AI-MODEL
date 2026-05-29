"""
메타러닝 공용 모듈: 사인파 few-shot 회귀 + 함수형 MLP + 메타 알고리즘
=====================================================================

▶ 메타러닝(Meta-Learning) = "배우는 법을 배우기 (learning to learn)"
  - 보통 학습: 하나의 문제를 잘 푸는 모델을 만든다.
  - 메타러닝: '새로운 문제를 빠르게 배우는 능력' 자체를 학습한다.
    → 새 문제를 단 몇 개의 예시(few-shot)만 보고도 금방 적응!

▶ 정석 예제: 사인파 회귀 (MAML 논문)
  - 각 '태스크'는 서로 다른 사인 함수:  y = A·sin(x + φ)   (A=진폭, φ=위상)
  - 진폭 A, 위상 φ 가 태스크마다 무작위로 다르다.
  - 새 사인파의 점 K개(예: 10개)만 보고 전체 곡선을 맞히는 것이 목표.
  - 무작위 초기값에서 시작하면 10개 점만으론 어림없지만,
    '메타학습된 초기값'에서 시작하면 몇 번의 갱신만으로 잘 맞춘다.

▶ 용어
  - 태스크(Task)      : 하나의 작은 학습 문제 (하나의 사인파)
  - 서포트셋(Support) : 적응(adaptation)에 쓰는 소량의 예시 (K개)
  - 쿼리셋(Query)     : 적응 후 성능을 평가하는 예시
  - 이너 루프(Inner)  : 한 태스크에 적응하는 과정
  - 아우터 루프(Outer): 태스크들 전반에 걸쳐 '초기값'을 개선하는 과정

이 모듈은 라이브러리 없이 numpy로 작은 MLP의 순전파/역전파를 직접 구현한다.
파라미터를 명시적으로 다뤄야(이너 루프에서 복제·갱신) 하므로 '함수형' 스타일을 쓴다.
"""

import numpy as np


# ----------------------------------------------------------------------
# 1) 태스크: 무작위 사인파
# ----------------------------------------------------------------------
class SineTask:
    """하나의 사인파 회귀 태스크. y = amp * sin(x + phase)"""

    def __init__(self, rng):
        # 진폭/위상이 태스크마다 다르다. (순수 numpy로 빠르고 안정적으로 학습되도록
        #  원논문(진폭 0.1~5.0)보다 진폭 범위를 약간 좁혀 사용한다.)
        self.amp = rng.uniform(0.5, 2.0)       # 진폭
        self.phase = rng.uniform(0, 2 * np.pi)  # 위상

    def sample(self, k, rng):
        """이 사인파에서 점 k개를 무작위로 뽑는다. (x in [-5,5])"""
        x = rng.uniform(-5, 5, size=(k, 1))
        y = self.amp * np.sin(x + self.phase)
        return x, y

    def true_curve(self, xs):
        """매끈한 정답 곡선 (시각화/평가용)."""
        return self.amp * np.sin(xs + self.phase)


# ----------------------------------------------------------------------
# 2) 함수형 MLP (입력1 -> 은닉 -> 은닉 -> 출력1, ReLU, 회귀=MSE)
#    파라미터를 외부에서 주고받아 이너 루프 갱신이 쉽도록 만든다.
# ----------------------------------------------------------------------
def init_params(rng, n_hidden=40):
    """가중치 초기화. params = [[W,b], [W,b], [W,b]] (3개 층)"""
    sizes = [1, n_hidden, n_hidden, 1]
    params = []
    for i in range(len(sizes) - 1):
        W = rng.normal(0, np.sqrt(2.0 / sizes[i]), (sizes[i], sizes[i + 1]))
        b = np.zeros(sizes[i + 1])
        params.append([W, b])
    return params


def clone(params):
    """파라미터를 깊은 복사 (이너 루프에서 원본을 건드리지 않도록)."""
    return [[W.copy(), b.copy()] for W, b in params]


def moving_average(x, window=50):
    """학습 곡선을 부드럽게 보기 위한 이동 평균."""
    x = np.asarray(x, dtype=float)
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


def forward(params, x):
    """순전파. 은닉층은 ReLU, 출력층은 선형. (예측, 캐시) 반환."""
    a = x
    caches = []
    for i, (W, b) in enumerate(params):
        z = a @ W + b
        caches.append((a, z))
        a = np.maximum(0, z) if i < len(params) - 1 else z
    return a, caches


def loss_and_grad(params, x, y):
    """MSE 손실과 모든 파라미터의 기울기(역전파)를 반환."""
    pred, caches = forward(params, x)
    n = x.shape[0]
    loss = np.mean((pred - y) ** 2)

    grad_a = 2.0 * (pred - y) / n        # dL/d(출력)
    grads = [None] * len(params)
    for i in reversed(range(len(params))):
        a_prev, z = caches[i]
        W, _ = params[i]
        dz = grad_a if i == len(params) - 1 else grad_a * (z > 0)  # 출력=선형, 은닉=ReLU'
        dW = a_prev.T @ dz
        db = dz.sum(axis=0)
        grads[i] = [dW, db]
        grad_a = dz @ W.T
    return loss, grads


def clip_grads(grads, max_norm=10.0):
    """기울기 전체의 크기(norm)가 너무 크면 잘라준다 (발산 방지).

    딥러닝에서 흔히 쓰는 'gradient clipping'. 큰 진폭 태스크에서 갱신이
    폭발(overflow/nan)하는 것을 막아 학습을 안정화한다.
    """
    total = np.sqrt(sum(np.sum(dW ** 2) + np.sum(db ** 2) for dW, db in grads))
    if total > max_norm and total > 0:
        scale = max_norm / total
        return [[dW * scale, db * scale] for dW, db in grads]
    return grads


def sgd_step(params, grads, lr):
    """경사하강 한 걸음: 새 파라미터를 반환 (원본 불변)."""
    return [[W - lr * dW, b - lr * db] for (W, b), (dW, db) in zip(params, grads)]


def inner_adapt(params, x_s, y_s, inner_lr, inner_steps):
    """한 태스크의 서포트셋으로 inner_steps 번 갱신해 '적응된 파라미터'를 반환."""
    p = clone(params)
    for _ in range(inner_steps):
        _, g = loss_and_grad(p, x_s, y_s)
        p = sgd_step(p, clip_grads(g), inner_lr)
    return p


# ----------------------------------------------------------------------
# 3) 메타러닝 알고리즘들
# ----------------------------------------------------------------------
def train_maml(n_iters=3000, meta_batch=10, k_shot=10, inner_lr=0.01,
               meta_lr=0.001, inner_steps=1, n_hidden=40, seed=0):
    """MAML (1차 근사, FOMAML).

    아이디어: "한 번 적응시킨 뒤(query) 성능이 좋아지도록 '초기값'을 갱신한다."
      이너: φ = θ - inner_lr * ∇L_support(θ)
      아우터: θ ← θ - meta_lr * ∇L_query(φ)

    * 진짜 MAML은 이너 갱신을 거슬러 미분(2차 미분)한다. 여기서는 그 항을
      생략한 1차 근사(FOMAML)를 쓴다 — 구현이 단순하고 성능도 비슷하다
      (Finn et al. 2017; Nichol et al. 2018).
    """
    rng = np.random.default_rng(seed)
    theta = init_params(rng, n_hidden)
    history = []

    for _ in range(n_iters):
        # 여러 태스크의 '적응 후 쿼리 기울기'를 평균내 초기값을 갱신
        grad_sum = [[np.zeros_like(W), np.zeros_like(b)] for W, b in theta]
        batch_loss = 0.0
        for _ in range(meta_batch):
            task = SineTask(rng)
            x_s, y_s = task.sample(k_shot, rng)   # 서포트(적응용)
            x_q, y_q = task.sample(k_shot, rng)   # 쿼리(평가용)

            phi = inner_adapt(theta, x_s, y_s, inner_lr, inner_steps)
            lq, gq = loss_and_grad(phi, x_q, y_q)  # 1차 근사: φ에서의 기울기를 그대로 사용
            for gs, g in zip(grad_sum, gq):
                gs[0] += g[0]
                gs[1] += g[1]
            batch_loss += lq

        grad_avg = [[gW / meta_batch, gb / meta_batch] for gW, gb in grad_sum]
        theta = sgd_step(theta, clip_grads(grad_avg), meta_lr)
        history.append(batch_loss / meta_batch)

    return theta, history


def train_reptile(n_iters=8000, k_shot=10, inner_lr=0.02, meta_lr=0.1,
                  inner_steps=5, n_hidden=40, seed=0):
    """Reptile (Nichol et al. 2018).

    MAML보다 더 단순한 1차 메타러닝:
      1) 한 태스크에서 여러 번(SGD) 학습해 φ 를 얻는다.
      2) 초기값을 φ 쪽으로 조금 당긴다:  θ ← θ + meta_lr * (φ - θ)
    기울기를 거슬러 미분할 필요가 전혀 없어 간단하고 빠르다.
    """
    rng = np.random.default_rng(seed)
    theta = init_params(rng, n_hidden)
    history = []

    for _ in range(n_iters):
        task = SineTask(rng)
        x_s, y_s = task.sample(k_shot, rng)
        phi = inner_adapt(theta, x_s, y_s, inner_lr, inner_steps)

        # 초기값을 적응된 파라미터 쪽으로 이동
        theta = [[W + meta_lr * (Wp - W), b + meta_lr * (bp - b)]
                 for (W, b), (Wp, bp) in zip(theta, phi)]

        l, _ = loss_and_grad(phi, x_s, y_s)
        history.append(l)

    return theta, history


def train_joint(n_iters=8000, k_shot=10, lr=0.01, n_hidden=40, seed=0):
    """베이스라인: '공동 학습(joint training)'.

    메타러닝 없이 여러 태스크의 데이터를 그냥 한 모델로 학습한다.
    → 모든 사인파의 '평균'(거의 0인 평평한 곡선)을 배우게 되어,
      새 태스크에 빠르게 적응하지 못한다. (메타러닝과 대조용)
    """
    rng = np.random.default_rng(seed)
    theta = init_params(rng, n_hidden)
    history = []
    for _ in range(n_iters):
        task = SineTask(rng)
        x, y = task.sample(k_shot, rng)
        l, g = loss_and_grad(theta, x, y)
        theta = sgd_step(theta, clip_grads(g), lr)
        history.append(l)
    return theta, history


# ----------------------------------------------------------------------
# 4) 평가: 초기값에서 K-shot 적응 후 곡선 오차
# ----------------------------------------------------------------------
def adaptation_curve(init_theta, test_tasks, k_shot, inner_lr, max_steps, seed=0):
    """여러 테스트 태스크에 대해, 적응 단계 수에 따른 평균 MSE를 반환.

    반환 길이는 max_steps+1 (0번 적응 = 초기값 그대로).
    """
    rng = np.random.default_rng(seed)
    x_eval = np.linspace(-5, 5, 100).reshape(-1, 1)
    curve = np.zeros(max_steps + 1)

    for task in test_tasks:
        x_s, y_s = task.sample(k_shot, rng)
        y_true = task.true_curve(x_eval)
        p = clone(init_theta)
        for step in range(max_steps + 1):
            pred, _ = forward(p, x_eval)
            curve[step] += np.mean((pred - y_true) ** 2)
            _, g = loss_and_grad(p, x_s, y_s)
            p = sgd_step(p, clip_grads(g), inner_lr)

    return curve / len(test_tasks)

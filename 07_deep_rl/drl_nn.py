"""
DRL용 신경망 + 리플레이 버퍼 (numpy 직접 구현)
================================================

심층 강화학습(DRL)에 필요한 공통 부품:

  MLP        : 다층 퍼셉트론. 일반 역전파(파라미터 기울기)뿐 아니라
               '입력에 대한 기울기'도 돌려준다.
               → DDPG/SAC에서 ∇_a Q(s,a) 를 액터로 흘려보내기 위해 꼭 필요.
  Adam       : 각 MLP가 내부적으로 사용하는 옵티마이저.
  soft/hard update : 타깃 네트워크 갱신(폴리야크 평균).
  ReplayBuffer: 경험 (s,a,r,s',done) 저장/샘플 (off-policy 학습용).

설계 포인트
  - forward(x) 는 중간값을 캐시한다.
  - backward(d_out) 는 (파라미터 기울기들, 입력 기울기) 를 함께 반환한다.
  - apply_grads(grads) 로 따로 갱신한다 (액터 갱신 때 크리틱은 갱신하지 않으려고
    '기울기 계산'과 '갱신'을 분리).
"""

import numpy as np


def _he(rng, fan_in, fan_out):
    return rng.normal(0, np.sqrt(2.0 / fan_in), (fan_in, fan_out))


class MLP:
    """은닉층 ReLU, 출력층 선형(기본)인 다층 퍼셉트론.

    out_activation: None(선형) | 'tanh'
    """

    def __init__(self, sizes, out_activation=None, lr=1e-3, seed=0):
        rng = np.random.default_rng(seed)
        self.sizes = sizes
        self.out_activation = out_activation
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            self.W.append(_he(rng, sizes[i], sizes[i + 1]))
            self.b.append(np.zeros(sizes[i + 1]))
        self.lr = lr
        # Adam 상태
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.t = 0

    def forward(self, x):
        """x: (N, in) -> (N, out). 역전파용 중간값 저장."""
        self._a = [x]
        self._z = []
        a = x
        L = len(self.W)
        for i in range(L):
            z = a @ self.W[i] + self.b[i]
            self._z.append(z)
            if i < L - 1:
                a = np.maximum(0, z)              # 은닉층 ReLU
            else:
                a = np.tanh(z) if self.out_activation == "tanh" else z
            self._a.append(a)
        return a

    def backward(self, d_out):
        """출력 기울기 d_out -> (param_grads, d_input).

        param_grads = [(dW, db), ...].  d_input = 입력 x에 대한 기울기.
        """
        L = len(self.W)
        grads = [None] * L
        # 출력층 활성화 미분
        if self.out_activation == "tanh":
            delta = d_out * (1 - self._a[-1] ** 2)
        else:
            delta = d_out
        for i in reversed(range(L)):
            dW = self._a[i].T @ delta
            db = delta.sum(axis=0)
            grads[i] = (dW, db)
            d_in = delta @ self.W[i].T
            if i > 0:
                delta = d_in * (self._z[i - 1] > 0)   # ReLU 미분
        return grads, d_in     # 마지막 d_in = 입력층 기울기

    def apply_grads(self, grads, max_norm=None):
        """Adam으로 파라미터 갱신. max_norm 주면 그래디언트 클리핑."""
        if max_norm is not None:
            total = np.sqrt(sum((dW ** 2).sum() + (db ** 2).sum() for dW, db in grads))
            if total > max_norm and total > 0:
                scale = max_norm / total
                grads = [(dW * scale, db * scale) for dW, db in grads]
        b1, b2, eps = 0.9, 0.999, 1e-8
        self.t += 1
        for i, (dW, db) in enumerate(grads):
            self.mW[i] = b1 * self.mW[i] + (1 - b1) * dW
            self.vW[i] = b2 * self.vW[i] + (1 - b2) * dW ** 2
            self.mb[i] = b1 * self.mb[i] + (1 - b1) * db
            self.vb[i] = b2 * self.vb[i] + (1 - b2) * db ** 2
            mW = self.mW[i] / (1 - b1 ** self.t)
            vW = self.vW[i] / (1 - b2 ** self.t)
            mb = self.mb[i] / (1 - b1 ** self.t)
            vb = self.vb[i] / (1 - b2 ** self.t)
            self.W[i] -= self.lr * mW / (np.sqrt(vW) + eps)
            self.b[i] -= self.lr * mb / (np.sqrt(vb) + eps)

    # ---- 타깃 네트워크 갱신 -------------------------------------------
    def hard_update_from(self, src):
        self.W = [w.copy() for w in src.W]
        self.b = [b.copy() for b in src.b]

    def soft_update_from(self, src, tau):
        for i in range(len(self.W)):
            self.W[i] = (1 - tau) * self.W[i] + tau * src.W[i]
            self.b[i] = (1 - tau) * self.b[i] + tau * src.b[i]


def softmax(z, axis=1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


class ReplayBuffer:
    """경험 재현 버퍼: (s, a, r, s', done) 를 저장하고 무작위로 샘플."""

    def __init__(self, capacity, obs_dim, action_dim=1, seed=0):
        self.capacity = capacity
        self.s = np.zeros((capacity, obs_dim))
        self.a = np.zeros((capacity, action_dim))
        self.r = np.zeros((capacity, 1))
        self.s2 = np.zeros((capacity, obs_dim))
        self.done = np.zeros((capacity, 1))
        self.ptr = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(self, s, a, r, s2, done):
        i = self.ptr
        self.s[i] = s
        self.a[i] = a
        self.r[i] = r
        self.s2[i] = s2
        self.done[i] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = self.rng.integers(0, self.size, size=batch_size)
        return (self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.done[idx])


def moving_average(x, window=20):
    x = np.asarray(x, dtype=float)
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")

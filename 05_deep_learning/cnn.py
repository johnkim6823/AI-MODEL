"""
직접 구현한 합성곱 신경망 (CNN) — 딥러닝 공용 모듈
====================================================

▶ CNN(Convolutional Neural Network)이란?
  - 이미지처럼 '격자 구조' 데이터를 잘 다루는 신경망.
  - 작은 필터(filter/kernel)를 이미지 위로 미끄러뜨리며(합성곱, convolution)
    국소적 패턴(모서리, 곡선 등)을 추출한다.
  - 같은 필터를 이미지 전체에 공유하므로 파라미터가 적고 위치에 강하다.

▶ 이 모듈의 구성 (numpy로 직접, im2col 기법으로 벡터화)
  - conv2d        : 합성곱 연산 (순전파용 함수)
  - Conv2D        : 학습 가능한 합성곱 층 (순전파/역전파)
  - max_pool      : 최대 풀링 (2x2) — 특징맵을 절반 크기로 줄임
  - SimpleCNN     : Conv -> ReLU -> MaxPool -> Flatten -> Dense -> softmax
                    (Adam으로 학습)
"""

import numpy as np


# ----------------------------------------------------------------------
# im2col / col2im : 합성곱을 행렬곱으로 바꿔 빠르게 계산하는 표준 기법
# ----------------------------------------------------------------------
def im2col(X, kh, kw, stride=1, pad=0):
    """(N,C,H,W) 입력을 (N*OH*OW, C*kh*kw) 행렬로 펼친다."""
    N, C, H, W = X.shape
    OH = (H + 2 * pad - kh) // stride + 1
    OW = (W + 2 * pad - kw) // stride + 1
    Xp = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    cols = np.zeros((N, C, kh, kw, OH, OW))
    for y in range(kh):
        y_end = y + stride * OH
        for x in range(kw):
            x_end = x + stride * OW
            cols[:, :, y, x, :, :] = Xp[:, :, y:y_end:stride, x:x_end:stride]
    cols = cols.transpose(0, 4, 5, 1, 2, 3).reshape(N * OH * OW, -1)
    return cols, OH, OW


def col2im(cols, x_shape, kh, kw, stride=1, pad=0):
    """im2col의 역연산: (N*OH*OW, C*kh*kw) -> (N,C,H,W). 역전파에 사용."""
    N, C, H, W = x_shape
    OH = (H + 2 * pad - kh) // stride + 1
    OW = (W + 2 * pad - kw) // stride + 1
    cols = cols.reshape(N, OH, OW, C, kh, kw).transpose(0, 3, 4, 5, 1, 2)
    Xp = np.zeros((N, C, H + 2 * pad, W + 2 * pad))
    for y in range(kh):
        y_end = y + stride * OH
        for x in range(kw):
            x_end = x + stride * OW
            Xp[:, :, y:y_end:stride, x:x_end:stride] += cols[:, :, y, x, :, :]
    return Xp if pad == 0 else Xp[:, :, pad:-pad, pad:-pad]


def conv2d(X, W, b, stride=1, pad=0):
    """합성곱 순전파(함수형). X:(N,C,H,W), W:(F,C,kh,kw), b:(F,) -> (N,F,OH,OW)"""
    F, C, kh, kw = W.shape
    N = X.shape[0]
    cols, OH, OW = im2col(X, kh, kw, stride, pad)
    out = cols @ W.reshape(F, -1).T + b
    return out.reshape(N, OH, OW, F).transpose(0, 3, 1, 2)


def max_pool(X, size=2, stride=2):
    """2x2 최대 풀링 순전파. (크기가 size로 나눠떨어진다고 가정)"""
    N, C, H, W = X.shape
    OH, OW = H // stride, W // stride
    Xr = X.reshape(N, C, OH, stride, OW, stride)
    out = Xr.max(axis=(3, 5))
    return out


# ----------------------------------------------------------------------
# 학습 가능한 층
# ----------------------------------------------------------------------
def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class SimpleCNN:
    """작은 CNN:  Conv(F개 필터) -> ReLU -> MaxPool(2x2) -> Flatten -> Dense -> softmax

    8x8 손글씨 숫자(1채널)용으로 설계. Adam으로 학습.
    """

    def __init__(self, n_filters=8, kernel=3, n_classes=10,
                 img_size=8, learning_rate=0.01, seed=0):
        rng = np.random.default_rng(seed)
        self.F, self.K = n_filters, kernel
        self.lr = learning_rate

        # 합성곱 필터 (F,1,K,K) — He 초기화
        self.Wc = rng.normal(0, np.sqrt(2.0 / (kernel * kernel)),
                             (n_filters, 1, kernel, kernel))
        self.bc = np.zeros(n_filters)

        conv_out = img_size - kernel + 1          # 8-3+1 = 6
        pool_out = conv_out // 2                   # 6//2 = 3
        self.flat_dim = n_filters * pool_out * pool_out
        self.pool_out = pool_out

        # 완전연결(Dense) 층 — He 초기화
        self.Wd = rng.normal(0, np.sqrt(2.0 / self.flat_dim), (self.flat_dim, n_classes))
        self.bd = np.zeros(n_classes)

        # Adam 상태
        self._init_adam()
        self.loss_history_, self.val_acc_history_ = [], []

    def _init_adam(self):
        self._state = {}
        for name, p in [("Wc", self.Wc), ("bc", self.bc), ("Wd", self.Wd), ("bd", self.bd)]:
            self._state[name] = [np.zeros_like(p), np.zeros_like(p)]  # m, v
        self._t = 0

    def _adam(self, name, param, grad):
        b1, b2, eps = 0.9, 0.999, 1e-8
        m, v = self._state[name]
        m[:] = b1 * m + (1 - b1) * grad
        v[:] = b2 * v + (1 - b2) * grad ** 2
        m_hat = m / (1 - b1 ** self._t)
        v_hat = v / (1 - b2 ** self._t)
        param -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

    # ---- 순전파 (역전파용 캐시 저장) -----------------------------------
    def forward(self, X):
        N = X.shape[0]
        # 1) 합성곱 (im2col 사용)
        cols, OH, OW = im2col(X, self.K, self.K)
        conv = (cols @ self.Wc.reshape(self.F, -1).T + self.bc)  # (N*OH*OW, F)
        conv = conv.reshape(N, OH, OW, self.F).transpose(0, 3, 1, 2)  # (N,F,OH,OW)
        # 2) ReLU
        relu = np.maximum(0, conv)
        # 3) MaxPool 2x2 (위치 기록은 마스크로)
        P = 2
        PH, PW = OH // P, OW // P
        relu_r = relu.reshape(N, self.F, PH, P, PW, P)
        pooled = relu_r.max(axis=(3, 5))                         # (N,F,PH,PW)
        # 4) Flatten -> Dense -> softmax
        flat = pooled.reshape(N, -1)
        logits = flat @ self.Wd + self.bd
        probs = _softmax(logits)

        self._cache = (X, cols, OH, OW, conv, relu, relu_r, pooled, flat, probs, N, PH, PW, P)
        return probs

    # ---- 역전파 + 갱신 -------------------------------------------------
    def backward(self, y):
        (X, cols, OH, OW, conv, relu, relu_r, pooled, flat, probs,
         N, PH, PW, P) = self._cache

        Y = np.zeros_like(probs)
        Y[np.arange(N), y] = 1
        dlogits = (probs - Y) / N                # softmax+CE

        # Dense 역전파
        dWd = flat.T @ dlogits
        dbd = dlogits.sum(axis=0)
        dflat = dlogits @ self.Wd.T
        dpooled = dflat.reshape(N, self.F, PH, PW)

        # MaxPool 역전파: 최댓값 위치로만 기울기 전달
        pooled_exp = pooled[:, :, :, None, :, None]
        mask = (relu_r == pooled_exp)
        # 동점일 경우 균등 분배 (안전)
        mask = mask / mask.sum(axis=(3, 5), keepdims=True)
        drelu = (mask * dpooled[:, :, :, None, :, None]).reshape(N, self.F, OH, OW)

        # ReLU 역전파
        dconv = drelu * (conv > 0)

        # Conv 역전파
        dconv_r = dconv.transpose(0, 2, 3, 1).reshape(-1, self.F)  # (N*OH*OW, F)
        dWc = (dconv_r.T @ cols).reshape(self.Wc.shape)
        dbc = dconv_r.sum(axis=0)

        # Adam 갱신
        self._t += 1
        self._adam("Wc", self.Wc, dWc)
        self._adam("bc", self.bc, dbc)
        self._adam("Wd", self.Wd, dWd)
        self._adam("bd", self.bd, dbd)

    # ---- 학습 / 추론 ---------------------------------------------------
    def fit(self, X, y, n_epochs=20, batch_size=64, X_val=None, y_val=None, seed=0):
        rng = np.random.default_rng(seed)
        N = X.shape[0]
        self.loss_history_, self.val_acc_history_ = [], []
        for _ in range(n_epochs):
            idx = rng.permutation(N)
            for s in range(0, N, batch_size):
                bi = idx[s:s + batch_size]
                probs = self.forward(X[bi])
                self.backward(y[bi])
            probs_all = self.forward(X)
            loss = -np.mean(np.log(probs_all[np.arange(N), y] + 1e-12))
            self.loss_history_.append(loss)
            if X_val is not None:
                self.val_acc_history_.append(self.score(X_val, y_val))
        return self

    def predict(self, X):
        return self.forward(X).argmax(axis=1)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

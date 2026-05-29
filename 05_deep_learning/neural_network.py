"""
직접 구현한 다층 신경망 (Multi-Layer Perceptron, MLP) — 딥러닝 공용 모듈
========================================================================

딥러닝의 핵심은 "여러 층(layer)을 쌓은 신경망을 역전파(backprop)로 학습"하는 것.
라이브러리 없이 numpy로 직접 구현해 그 원리를 그대로 본다.
(지도학습의 model.py, 강화학습의 gridworld.py 와 같은 역할)

▶ 신경망이란?
  - 입력 -> [은닉층들] -> 출력 로 이어지는 함수.
  - 각 층: z = (입력 · 가중치 W) + 편향 b,  그 다음 활성화 함수 적용.
  - 활성화 함수(ReLU/tanh 등)의 '비선형성' 덕분에 직선으로 못 나누는
    복잡한 패턴(예: XOR)도 학습할 수 있다.

▶ 학습 방법: 역전파 + 경사하강법
  - 순전파(forward): 입력을 넣어 예측을 계산.
  - 손실(loss): 예측이 정답과 얼마나 다른지 (분류는 교차 엔트로피).
  - 역전파(backward): 손실의 기울기를 출력에서 입력 쪽으로 거꾸로 전파해
    모든 가중치의 기울기를 구한다.
  - 갱신: 가중치 <- 가중치 - 학습률 * 기울기.
"""

import numpy as np


def _activation(name):
    """활성화 함수와 그 도함수를 (함수, 도함수) 쌍으로 반환."""
    if name == "relu":
        return (lambda z: np.maximum(0, z),
                lambda z: (z > 0).astype(z.dtype))
    if name == "tanh":
        return (np.tanh,
                lambda z: 1.0 - np.tanh(z) ** 2)
    if name == "sigmoid":
        sig = lambda z: 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        return (sig,
                lambda z: sig(z) * (1 - sig(z)))
    raise ValueError(f"알 수 없는 활성화 함수: {name}")


def _softmax(z):
    """출력을 확률(합=1)로 변환. (수치 안정성을 위해 최댓값을 빼줌)"""
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class MLPClassifier:
    """은닉층을 쌓은 분류용 신경망 (softmax + 교차 엔트로피).

    Parameters
    ----------
    hidden_layers : tuple
        은닉층 뉴런 수. 예) (16, 16) = 16개짜리 은닉층 2개. () = 은닉층 없음(=선형 분류).
    activation : 'relu' | 'tanh' | 'sigmoid'
    learning_rate : 경사하강법 보폭
    n_epochs : 전체 데이터 반복 학습 횟수
    batch_size : 미니배치 크기 (None이면 전체 배치)
    """

    def __init__(self, hidden_layers=(16, 16), activation="relu",
                 learning_rate=0.1, n_epochs=300, batch_size=None, seed=0):
        self.hidden_layers = tuple(hidden_layers)
        self.activation = activation
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.seed = seed
        self.loss_history_ = []
        self.acc_history_ = []

    def _init_params(self, n_in, n_out):
        rng = np.random.default_rng(self.seed)
        sizes = [n_in] + list(self.hidden_layers) + [n_out]
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            fan_in = sizes[i]
            # ReLU에는 He 초기화, 그 외에는 Xavier 초기화 (학습 안정화)
            scale = np.sqrt(2.0 / fan_in) if self.activation == "relu" else np.sqrt(1.0 / fan_in)
            self.W.append(rng.normal(0, scale, (sizes[i], sizes[i + 1])))
            self.b.append(np.zeros(sizes[i + 1]))

    def _forward(self, X):
        """순전파. 역전파에 필요한 중간값(z, a)을 저장하며 출력 확률을 반환."""
        act, _ = _activation(self.activation)
        self._zs, self._as = [], [X]
        a = X
        for i in range(len(self.W)):
            z = a @ self.W[i] + self.b[i]
            self._zs.append(z)
            if i < len(self.W) - 1:
                a = act(z)            # 은닉층: 활성화 함수
            else:
                a = _softmax(z)       # 출력층: softmax
            self._as.append(a)
        return a

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        n_samples, n_features = X.shape
        n_classes = int(y.max()) + 1
        self.n_classes_ = n_classes
        self._init_params(n_features, n_classes)

        # 정답을 원-핫 벡터로 (예: 2 -> [0,0,1,0])
        Y = np.zeros((n_samples, n_classes))
        Y[np.arange(n_samples), y] = 1

        _, act_deriv = _activation(self.activation)
        rng = np.random.default_rng(self.seed + 1)
        batch = self.batch_size or n_samples

        self.loss_history_, self.acc_history_ = [], []
        for _ in range(self.n_epochs):
            idx = rng.permutation(n_samples)
            for start in range(0, n_samples, batch):
                bi = idx[start:start + batch]
                xb, yb = X[bi], Y[bi]
                probs = self._forward(xb)

                # 역전파: 출력층 기울기부터 시작
                delta = (probs - yb) / len(bi)   # softmax+CE 의 깔끔한 기울기
                for i in reversed(range(len(self.W))):
                    dW = self._as[i].T @ delta
                    db = delta.sum(axis=0)
                    if i > 0:  # 이전 층으로 기울기 전파
                        delta = (delta @ self.W[i].T) * act_deriv(self._zs[i - 1])
                    self.W[i] -= self.learning_rate * dW
                    self.b[i] -= self.learning_rate * db

            # 에폭마다 전체 손실/정확도 기록 (학습 곡선용)
            probs_all = self._forward(X)
            eps = 1e-12
            loss = -np.mean(np.log(probs_all[np.arange(n_samples), y] + eps))
            self.loss_history_.append(loss)
            self.acc_history_.append(np.mean(probs_all.argmax(axis=1) == y))
        return self

    def predict_proba(self, X):
        return self._forward(np.asarray(X, dtype=float))

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def score(self, X, y):
        return np.mean(self.predict(X) == np.asarray(y).astype(int))

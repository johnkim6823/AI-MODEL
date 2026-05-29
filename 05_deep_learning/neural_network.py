"""
직접 구현한 다층 신경망 (Multi-Layer Perceptron, MLP) — 딥러닝 공용 모듈
========================================================================

딥러닝의 핵심은 "여러 층(layer)을 쌓은 신경망을 역전파(backprop)로 학습"하는 것.
라이브러리 없이 numpy로 직접 구현해 그 원리를 그대로 본다.

이 모듈은 다음을 지원한다(뒤에 추가한 고급 기능 포함):
  - 활성화 함수: relu / tanh / sigmoid
  - 정규화(regularization): L2 가중치 감쇠, 드롭아웃(dropout)  → 과적합 억제
  - 옵티마이저(optimizer): sgd / momentum / adam            → 학습 속도/안정성

▶ 학습 방법: 역전파 + 경사하강법(또는 그 개선판)
  - 순전파(forward): 입력 -> 예측
  - 손실(loss): 교차 엔트로피 (+ L2 페널티)
  - 역전파(backward): 출력에서 입력 쪽으로 기울기 전파
  - 갱신(update): 옵티마이저가 기울기로 가중치를 수정
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
        은닉층 뉴런 수. 예) (16, 16). () = 은닉층 없음(=선형 분류).
    activation : 'relu' | 'tanh' | 'sigmoid'
    learning_rate : 갱신 보폭
    n_epochs : 전체 데이터 반복 학습 횟수
    batch_size : 미니배치 크기 (None이면 전체 배치)
    l2 : L2 정규화 계수 (가중치가 너무 커지지 않게 눌러 과적합 억제). 0이면 미사용.
    dropout : 학습 중 은닉 뉴런을 무작위로 끄는 비율 (0~1). 과적합 억제. 0이면 미사용.
    optimizer : 'sgd' | 'momentum' | 'adam'
    """

    def __init__(self, hidden_layers=(16, 16), activation="relu",
                 learning_rate=0.1, n_epochs=300, batch_size=None,
                 l2=0.0, dropout=0.0, optimizer="sgd", seed=0):
        self.hidden_layers = tuple(hidden_layers)
        self.activation = activation
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.dropout = dropout
        self.optimizer = optimizer
        self.seed = seed
        self.loss_history_ = []
        self.acc_history_ = []

    # ---- 초기화 --------------------------------------------------------
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
        # 옵티마이저 상태 초기화
        self._mW = [np.zeros_like(W) for W in self.W]
        self._vW = [np.zeros_like(W) for W in self.W]
        self._mb = [np.zeros_like(b) for b in self.b]
        self._vb = [np.zeros_like(b) for b in self.b]
        self._t = 0

    # ---- 순전파 --------------------------------------------------------
    def _forward(self, X, training=False):
        """순전파. 역전파에 필요한 중간값을 저장하며 출력 확률을 반환.

        training=True 이면 드롭아웃을 적용(역드롭아웃: 켜진 뉴런을 1/(1-p)로 보정).
        """
        act, _ = _activation(self.activation)
        self._zs, self._as, self._masks = [], [X], {}
        a = X
        for i in range(len(self.W)):
            z = a @ self.W[i] + self.b[i]
            self._zs.append(z)
            if i < len(self.W) - 1:
                a = act(z)                       # 은닉층: 활성화 함수
                if training and self.dropout > 0:
                    mask = (self._drop_rng.random(a.shape) > self.dropout) / (1 - self.dropout)
                    a = a * mask
                    self._masks[i] = mask
            else:
                a = _softmax(z)                  # 출력층: softmax
            self._as.append(a)
        return a

    # ---- 옵티마이저 한 걸음 --------------------------------------------
    def _update(self, i, dW, db):
        lr = self.learning_rate
        if self.optimizer == "sgd":
            self.W[i] -= lr * dW
            self.b[i] -= lr * db
        elif self.optimizer == "momentum":
            beta = 0.9
            self._mW[i] = beta * self._mW[i] + dW
            self._mb[i] = beta * self._mb[i] + db
            self.W[i] -= lr * self._mW[i]
            self.b[i] -= lr * self._mb[i]
        elif self.optimizer == "adam":
            b1, b2, eps = 0.9, 0.999, 1e-8
            self._mW[i] = b1 * self._mW[i] + (1 - b1) * dW
            self._vW[i] = b2 * self._vW[i] + (1 - b2) * dW ** 2
            self._mb[i] = b1 * self._mb[i] + (1 - b1) * db
            self._vb[i] = b2 * self._vb[i] + (1 - b2) * db ** 2
            mW_hat = self._mW[i] / (1 - b1 ** self._t)
            vW_hat = self._vW[i] / (1 - b2 ** self._t)
            mb_hat = self._mb[i] / (1 - b1 ** self._t)
            vb_hat = self._vb[i] / (1 - b2 ** self._t)
            self.W[i] -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
            self.b[i] -= lr * mb_hat / (np.sqrt(vb_hat) + eps)
        else:
            raise ValueError(f"알 수 없는 옵티마이저: {self.optimizer}")

    # ---- 학습 ----------------------------------------------------------
    def fit(self, X, y, X_val=None, y_val=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        n_samples, n_features = X.shape
        n_classes = int(y.max()) + 1
        self.n_classes_ = n_classes
        self._init_params(n_features, n_classes)
        self._drop_rng = np.random.default_rng(self.seed + 2)

        Y = np.zeros((n_samples, n_classes))   # 원-핫 정답
        Y[np.arange(n_samples), y] = 1

        _, act_deriv = _activation(self.activation)
        rng = np.random.default_rng(self.seed + 1)
        batch = self.batch_size or n_samples

        self.loss_history_, self.acc_history_ = [], []
        self.val_acc_history_ = []
        for _ in range(self.n_epochs):
            idx = rng.permutation(n_samples)
            for start in range(0, n_samples, batch):
                bi = idx[start:start + batch]
                xb, yb = X[bi], Y[bi]
                probs = self._forward(xb, training=True)
                self._t += 1

                # 역전파: 출력층 기울기부터
                delta = (probs - yb) / len(bi)
                for i in reversed(range(len(self.W))):
                    dW = self._as[i].T @ delta + self.l2 * self.W[i]  # L2 감쇠
                    db = delta.sum(axis=0)
                    if i > 0:
                        delta = delta @ self.W[i].T
                        if self.dropout > 0 and (i - 1) in self._masks:
                            delta *= self._masks[i - 1]      # 드롭아웃 마스크 전파
                        delta *= act_deriv(self._zs[i - 1])
                    self._update(i, dW, db)

            # 에폭마다 손실/정확도 기록 (드롭아웃 없이 평가)
            probs_all = self._forward(X, training=False)
            eps = 1e-12
            loss = -np.mean(np.log(probs_all[np.arange(n_samples), y] + eps))
            self.loss_history_.append(loss)
            self.acc_history_.append(np.mean(probs_all.argmax(axis=1) == y))
            if X_val is not None:
                self.val_acc_history_.append(self.score(X_val, y_val))
        return self

    # ---- 추론 ----------------------------------------------------------
    def predict_proba(self, X):
        return self._forward(np.asarray(X, dtype=float), training=False)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def score(self, X, y):
        return np.mean(self.predict(X) == np.asarray(y).astype(int))

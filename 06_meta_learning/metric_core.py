"""
메트릭 기반 메타러닝 공용 모듈 (few-shot 분류)
===============================================

▶ 메트릭 기반(metric-based) 메타러닝이란?
  - MAML/Reptile은 '좋은 초기값'을 배워 경사하강으로 적응했다(최적화 기반).
  - 메트릭 기반은 다르게 접근한다: 입력을 '임베딩 공간'으로 보내고,
    그 공간에서의 '거리/유사도'로 분류한다. 새 클래스도 예시 몇 개의
    임베딩만 있으면 (경사하강 적응 없이) 바로 분류할 수 있다.

▶ 두 대표 방법
  - Prototypical Networks: 각 클래스의 임베딩 평균(프로토타입)을 만들고,
    쿼리를 '가장 가까운 프로토타입'의 클래스로 분류 (유클리드 거리).
  - Matching Networks: 쿼리와 각 서포트 예시의 '코사인 유사도'로 가중 투표
    (가까운 예시일수록 그 예시의 라벨에 더 투표).

▶ few-shot 분류 태스크 (N-way K-shot)
  - 손글씨 숫자(8x8)에서 클래스를 나눠, 학습에 안 쓴 '새 클래스'를 몇 장(K)만
    보고 맞히게 한다. (메타학습: 학습 클래스로 '임베딩 잘 만드는 법'을 배움)
  - 학습 클래스: 0~5,  테스트(새) 클래스: 6~9

임베딩 신경망의 순전파/역전파, 두 손실의 기울기를 모두 numpy로 직접 구현한다.
"""

import numpy as np
from sklearn.datasets import load_digits


# ----------------------------------------------------------------------
# 1) few-shot 분류 태스크 생성기
# ----------------------------------------------------------------------
class FewShotTasks:
    """손글씨 숫자에서 N-way K-shot 태스크를 뽑는다."""

    def __init__(self, classes):
        digits = load_digits()
        X = digits.data / 16.0          # 0~1 정규화, (N,64)
        y = digits.target
        self.classes = list(classes)
        # 클래스별 샘플 인덱스 모음
        self.by_class = {c: np.where(y == c)[0] for c in self.classes}
        self.X = X

    def sample(self, n_way, k_shot, q_query, rng):
        """반환: 서포트(Xs,ys), 쿼리(Xq,yq). 라벨은 0..n_way-1 로 새로 매김."""
        chosen = rng.choice(self.classes, size=n_way, replace=False)
        Xs, ys, Xq, yq = [], [], [], []
        for new_label, c in enumerate(chosen):
            idx = rng.choice(self.by_class[c], size=k_shot + q_query, replace=False)
            Xs.append(self.X[idx[:k_shot]])
            ys += [new_label] * k_shot
            Xq.append(self.X[idx[k_shot:]])
            yq += [new_label] * q_query
        return (np.vstack(Xs), np.array(ys),
                np.vstack(Xq), np.array(yq))


# ----------------------------------------------------------------------
# 2) 임베딩 신경망 (64 -> 64 -> emb_dim), Adam으로 학습
# ----------------------------------------------------------------------
class Embedder:
    """입력을 임베딩 벡터로 바꾸는 MLP. 마지막 층은 선형(임베딩)."""

    def __init__(self, in_dim=64, hidden=64, emb_dim=32, lr=0.005,
                 weight_decay=0.0, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2.0 / in_dim), (in_dim, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, np.sqrt(2.0 / hidden), (hidden, emb_dim))
        self.b2 = np.zeros(emb_dim)
        self.lr = lr
        self.weight_decay = weight_decay   # 과적합 억제 (AdamW 방식)
        self._state = {n: [np.zeros_like(p), np.zeros_like(p)]
                       for n, p in [("W1", self.W1), ("b1", self.b1),
                                    ("W2", self.W2), ("b2", self.b2)]}
        self._t = 0

    def forward(self, X):
        self._X = X
        self._z1 = X @ self.W1 + self.b1
        self._a1 = np.maximum(0, self._z1)
        return self._a1 @ self.W2 + self.b2     # 임베딩 (선형)

    def backward_step(self, d_emb):
        """임베딩에 대한 상류 기울기 d_emb 를 받아 역전파 + Adam 갱신."""
        dW2 = self._a1.T @ d_emb
        db2 = d_emb.sum(axis=0)
        da1 = d_emb @ self.W2.T
        dz1 = da1 * (self._z1 > 0)
        dW1 = self._X.T @ dz1
        db1 = dz1.sum(axis=0)
        self._adam("W1", self.W1, dW1)
        self._adam("b1", self.b1, db1)
        self._adam("W2", self.W2, dW2)
        self._adam("b2", self.b2, db2)

    def _adam(self, name, param, grad):
        b1, b2, eps = 0.9, 0.999, 1e-8
        m, v = self._state[name]
        m[:] = b1 * m + (1 - b1) * grad
        v[:] = b2 * v + (1 - b2) * grad ** 2
        self._t_local = self._t  # (t는 호출 측에서 증가)
        m_hat = m / (1 - b1 ** self._t)
        v_hat = v / (1 - b2 ** self._t)
        param -= self.lr * (m_hat / (np.sqrt(v_hat) + eps) + self.weight_decay * param)


def _softmax(z, axis=1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


# ----------------------------------------------------------------------
# 3) 손실 + 임베딩에 대한 기울기 (직접 유도)
# ----------------------------------------------------------------------
def protonet_loss(emb_s, ys, emb_q, yq, n_way):
    """Prototypical Networks 손실. 쿼리를 가장 가까운 '클래스 평균'으로 분류.

    반환: (loss, d_emb_s, d_emb_q, accuracy)
    """
    D = emb_s.shape[1]
    counts = np.array([np.sum(ys == k) for k in range(n_way)])
    protos = np.stack([emb_s[ys == k].mean(axis=0) for k in range(n_way)])  # (n_way,D)

    diff = emb_q[:, None, :] - protos[None, :, :]     # (Nq, n_way, D)
    dist2 = (diff ** 2).sum(axis=2)                   # (Nq, n_way)
    logits = -dist2
    p = _softmax(logits, axis=1)
    Nq = emb_q.shape[0]
    loss = -np.mean(np.log(p[np.arange(Nq), yq] + 1e-12))
    acc = np.mean(logits.argmax(axis=1) == yq)

    # 역전파 (유클리드 거리 기반)
    dlogits = p.copy()
    dlogits[np.arange(Nq), yq] -= 1
    dlogits /= Nq                                     # dloss/dlogits
    ddiff = (-dlogits)[:, :, None] * 2 * diff         # logits=-dist2 -> chain
    d_emb_q = ddiff.sum(axis=1)                       # (Nq, D)
    d_protos = -ddiff.sum(axis=0)                     # (n_way, D)
    d_emb_s = np.zeros_like(emb_s)
    for k in range(n_way):
        d_emb_s[ys == k] = d_protos[k] / counts[k]    # 평균이므로 1/count 분배
    return loss, d_emb_s, d_emb_q, acc


def matching_loss(emb_s, ys, emb_q, yq, n_way):
    """Matching Networks 손실. 쿼리-서포트 '코사인 유사도'로 가중 투표.

    반환: (loss, d_emb_s, d_emb_q, accuracy)
    """
    eps = 1e-8
    ns = np.linalg.norm(emb_s, axis=1, keepdims=True) + eps
    nq = np.linalg.norm(emb_q, axis=1, keepdims=True) + eps
    Sn = emb_s / ns                                   # 정규화 (Ns,D)
    Qn = emb_q / nq                                   # (Nq,D)
    cos = Qn @ Sn.T                                   # (Nq, Ns) 코사인 유사도
    a = _softmax(cos, axis=1)                         # 서포트에 대한 어텐션
    Ys = np.zeros((emb_s.shape[0], n_way))
    Ys[np.arange(emb_s.shape[0]), ys] = 1
    probs = a @ Ys                                    # (Nq, n_way) 클래스별 투표 합
    Nq = emb_q.shape[0]
    loss = -np.mean(np.log(probs[np.arange(Nq), yq] + 1e-12))
    acc = np.mean(probs.argmax(axis=1) == yq)

    # 역전파: loss -> probs -> a -> cos -> Qn,Sn -> emb_q,emb_s
    dprobs = np.zeros_like(probs)
    dprobs[np.arange(Nq), yq] = -1.0 / (probs[np.arange(Nq), yq] + 1e-12)
    dprobs /= Nq
    da = dprobs @ Ys.T                                # (Nq, Ns)
    # softmax(행 방향) 역전파
    dcos = a * (da - (da * a).sum(axis=1, keepdims=True))
    dQn = dcos @ Sn                                   # (Nq, D)
    dSn = dcos.T @ Qn                                 # (Ns, D)
    # 정규화 역전파: u=x/||x|| 이면 dx = (du - (du·u)u)/||x||
    d_emb_q = (dQn - (dQn * Qn).sum(axis=1, keepdims=True) * Qn) / nq
    d_emb_s = (dSn - (dSn * Sn).sum(axis=1, keepdims=True) * Sn) / ns
    return loss, d_emb_s, d_emb_q, acc


# ----------------------------------------------------------------------
# 4) 학습 / 평가
# ----------------------------------------------------------------------
def train(loss_fn, tasks, n_way=3, k_shot=5, q_query=5, n_iters=1000,
          emb_dim=16, lr=0.003, weight_decay=1e-3, seed=0):
    """주어진 손실(protonet_loss/matching_loss)로 임베더를 메타학습."""
    embedder = Embedder(emb_dim=emb_dim, lr=lr, weight_decay=weight_decay, seed=seed)
    rng = np.random.default_rng(seed)
    acc_history = []
    for _ in range(n_iters):
        Xs, ys, Xq, yq = tasks.sample(n_way, k_shot, q_query, rng)
        n_s = Xs.shape[0]
        emb_all = embedder.forward(np.vstack([Xs, Xq]))   # 한 번에 순전파
        emb_s, emb_q = emb_all[:n_s], emb_all[n_s:]
        loss, d_s, d_q, acc = loss_fn(emb_s, ys, emb_q, yq, n_way)
        embedder._t += 1
        embedder.backward_step(np.vstack([d_s, d_q]))
        acc_history.append(acc)
    return embedder, acc_history


def evaluate(embedder, loss_fn, tasks, n_way=3, k_shot=5, q_query=5,
             n_tasks=200, seed=999):
    """새(테스트) 클래스 태스크들에서 평균 few-shot 정확도."""
    rng = np.random.default_rng(seed)
    accs = []
    for _ in range(n_tasks):
        Xs, ys, Xq, yq = tasks.sample(n_way, k_shot, q_query, rng)
        n_s = Xs.shape[0]
        emb_all = embedder.forward(np.vstack([Xs, Xq]))
        emb_s, emb_q = emb_all[:n_s], emb_all[n_s:]
        _, _, _, acc = loss_fn(emb_s, ys, emb_q, yq, n_way)
        accs.append(acc)
    return float(np.mean(accs))


def moving_average(x, window=50):
    x = np.asarray(x, dtype=float)
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")

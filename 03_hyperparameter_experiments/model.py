"""
경사하강법(Gradient Descent)을 직접 구현한 로지스틱 회귀 분류기
================================================================

scikit-learn을 쓰면 한 줄로 끝나지만, "학습률"과 "반복 횟수"가
학습에 어떤 영향을 주는지 눈으로 보려면 직접 구현하는 게 가장 좋다.

▶ 핵심 개념
  - 모델 예측: p = sigmoid(X·w + b)      # 0~1 사이 확률
  - 손실(loss): 이진 교차 엔트로피        # 예측이 틀릴수록 커짐
  - 학습(경사하강법):
      각 단계마다 손실이 줄어드는 방향(기울기, gradient)을 계산하고
      w, b 를 그 반대 방향으로 "학습률(lr)"만큼 이동시킨다.
          w <- w - lr * gradient
  - 이 과정을 "반복 횟수(n_iterations)"만큼 되풀이한다.

▶ 학습률(lr)이 핵심인 이유
  - 너무 작으면: 한 걸음이 너무 작아 느리게 수렴
  - 너무 크면 : 최저점을 지나쳐 튕겨나가 발산(학습 실패)
"""

import numpy as np


def sigmoid(z):
    """실수를 0~1 사이 확률로 변환. (오버플로 방지를 위해 z를 잘라줌)"""
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


class GradientDescentClassifier:
    """경사하강법으로 학습하는 이진 분류기 (로지스틱 회귀).

    Parameters
    ----------
    learning_rate : float
        가중치를 한 번에 얼마나 크게 이동할지 (학습률).
    n_iterations : int
        전체 데이터를 몇 번 반복 학습할지 (에폭/반복 횟수).
    record_history : bool
        매 반복마다 손실/정확도를 기록할지 여부 (그래프용).
    """

    def __init__(self, learning_rate=0.1, n_iterations=100, record_history=False):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.record_history = record_history
        self.weights = None
        self.bias = None
        # 학습 과정 기록 (record_history=True 일 때 채워짐)
        self.loss_history = []        # 매 반복의 손실
        self.train_acc_history = []   # 매 반복의 학습 데이터 정확도
        self.val_acc_history = []     # 매 반복의 검증/테스트 정확도 (X_val 제공 시)

    def fit(self, X, y, X_val=None, y_val=None):
        """경사하강법으로 weights, bias를 학습한다.

        record_history=True 이면 매 반복마다 손실과 학습 정확도를 기록하고,
        X_val, y_val 을 주면 검증(테스트) 정확도도 함께 기록한다.
        """
        n_samples, n_features = X.shape
        # 가중치는 0에서 시작 (모든 특성의 중요도를 모른다고 가정)
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        for _ in range(self.n_iterations):
            # 1) 예측: 현재 가중치로 확률 계산
            linear = np.dot(X, self.weights) + self.bias
            y_pred = sigmoid(linear)

            # 2) 기울기(gradient) 계산: 손실을 줄이는 방향
            error = y_pred - y
            dw = np.dot(X.T, error) / n_samples
            db = np.sum(error) / n_samples

            # 3) 가중치 갱신: 학습률만큼 반대 방향으로 이동
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # 4) (선택) 이번 단계의 손실/정확도 기록
            #    한 번의 학습으로 학습/검증 곡선을 모두 얻기 위해 둘 다 기록한다.
            if self.record_history:
                self.loss_history.append(self._bce_loss(y, y_pred))
                self.train_acc_history.append(self.score(X, y))
                if X_val is not None and y_val is not None:
                    self.val_acc_history.append(self.score(X_val, y_val))

        return self

    @staticmethod
    def _bce_loss(y_true, y_pred):
        """이진 교차 엔트로피 손실. (log(0) 방지를 위해 eps 사용)"""
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def predict_proba(self, X):
        """그룹 1에 속할 확률 반환."""
        return sigmoid(np.dot(X, self.weights) + self.bias)

    def predict(self, X):
        """확률 0.5 이상이면 1, 아니면 0으로 분류."""
        return (self.predict_proba(X) >= 0.5).astype(int)

    def score(self, X, y):
        """정확도(accuracy) = 맞게 맞힌 비율."""
        return np.mean(self.predict(X) == y)

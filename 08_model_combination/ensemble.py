"""
앙상블: 여러 종류의 모델을 합쳐 더 좋은 예측 만들기
====================================================

▶ 왜 합치나?
  - 모델마다 잘 맞히는 부분/실수하는 부분이 다르다.
  - 여러 모델의 예측을 합치면 개별 모델의 실수가 서로 상쇄되어
    보통 단일 모델보다 더 정확하고 안정적이다.
  - 입력이 고차원·복잡할수록 단일 모델이 놓치는 부분이 많아져 효과가 커진다.

▶ 세 가지 결합 방식
  - voting (하드 투표): 각 모델이 한 표씩, 다수결.
  - soft   (소프트 평균): 각 모델의 '클래스 확률'을 평균해 가장 높은 클래스 선택.
  - stacking (스태킹): 베이스 모델들의 예측을 입력으로 받는 '메타 모델'을 학습.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def make_model(name):
    """이름으로 기본 모델 생성 (표준화가 필요한 모델은 파이프라인으로)."""
    name = name.lower()
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    if name == "tree":
        return DecisionTreeClassifier(max_depth=6, random_state=0)
    if name == "knn":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7))
    if name == "svm":
        return make_pipeline(StandardScaler(), SVC(probability=True))
    if name == "forest":
        return RandomForestClassifier(n_estimators=120, random_state=0)
    raise ValueError(f"알 수 없는 모델: {name}")


class Ensemble:
    """여러 베이스 모델을 합치는 앙상블 (voting / soft / stacking)."""

    def __init__(self, model_names, method="soft"):
        self.model_names = model_names
        self.method = method
        self.models = [make_model(n) for n in model_names]
        self.meta = None   # stacking용 메타 모델

    def fit(self, X, y):
        if self.method == "stacking":
            # 누설(leakage) 방지: 교차검증 예측으로 메타 학습 데이터 생성
            meta_features = []
            for m in self.models:
                proba = cross_val_predict(m, X, y, cv=5, method="predict_proba")
                meta_features.append(proba)
                m.fit(X, y)   # 최종 예측용으로 전체 데이터에 다시 학습
            Z = np.hstack(meta_features)
            self.meta = LogisticRegression(max_iter=2000)
            self.meta.fit(Z, y)
        else:
            for m in self.models:
                m.fit(X, y)
        return self

    def _base_proba(self, X):
        return [m.predict_proba(X) for m in self.models]

    def predict(self, X):
        if self.method == "voting":
            preds = np.array([m.predict(X) for m in self.models])  # (n_models, N)
            # 각 샘플에 대해 다수결
            from scipy.stats import mode
            return mode(preds, axis=0, keepdims=False).mode
        if self.method == "soft":
            avg = np.mean(self._base_proba(X), axis=0)
            return avg.argmax(axis=1)
        if self.method == "stacking":
            Z = np.hstack(self._base_proba(X))
            return self.meta.predict(Z)
        raise ValueError(self.method)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == y))

    def base_scores(self, X, y):
        """개별 베이스 모델 각각의 정확도 (비교용)."""
        return {n: float(np.mean(m.predict(X) == y))
                for n, m in zip(self.model_names, self.models)}

"""
[지도학습 - 분류] 로지스틱 회귀 (Logistic Regression)
======================================================

▶ 분류(Classification)란?
  - 정답이 "범주(class)"인 문제. (예: 합격/불합격, 개/고양이)
  - 여기서는 2개의 그룹을 나누는 "이진 분류(binary classification)"를 다룬다.

▶ 로지스틱 회귀
  - 이름은 '회귀'지만 사실은 "분류" 알고리즘이다.
  - 입력을 받아 "그룹 1에 속할 확률(0~1)"을 출력하고,
    0.5를 기준으로 어느 그룹인지 판단한다.

▶ 이 스크립트가 보여주는 것
  - 2차원 데이터(특성 2개)를 두 그룹으로 분류
  - 학습/테스트 데이터를 나눠 정확도(accuracy) 측정
  - "결정 경계(decision boundary)" 시각화
    : 모델이 두 그룹을 어떻게 가르는지 보여주는 선
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def main():
    # 1) 데이터 생성 ------------------------------------------------------
    #    특성 2개짜리 데이터를 두 그룹(class 0, class 1)으로 만든다.
    X, y = make_classification(
        n_samples=300,
        n_features=2,        # 특성 2개 (그래프로 그리기 위해)
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        class_sep=1.2,       # 두 그룹이 떨어진 정도 (클수록 분류 쉬움)
        random_state=42,
    )

    # 2) 학습/테스트 분리 -------------------------------------------------
    #    학습에 쓰지 않은 데이터로 평가해야 "진짜 실력"을 알 수 있다.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 3) 모델 학습 --------------------------------------------------------
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # 4) 평가 -------------------------------------------------------------
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    cm = confusion_matrix(y_test, model.predict(X_test))

    print("=" * 55)
    print(" 로지스틱 회귀 (Logistic Regression) 분류 결과")
    print("=" * 55)
    print(f"  학습 데이터 정확도: {train_acc:.3f}")
    print(f"  테스트 정확도     : {test_acc:.3f}  <- 진짜 실력!")
    print("\n  혼동 행렬(Confusion Matrix):")
    print("            예측0  예측1")
    print(f"   실제0 :   {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"   실제1 :   {cm[1,0]:4d}  {cm[1,1]:4d}")

    # 5) 시각화: 데이터 + 결정 경계 ---------------------------------------
    plt.figure(figsize=(8, 6))

    # 배경 전체를 격자로 만들어 각 위치에서 모델이 어떤 그룹으로 예측하는지 색칠
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.2, cmap="coolwarm")

    # 실제 데이터 점 찍기 (그룹별 색 구분)
    plt.scatter(X[y == 0, 0], X[y == 0, 1], color="#4C72B0", edgecolor="k",
                s=30, label="Class 0")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color="#C44E52", edgecolor="k",
                s=30, label="Class 1")

    plt.title(f"Logistic Regression Decision Boundary (test acc={test_acc:.2f})",
              fontsize=12)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()

    save_path = os.path.join(OUTPUT_DIR, "02_logistic_regression.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 그래프 저장됨 -> {save_path}")


if __name__ == "__main__":
    main()

"""
[지도학습] 여러 분류 모델 비교하기
====================================

같은 문제라도 알고리즘은 여러 가지가 있다.
어떤 모델이 더 잘 맞히는지 "정확도(accuracy)"로 비교해 본다.

▶ 비교하는 모델들
  - Logistic Regression : 선형 분류의 기본
  - K-Nearest Neighbors : 가까운 이웃 k개를 보고 다수결
  - Decision Tree       : 스무고개처럼 질문을 나눠 분류
  - SVM                 : 두 그룹 사이 간격을 최대로 벌리는 경계
  - Random Forest       : 결정 트리 여러 개의 투표 (앙상블)

▶ 사용 데이터: 유방암 진단 데이터 (sklearn 내장, 실제 의료 데이터)
  - 30개 특성으로 종양이 '악성/양성'인지 분류하는 이진 분류 문제
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def main():
    # 1) 데이터 로드 ------------------------------------------------------
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 2) 비교할 모델들 정의 ----------------------------------------------
    #    StandardScaler로 특성 크기를 맞춰주면(표준화) 거리 기반 모델 성능이 좋아진다.
    #    make_pipeline: 표준화 → 모델 을 하나로 묶어준다.
    models = {
        "Logistic\nRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
        "KNN\n(k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "Decision\nTree": DecisionTreeClassifier(max_depth=4, random_state=42),
        "SVM\n(RBF)": make_pipeline(StandardScaler(), SVC()),
        "Random\nForest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    # 3) 각 모델 학습 + 정확도 측정 --------------------------------------
    names, accuracies = [], []
    print("=" * 55)
    print(" 분류 모델별 테스트 정확도 비교")
    print("=" * 55)
    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        names.append(name.replace("\n", " "))
        accuracies.append(acc)
        print(f"  {name.replace(chr(10), ' '):24s}: {acc:.4f}")

    # 4) 막대그래프로 비교 시각화 ----------------------------------------
    plt.figure(figsize=(9, 6))
    bars = plt.bar(list(models.keys()), accuracies,
                   color=["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974"])

    # 막대 위에 정확도 숫자 표시
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, acc + 0.005,
                 f"{acc:.3f}", ha="center", va="bottom", fontsize=10)

    plt.title("Classification Model Comparison (Breast Cancer Dataset)", fontsize=13)
    plt.ylabel("Test Accuracy")
    plt.ylim(min(accuracies) - 0.05, 1.02)
    plt.grid(True, axis="y", alpha=0.3)

    save_path = os.path.join(OUTPUT_DIR, "03_model_comparison.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 그래프 저장됨 -> {save_path}")


if __name__ == "__main__":
    main()

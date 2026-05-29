"""
[지도학습 - 회귀] 선형 회귀 (Linear Regression)
==================================================

▶ 지도학습(Supervised Learning)이란?
  - 입력 X와 "정답" y를 함께 주고, X로부터 y를 예측하는 함수를 배우는 것.

▶ 회귀(Regression)란?
  - 예측하려는 정답이 "연속적인 숫자"인 문제. (예: 집값, 온도, 키)
  - 분류(Classification)는 정답이 "범주(class)"인 문제. (예: 합격/불합격)

▶ 선형 회귀의 아이디어
  - 데이터를 가장 잘 설명하는 직선 y = w*x + b 를 찾는다.
  - w(기울기)와 b(절편)를 데이터로부터 학습한다.

이 스크립트는 가상의 데이터(공부 시간 → 시험 점수)를 만들어
직선을 학습시키고, 결과를 그래프로 저장한다.
"""

import os

import matplotlib

matplotlib.use("Agg")  # 화면 없이 파일로만 저장 (서버 환경 대응)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 그래프를 저장할 폴더 경로 (이 파일 기준 상위 폴더의 outputs/)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def main():
    # 1) 가상 데이터 생성 -------------------------------------------------
    #    "공부 시간(x)이 늘면 시험 점수(y)도 대체로 오른다"는 데이터를 만든다.
    #    실제 데이터는 직선에서 약간씩 벗어나므로 noise(잡음)를 더해준다.
    rng = np.random.default_rng(42)  # 결과 재현을 위한 고정 시드
    n_samples = 50
    study_hours = rng.uniform(0, 10, size=n_samples)          # 0~10시간
    true_w, true_b = 8.0, 20.0                                # 실제 규칙(우리는 모른다고 가정)
    noise = rng.normal(0, 8, size=n_samples)                  # 현실의 불규칙성
    scores = true_w * study_hours + true_b + noise            # 시험 점수
    scores = np.clip(scores, 0, 100)                          # 점수는 0~100 사이

    # scikit-learn 모델은 입력 X가 2차원(행=샘플, 열=특성) 이어야 한다.
    X = study_hours.reshape(-1, 1)
    y = scores

    # 2) 모델 학습 --------------------------------------------------------
    #    LinearRegression은 최적의 w, b를 수학적으로 한 번에 계산한다.
    model = LinearRegression()
    model.fit(X, y)

    learned_w = model.coef_[0]
    learned_b = model.intercept_

    # 3) 예측 및 평가 -----------------------------------------------------
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)   # 평균 제곱 오차 (작을수록 좋음)
    r2 = r2_score(y, y_pred)              # 결정계수 R² (1에 가까울수록 좋음)

    print("=" * 55)
    print(" 선형 회귀 (Linear Regression) 결과")
    print("=" * 55)
    print(f"  실제 규칙  : y = {true_w:.2f} * x + {true_b:.2f}  (정답)")
    print(f"  학습한 규칙: y = {learned_w:.2f} * x + {learned_b:.2f}  (모델이 찾은 것)")
    print(f"  MSE(평균제곱오차): {mse:.2f}")
    print(f"  R² (1에 가까울수록 좋음): {r2:.3f}")
    print()
    print("  예) 7시간 공부하면 예상 점수:", round(model.predict([[7]])[0], 1), "점")

    # 4) 시각화 -----------------------------------------------------------
    plt.figure(figsize=(8, 6))
    plt.scatter(study_hours, scores, color="#4C72B0", alpha=0.7, label="Data (actual)")

    # 학습된 직선을 그리기 위해 x축을 촘촘히 만든다.
    x_line = np.linspace(0, 10, 100).reshape(-1, 1)
    plt.plot(x_line, model.predict(x_line), color="#C44E52", linewidth=2.5,
             label=f"Learned line: y={learned_w:.1f}x+{learned_b:.1f}")

    plt.title("Linear Regression: Study Hours vs Exam Score", fontsize=13)
    plt.xlabel("Study Hours")
    plt.ylabel("Exam Score")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(OUTPUT_DIR, "01_linear_regression.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 그래프 저장됨 -> {save_path}")


if __name__ == "__main__":
    main()

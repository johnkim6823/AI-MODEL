# 🤖 AI 모델 기초 학습 (AI Model Basics)

머신러닝(ML)의 기본 개념을 **직접 코드로 돌려보며** 공부하는 자료입니다.
지도학습 / 비지도학습부터 시작해서, **학습률(learning rate)과 반복 횟수(epochs)를
바꿨을 때 정확도(accuracy)가 어떻게 달라지는지**를 matplotlib 그래프로 확인합니다.

---

## 📚 머신러닝 한눈에 보기

```
                        머신러닝 (Machine Learning)
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
   지도학습                   비지도학습                  강화학습
 (Supervised)              (Unsupervised)            (Reinforcement)
        │                         │                  * 본 자료 범위 밖
   "정답(label)이 있다"      "정답이 없다"
        │                         │
   ┌────┴────┐              ┌─────┴─────┐
 회귀(Regression)        군집화(Clustering)
 분류(Classification)    차원축소(Dim. Reduction)
```

| 구분 | 입력 데이터 | 목표 | 예시 |
|------|------------|------|------|
| **지도학습** | 입력 X + 정답 y | 정답을 맞히는 함수 학습 | 스팸 분류, 집값 예측 |
| **비지도학습** | 입력 X 만 (정답 없음) | 숨은 구조/패턴 발견 | 고객 그룹화, 데이터 압축 |

---

## 🗂️ 폴더 구성

```
AI-MODEL/
├── 01_supervised_learning/        # 1. 지도학습
│   ├── 01_linear_regression.py        선형 회귀 (연속값 예측)
│   ├── 02_logistic_regression.py      로지스틱 회귀 (분류 + 결정 경계)
│   └── 03_model_comparison.py         여러 분류 모델 정확도 비교
│
├── 02_unsupervised_learning/      # 2. 비지도학습
│   ├── 01_kmeans_clustering.py        K-평균 군집화 + 엘보우 기법
│   └── 02_pca_dimensionality.py       PCA 차원 축소
│
├── 03_hyperparameter_experiments/ # 3. 하이퍼파라미터 실험 ⭐ 핵심
│   ├── model.py                       경사하강법 직접 구현 (from scratch)
│   ├── 00_gradient_descent_intuition.py 학습률의 원리 (그릇 모델, 발산 시연)
│   ├── 01_learning_rate_experiment.py 학습률 → 정확도
│   ├── 02_epochs_experiment.py        반복 횟수 → 정확도 (과적합 관찰)
│   └── 03_grid_search_heatmap.py      학습률×반복횟수 히트맵
│
├── outputs/                       # 생성된 그래프(png)가 저장되는 곳
├── requirements.txt
└── run_all.py                     # 모든 예제 한 번에 실행
```

---

## 🚀 실행 방법

```bash
# 1) 패키지 설치
pip install -r requirements.txt

# 2) 개별 실행 (예시)
python 01_supervised_learning/01_linear_regression.py

# 3) 전체 실행 → outputs/ 폴더에 그래프가 모두 생성됨
python run_all.py
```

> 그래프는 화면에 띄우지 않고 `outputs/` 폴더에 **PNG 파일로 저장**됩니다.
> (서버 환경에서도 동작하도록 `matplotlib`의 `Agg` 백엔드를 사용합니다.)

---

## 🎯 핵심 용어 정리

| 용어 | 의미 |
|------|------|
| **특성 (Feature, X)** | 모델에 넣는 입력값 (예: 키, 몸무게) |
| **레이블 (Label, y)** | 맞혀야 하는 정답 (예: 합격/불합격) |
| **학습 (Training)** | 데이터로 모델의 내부 값(가중치)을 조정하는 과정 |
| **가중치 (Weight, w)** | 모델이 학습하는 숫자. 입력의 중요도 |
| **손실 (Loss)** | 예측이 정답과 얼마나 틀렸는지 나타내는 값 (작을수록 좋음) |
| **경사하강법 (Gradient Descent)** | 손실을 줄이는 방향으로 가중치를 조금씩 옮기는 학습 방법 |
| **학습률 (Learning Rate)** | 가중치를 "얼마나 크게" 옮길지 정하는 값 |
| **에폭 (Epoch) / 반복 횟수** | 데이터 전체를 몇 번 반복 학습할지 |
| **정확도 (Accuracy)** | 전체 중 맞게 분류한 비율 (분류 문제 평가지표) |
| **과적합 (Overfitting)** | 학습 데이터만 잘 맞히고 새 데이터는 못 맞히는 상태 |

---

## 🔑 학습률·반복 횟수가 정확도에 주는 영향 (직관)

```
학습률이 너무 작으면 → 학습이 너무 느림 (정해진 횟수 안에 수렴 못 함)
학습률이 적당하면     → 빠르고 안정적으로 수렴 ✅
학습률이 너무 크면     → 손실이 발산하거나 진동 (학습 실패)

반복 횟수가 너무 적으면 → 덜 배움 (과소적합, Underfitting)
반복 횟수가 적당하면     → 잘 배움 ✅
반복 횟수가 너무 많으면 → 학습 데이터에만 과하게 맞춤 (과적합, Overfitting)
```

이 내용을 `03_hyperparameter_experiments/` 의 그래프로 직접 확인하세요!

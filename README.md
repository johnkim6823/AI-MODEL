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
        │                         │                         │
   "정답(label)이 있다"      "정답이 없다"          "보상(reward)으로 배운다"
        │                         │                         │
   ┌────┴────┐              ┌─────┴─────┐             ┌─────┴─────┐
 회귀(Regression)        군집화(Clustering)        밴딧(Bandit)
 분류(Classification)    차원축소(Dim. Reduction)   Q러닝(Q-Learning)
```

| 구분 | 입력 데이터 | 목표 | 예시 |
|------|------------|------|------|
| **지도학습** | 입력 X + 정답 y | 정답을 맞히는 함수 학습 | 스팸 분류, 집값 예측 |
| **비지도학습** | 입력 X 만 (정답 없음) | 숨은 구조/패턴 발견 | 고객 그룹화, 데이터 압축 |
| **강화학습** | 환경과의 상호작용 (보상) | 보상을 최대로 하는 행동 학습 | 게임 AI, 로봇 제어 |

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
├── 04_reinforcement_learning/     # 4. 강화학습
│   ├── gridworld.py                   GridWorld 환경 + Q러닝 (from scratch)
│   ├── 01_multi_armed_bandit.py       멀티암드 밴딧 (탐험 vs 활용)
│   ├── 02_q_learning_gridworld.py     Q러닝 길 찾기 + 정책 시각화
│   └── 03_rl_hyperparameter_experiments.py 학습률·탐험율·할인율 → 성능
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

### 강화학습 용어

| 용어 | 의미 |
|------|------|
| **에이전트 (Agent)** | 학습하며 행동하는 주체 (예: 로봇, 게임 플레이어) |
| **환경 (Environment)** | 에이전트가 상호작용하는 세계 (예: 격자 미로) |
| **상태 (State)** | 현재 상황 (예: 격자에서의 위치) |
| **행동 (Action)** | 에이전트가 할 수 있는 선택 (예: 상/하/좌/우) |
| **보상 (Reward)** | 행동의 결과로 받는 점수 (목표: 총 보상 최대화) |
| **정책 (Policy)** | 각 상태에서 어떤 행동을 할지에 대한 규칙 |
| **Q값 (Q-value)** | "이 상태에서 이 행동을 하면 받을 총 보상의 기대치" |
| **탐험/활용 (Explore/Exploit)** | 새 행동을 시도 vs 지금까지 최선을 선택 |
| **탐험율 (ε, epsilon)** | 무작위로 탐험할 확률 |
| **할인율 (γ, gamma)** | 미래 보상을 얼마나 중요하게 볼지 (0~1) |

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

---

## 🕹️ 강화학습 직관

```
탐험율(ε)이 너무 작으면 → 더 좋은 선택을 못 찾고 갇힐 수 있음
탐험율이 적당하면        → 탐험으로 좋은 길을 찾고 활용 ✅
탐험율이 너무 크면        → 다 알면서도 자꾸 무작위로 행동해 손해

학습률(α)이 너무 작으면  → Q값이 느리게 갱신되어 학습이 느림
학습률이 적당하면        → 빠르게 수렴 ✅
할인율(γ)                → 클수록 먼 미래 보상까지 고려 (장기 문제에서 중요)
```

지도학습의 "학습률·반복 횟수"처럼, 강화학습도 이런 값들을 바꾸면
성능이 달라진다. `04_reinforcement_learning/`의 그래프로 직접 확인하세요!

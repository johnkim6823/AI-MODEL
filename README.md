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

> **딥러닝(Deep Learning)** 과 **메타러닝(Meta-Learning)** 은 위 3분류와 '나란히'
> 있는 개념이 아니라 가로지르는 개념이다.
> - **딥러닝**: 여러 층을 쌓은 '신경망'이라는 *모델/도구*. 지도·비지도·강화학습
>   어디에나 쓰인다.
> - **심층 강화학습(Deep RL)**: 강화학습에 신경망을 결합한 것. (DQN, A2C, SAC 등)
> - **메타러닝**: *'새 문제에 빨리 적응하는 능력'을 배우는 학습 방식* (few-shot).
> - **모델 결합**: 한 모델만 쓰지 않고 *여러 모델을 합쳐* 더 좋고 안정적인 결과를
>   내는 것. (앙상블, 메타러닝+RL 하이브리드)

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
├── 05_deep_learning/              # 5. 딥러닝 (신경망)
│   ├── neural_network.py              다층 신경망(MLP) + 역전파 + 정규화/옵티마이저
│   ├── cnn.py                         합성곱 신경망(CNN) 직접 구현 (im2col)
│   ├── 01_perceptron_to_mlp.py        왜 은닉층이 필요한가 (XOR 문제)
│   ├── 02_mlp_classification.py       신경망 비선형 분류 + 학습 곡선
│   ├── 03_depth_activation_experiment.py 망 깊이·너비·활성화 → 정확도
│   ├── 04_regularization.py           L2·드롭아웃으로 과적합 줄이기
│   ├── 05_optimizer_comparison.py     SGD vs Momentum vs Adam
│   └── 06_cnn_basics.py               합성곱 직관 + 학습된 필터 (CNN)
│
├── 06_meta_learning/              # 6. 메타러닝 (배우는 법을 배우기)
│   ├── meta_core.py                   사인파 태스크 + 함수형 MLP + 메타 알고리즘
│   ├── metric_core.py                 few-shot 분류 태스크 + 임베딩망 + 메트릭 손실
│   ├── 01_few_shot_sine_problem.py    few-shot 문제 소개
│   ├── 02_maml.py                     MAML (빠른 적응 초기값 학습)
│   ├── 03_reptile.py                  Reptile (더 단순한 메타러닝)
│   ├── 04_compare_methods.py          Random/Joint/Reptile/MAML 비교 + K-shot
│   ├── 05_meta_sgd.py                 Meta-SGD (학습률까지 학습)
│   ├── 06_maml_first_vs_second_order.py  1차(FOMAML) vs 2차 MAML
│   ├── 07_prototypical_networks.py    Prototypical Networks (거리 기반)
│   └── 08_matching_networks.py        Matching Networks (유사도 어텐션)
│
├── 07_deep_rl/                    # 7. 심층 강화학습 (DRL)
│   ├── envs.py                        CartPole(이산)·Pendulum(연속) 환경 직접 구현
│   ├── drl_nn.py                      입력기울기 지원 MLP + Adam + 리플레이버퍼
│   ├── dqn.py / a2c.py / a3c.py       DQN·DDQN / A2C / A3C 에이전트
│   ├── ddpg.py / sac.py               DDPG / SAC (연속 제어)
│   ├── 01_dqn_ddqn_cartpole.py        DQN vs Double DQN
│   ├── 02_a2c_a3c_cartpole.py         A2C vs A3C
│   └── 03_ddpg_sac_pendulum.py        DDPG vs SAC (연속)
│
├── 08_model_combination/          # 8. 모델 결합 (앙상블 / 하이브리드)
│   ├── exp_config.py                  config + 커맨드라인 인자 둘 다 지원
│   ├── ensemble.py                    voting / soft / stacking 앙상블
│   ├── hybrid_core.py                 Reptile + 정책경사(RL) 하이브리드 코어
│   ├── 01_ensemble_experiment.py      여러 모델 합치기 (입력 차원 sweep) ⭐설정가능
│   └── 02_meta_rl_hybrid.py           메타러닝+RL 하이브리드 ⭐설정가능
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

### 🎛️ 직접 값을 넣어 실험하기 (8번 모델 결합)

`08_model_combination/` 의 실험은 **config 딕셔너리**(파일 상단)와
**커맨드라인 인자**를 둘 다 지원한다. 그냥 실행하면 기본값을 쓰고,
인자로 값을 덮어쓸 수 있다.

```bash
# 앙상블: 합칠 모델·결합방식·입력차원 등을 직접 지정
python 08_model_combination/01_ensemble_experiment.py \
       --models logreg,tree,svm,forest --ensemble stacking --n_features 100

# 메타러닝+RL 하이브리드: 입력차원·적응스텝·메타보폭 등 조절
python 08_model_combination/02_meta_rl_hybrid.py --d 20 --inner_steps 15 --meta_lr 0.2
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

### 딥러닝 용어

| 용어 | 의미 |
|------|------|
| **신경망 (Neural Network)** | 여러 층의 뉴런으로 입력→출력을 잇는 모델 |
| **층 (Layer) / 은닉층** | 뉴런들의 묶음. 입력·출력 사이의 층이 '은닉층' |
| **활성화 함수 (Activation)** | 비선형성을 주는 함수 (ReLU/tanh/sigmoid). 딥러닝의 표현력 핵심 |
| **역전파 (Backpropagation)** | 손실의 기울기를 출력→입력으로 거꾸로 전파해 모든 가중치 기울기를 구함 |
| **ReLU** | max(0, x). 가장 널리 쓰는 활성화 함수 (빠르고 안정적) |
| **정규화 (Regularization)** | 과적합 억제 기법. L2(가중치 감쇠), 드롭아웃 등 |
| **드롭아웃 (Dropout)** | 학습 중 뉴런 일부를 무작위로 꺼 특정 뉴런 의존을 줄임 |
| **옵티마이저 (Optimizer)** | 기울기로 가중치를 갱신하는 규칙 (SGD/Momentum/Adam) |
| **Adam** | 파라미터별 보폭을 자동 조절하는 옵티마이저 (보통 빠르고 안정적) |
| **합성곱/CNN** | 작은 필터를 이미지에 미끄러뜨려 국소 패턴을 추출하는 신경망 |
| **풀링 (Pooling)** | 특징맵을 줄여(예: 2x2 최댓값) 계산량↓·위치 강건성↑ |

### 메타러닝 용어

| 용어 | 의미 |
|------|------|
| **메타러닝 (Meta-Learning)** | '배우는 법'을 배우기. 새 문제에 빨리 적응하는 능력을 학습 |
| **태스크 (Task)** | 하나의 작은 학습 문제 (예: 하나의 사인파) |
| **few-shot 학습** | 아주 적은 예시(K개)만으로 새 문제를 푸는 것 |
| **서포트셋/쿼리셋** | 적응에 쓰는 예시 / 적응 후 평가에 쓰는 예시 |
| **이너/아우터 루프** | 한 태스크에 적응하는 과정 / 초기값을 개선하는 과정 |
| **MAML / Reptile** | 최적화 기반 메타러닝 (빠르게 적응하는 '초기값'을 학습) |
| **Meta-SGD** | 초기값뿐 아니라 '파라미터별 학습률'까지 학습 |
| **N-way K-shot** | N개 클래스를 각 K개 예시로 구분하는 few-shot 분류 문제 |
| **메트릭 기반** | 임베딩 공간의 거리/유사도로 분류 (ProtoNet, Matching Networks) |
| **프로토타입 (Prototype)** | 한 클래스 서포트 임베딩들의 평균 (ProtoNet) |

### 심층 강화학습(DRL) 용어

| 용어 | 의미 |
|------|------|
| **DQN / Double DQN** | 신경망으로 Q값을 근사하는 가치 기반 DRL (DDQN은 과대평가 완화) |
| **리플레이 버퍼** | 과거 경험을 저장해 무작위로 뽑아 학습 (상관성↓) |
| **타깃 네트워크** | 목표값 계산용 별도 네트워크(천천히 갱신)로 학습 안정화 |
| **액터-크리틱** | 정책망(액터) + 가치망(크리틱)을 함께 학습 (A2C/A3C/DDPG/SAC) |
| **A2C / A3C** | 어드밴티지 액터-크리틱 / 그 비동기·병렬 워커 버전 |
| **DDPG** | 연속 행동용 결정적 정책 경사 (off-policy) |
| **SAC** | 연속 행동용, 엔트로피 보상 + 트윈 크리틱 (안정적) |

### 모델 결합 용어

| 용어 | 의미 |
|------|------|
| **앙상블 (Ensemble)** | 여러 모델의 예측을 합쳐 더 좋고 안정적인 결과 |
| **voting / soft / stacking** | 다수결 / 확률 평균 / 메타모델로 합치는 방식 |
| **하이브리드 (Hybrid)** | 서로 다른 패러다임을 결합 (예: 메타러닝 + 강화학습) |

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

---

## 🧠 딥러닝 직관

```
은닉층이 없으면(=선형 모델) → XOR 같은 복잡한 패턴을 못 푼다 (정확도 ≈ 찍기)
은닉층 + 활성화 함수        → 곡선 경계를 만들어 복잡한 패턴도 학습 ✅
망을 키우면(깊이·너비↑)     → 표현력↑ (단, 너무 키우면 과적합 위험)
활성화 함수                → 보통 ReLU가 빠르고 안정적, sigmoid는 느려지기 쉬움
정규화(L2/드롭아웃)         → 과적합을 줄여 테스트 성능↑
옵티마이저(Adam 등)         → SGD보다 빠르고 안정적으로 수렴
CNN(합성곱)                → 이미지의 국소 패턴(모서리 등)을 효율적으로 추출
```

`05_deep_learning/`에서 XOR·망 구조·활성화·정규화·옵티마이저·CNN을 확인하세요!

---

## 🧩 메타러닝 직관

```
보통 학습   : 하나의 문제를 잘 푸는 모델을 만든다.
메타러닝     : '새 문제를 몇 개의 예시로 빨리 푸는 능력'을 배운다 (few-shot).

무작위 초기값 → 점 10개로는 새 사인파를 못 맞힘
메타학습 초기값(MAML/Reptile) → 같은 10개로 몇 스텝 만에 잘 맞힘 ✅
예시 수 K가 많을수록 → 적응 후 오차↓

[메타러닝 두 갈래]
최적화 기반 : MAML, Reptile, Meta-SGD (좋은 초기값/학습률을 배워 빠르게 적응)
메트릭 기반 : ProtoNet, Matching Networks (임베딩 공간의 거리/유사도로 분류)
```

`06_meta_learning/`에서 MAML·Reptile·Meta-SGD(최적화 기반)와
ProtoNet·Matching Networks(메트릭 기반)를 모두 확인하세요!

---

## 🎮 심층 강화학습(DRL) 직관

```
표(Q-table)로는 불가능 → 상태가 연속이면 '신경망'으로 Q/정책을 근사 (= DRL)

가치 기반 : DQN, Double DQN   (Q값을 신경망으로; 리플레이+타깃망으로 안정화)
정책 기반 : A2C, A3C          (정책을 직접 학습; A3C는 병렬 워커로 가속)
연속 제어 : DDPG, SAC         (토크 같은 연속 행동; SAC는 엔트로피로 더 안정적)
```

`07_deep_rl/`에서 CartPole(균형 잡기)·Pendulum(진자 세우기)을 학습하는
6가지 DRL의 학습 곡선을 확인하세요!

---

## 🧬 모델 결합 직관 (여러 모델 합치기)

```
[왜?] 모델마다 잘하는/실수하는 부분이 달라서, 합치면 실수가 상쇄된다.

앙상블        : 여러 종류 모델의 예측을 합침 (voting/soft/stacking)
              → '최고 단일 모델'과 대등 + '평균 단일 모델'보다 훨씬 나음
              → 입력이 많고 복잡할수록 이점이 커진다 ✅
하이브리드     : 다른 패러다임을 결합 (예: Reptile로 초기화 → RL로 빠르게 적응)
              → 새 과제에 가장 빠르게 적응

설정 가능: config 딕셔너리 또는 커맨드라인 인자로 직접 값을 넣어 실험.
```

`08_model_combination/`에서 앙상블 실험과 메타러닝+RL 하이브리드를
직접 값을 바꿔가며 실험하세요!

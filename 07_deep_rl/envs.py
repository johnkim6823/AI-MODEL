"""
강화학습 환경 (라이브러리 없이 직접 구현) — CartPole, Pendulum
================================================================

DRL 예제들이 함께 쓰는 환경. gym과 비슷한 인터페이스를 제공한다.
  env.reset() -> obs
  env.step(action) -> (obs, reward, done, info)

▶ CartPole (이산 행동, 2개)
  - 막대가 달린 수레를 좌/우로 밀어 막대를 쓰러뜨리지 않고 버티기.
  - 상태 4차원 [위치, 속도, 각도, 각속도], 매 스텝 +1 보상.
  - 막대가 너무 기울거나(±12°) 수레가 화면 밖(±2.4)이면 종료.

▶ Pendulum (연속 행동, 토크 1차원)
  - 진자를 흔들어 거꾸로 세우고 유지하기.
  - 상태 3차원 [cosθ, sinθ, 각속도], 행동은 토크 u∈[-2,2].
  - 보상 = -(각도² + 0.1·각속도² + 0.001·u²)  (위로 설수록 0에 가까움)
"""

import numpy as np


class CartPole:
    """고전 제어 문제: 막대 균형 잡기 (이산 행동)."""

    def __init__(self, max_steps=200, seed=0):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masscart + self.masspole
        self.length = 0.5                       # 막대 절반 길이
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02                         # 시간 간격
        self.theta_threshold = 12 * np.pi / 180  # 막대 한계 각도
        self.x_threshold = 2.4                  # 수레 한계 위치
        self.max_steps = max_steps
        self.obs_dim = 4
        self.n_actions = 2
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.state = self.rng.uniform(-0.05, 0.05, size=4)
        self.steps = 0
        return self.state.copy()

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta, sintheta = np.cos(theta), np.sin(theta)

        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # 오일러 적분
        x += self.tau * x_dot
        x_dot += self.tau * xacc
        theta += self.tau * theta_dot
        theta_dot += self.tau * thetaacc
        self.state = np.array([x, x_dot, theta, theta_dot])
        self.steps += 1

        done = bool(abs(x) > self.x_threshold or abs(theta) > self.theta_threshold
                    or self.steps >= self.max_steps)
        reward = 1.0   # 버틴 매 스텝마다 +1
        return self.state.copy(), reward, done, {}


class Pendulum:
    """진자 세우기 (연속 행동: 토크)."""

    def __init__(self, max_steps=200, seed=0):
        self.g = 10.0
        self.m = 1.0
        self.l = 1.0
        self.dt = 0.05
        self.max_speed = 8.0
        self.max_torque = 2.0
        self.max_steps = max_steps
        self.obs_dim = 3
        self.action_dim = 1
        self.action_high = self.max_torque
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.theta = self.rng.uniform(-np.pi, np.pi)
        self.theta_dot = self.rng.uniform(-1, 1)
        self.steps = 0
        return self._obs()

    def _obs(self):
        return np.array([np.cos(self.theta), np.sin(self.theta), self.theta_dot])

    @staticmethod
    def _angle_normalize(x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi

    def step(self, action):
        u = float(np.clip(np.ravel(action)[0], -self.max_torque, self.max_torque))
        th, thdot = self.theta, self.theta_dot

        cost = self._angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * u ** 2

        newthdot = thdot + (3 * self.g / (2 * self.l) * np.sin(th)
                            + 3.0 / (self.m * self.l ** 2) * u) * self.dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * self.dt

        self.theta, self.theta_dot = newth, newthdot
        self.steps += 1
        done = bool(self.steps >= self.max_steps)
        return self._obs(), -cost, done, {}

"""
[모델 결합] 메타러닝 + 강화학습 하이브리드 (Reptile + 정책경사) ⭐
==================================================================

"한 종류 모델만 쓰지 말고, 메타러닝과 강화학습을 '합쳐' 새 과제에 빠르게
적응하자" — reptile+SAC 같은 결합의 아이디어를 직접 확인한다.

▶ 무엇을 하나
  - 여러 RL 태스크(컨텍스추얼 밴딧) 분포에서 Reptile로 '정책의 좋은 초기값'을
    메타학습한다(메타러닝).
  - 새 태스크에서 그 초기값으로 시작해 REINFORCE(강화학습)로 적응한다.

▶ 비교 (적응 곡선: 적응 단계 수 vs 성능)
  - Random     : 학습 안 한 무작위 초기값
  - Joint      : 메타 없이 여러 태스크를 그냥 RL로 학습(평균만 배워 적응 둔함)
  - Reptile+RL : 메타학습된 초기값 → 가장 빠르게 적응 ✅

▶ 직접 값을 바꿔 실험 (config 또는 커맨드라인 둘 다 지원)
    python 02_meta_rl_hybrid.py --d 20 --inner_steps 15 --meta_lr 0.2
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import hybrid_core as h
from exp_config import build_config, print_config

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")

# ── 기본 설정 (여기 값을 바꾸거나, 커맨드라인 인자로 덮어쓰기) ──────────
CONFIG = {
    "d": 10,               # 맥락(입력) 차원
    "n_iters": 1500,       # Reptile 메타 반복
    "inner_steps": 10,     # 태스크당 RL 적응 스텝
    "inner_lr": 0.05,
    "meta_lr": 0.3,        # Reptile 보폭
    "batch": 32,
    "sigma": 0.3,          # 정책 탐험 표준편차
    "max_steps": 15,       # 평가 시 적응 스텝 수
    "n_test_tasks": 50,
    "seed": 0,
}


def main():
    cfg = build_config(CONFIG, "메타러닝+RL 하이브리드")
    d = cfg["d"]
    print("=" * 60)
    print(" 메타러닝 + 강화학습 하이브리드 (Reptile + REINFORCE)")
    print("=" * 60)
    print_config(cfg)
    print("\n  학습 중...")

    theta_rep = h.train_reptile_rl(
        d, cfg["n_iters"], cfg["inner_steps"], cfg["inner_lr"],
        cfg["meta_lr"], cfg["batch"], cfg["sigma"], seed=cfg["seed"])
    theta_joint = h.train_joint_rl(
        d, cfg["n_iters"] * cfg["inner_steps"], cfg["inner_lr"],
        cfg["batch"], cfg["sigma"], seed=cfg["seed"])
    theta_rand = h.init_params(np.random.default_rng(123), d)

    test_tasks = [h.BanditTask(np.random.default_rng(5000 + i), d)
                  for i in range(cfg["n_test_tasks"])]

    methods = {
        "Random": (theta_rand, "#999999"),
        "Joint (no meta)": (theta_joint, "#CCB974"),
        "Reptile + RL": (theta_rep, "#C44E52"),
    }
    curves = {}
    print(f"\n  [적응 곡선] 새 태스크에 RL 적응 ({cfg['max_steps']}스텝까지)")
    for name, (th, _) in methods.items():
        c = h.adaptation_curve(th, test_tasks, d, cfg["max_steps"],
                               cfg["inner_lr"], cfg["batch"], cfg["sigma"], seed=7)
        curves[name] = c
        print(f"     {name:16s}: 적응전 {c[0]:.3f} -> 최종 {c[-1]:.3f}")

    # 시각화 --------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    for name, (_, color) in methods.items():
        plt.plot(range(cfg["max_steps"] + 1), curves[name], "o-", color=color,
                 linewidth=2, markersize=4, label=name)
    plt.title("Meta-RL hybrid: Reptile init adapts fastest (contextual bandit)")
    plt.xlabel("RL adaptation steps on a new task")
    plt.ylabel("Performance  (-MSE, higher = better)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(OUTPUT_DIR, "30_meta_rl_hybrid.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 그래프 저장됨 -> {save_path}")
    print("\n  💡 해석: 메타러닝(Reptile)으로 만든 정책 초기값에서 시작하면, 새 RL")
    print("     태스크에 가장 빠르게 적응한다. 메타러닝+RL '결합'의 효과다.")
    print("     (같은 아우터에 SAC/DDPG 등 다른 RL을 끼워도 원리는 동일하다.)")


if __name__ == "__main__":
    main()

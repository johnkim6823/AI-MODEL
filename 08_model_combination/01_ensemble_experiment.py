"""
[모델 결합] 앙상블 실험 — 여러 모델을 합쳐 더 좋은 결과 ⭐
============================================================

"하나의 모델만 쓰지 말고, 여러 종류의 모델을 합치면 (특히 입력이 많고
복잡할 때) 더 좋고 안정적인 결과가 나온다"를 직접 확인한다.

▶ 핵심 메시지 (정직하게)
  - 앙상블은 '가장 좋은 단일 모델'과 대등하거나 조금 낫다.
  - 그런데 우리는 보통 '어느 모델이 최고일지 미리 모른다'. 앙상블은 그걸
    몰라도 항상 상위권 성능을 주고, '평균적인 단일 모델'보다 훨씬 낫다.
  - 입력 차원(특성 수)이 커질수록 단일 모델은 들쭉날쭉해지고,
    앙상블이 평균 단일 모델을 앞서는 폭이 커진다.

▶ 직접 값을 바꿔 실험 (config 또는 커맨드라인 둘 다 지원)
    python 01_ensemble_experiment.py --models logreg,tree,knn,svm,forest --ensemble stacking
    python 01_ensemble_experiment.py --n_features 120 --n_samples 1000 --seed 1
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from ensemble import Ensemble
from exp_config import build_config, print_config

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")

# ── 기본 설정 (여기 값을 바꾸거나, 커맨드라인 인자로 덮어쓰기) ──────────
CONFIG = {
    "models": ["logreg", "tree", "knn", "svm", "forest"],  # 합칠 모델들
    "ensemble": "stacking",   # voting | soft | stacking
    "n_samples": 900,
    "n_features": 50,         # 설정 실험에서 쓸 입력 차원
    "n_informative": 6,
    "flip_y": 0.03,           # 라벨 잡음
    "seed": 0,
}


def make_data(n_features, cfg):
    X, y = make_classification(
        n_samples=cfg["n_samples"], n_features=n_features,
        n_informative=cfg["n_informative"], n_redundant=2,
        n_classes=3, n_clusters_per_class=2,   # 복잡한 결정 경계(단일 모델로는 어려움)
        flip_y=cfg["flip_y"], random_state=cfg["seed"])
    return train_test_split(X, y, test_size=0.3, random_state=cfg["seed"], stratify=y)


def run_once(n_features, cfg):
    X_tr, X_te, y_tr, y_te = make_data(n_features, cfg)
    ens = Ensemble(cfg["models"], method=cfg["ensemble"]).fit(X_tr, y_tr)
    base = ens.base_scores(X_te, y_te)
    return base, ens.score(X_te, y_te)


def main():
    cfg = build_config(CONFIG, "앙상블 실험")
    print("=" * 60)
    print(" 앙상블 실험: 여러 모델 합치기")
    print("=" * 60)
    print_config(cfg)

    # 1) 설정한 차원에서 개별 vs 앙상블 -----------------------------------
    base, ens_acc = run_once(cfg["n_features"], cfg)
    print(f"\n  [입력 {cfg['n_features']}차원, 결합={cfg['ensemble']}]")
    for name, acc in base.items():
        print(f"     {name:8s}: {acc:.3f}")
    print(f"     {'앙상블':8s}: {ens_acc:.3f}  (최고단일 {max(base.values()):.3f}, "
          f"평균단일 {np.mean(list(base.values())):.3f})")

    # 2) 입력 차원을 늘려가며 비교 ----------------------------------------
    dims = [8, 20, 50, 100, 200]
    best_curve, mean_curve, ens_curve = [], [], []
    print("\n  [입력 차원 증가에 따른 비교]")
    for d in dims:
        b, e = run_once(d, cfg)
        best_curve.append(max(b.values()))
        mean_curve.append(np.mean(list(b.values())))
        ens_curve.append(e)
        print(f"     차원 {d:3d}: 앙상블 {e:.3f} | 최고단일 {max(b.values()):.3f} | "
              f"평균단일 {np.mean(list(b.values())):.3f} | (앙상블-평균 {e-np.mean(list(b.values())):+.3f})")

    # 3) 시각화 -----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    names = list(base.keys()) + ["ENSEMBLE"]
    vals = list(base.values()) + [ens_acc]
    colors = ["#4C72B0"] * len(base) + ["#C44E52"]
    bars = axes[0].bar(names, vals, color=colors)
    for bar, v in zip(bars, vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2, v + 0.005, f"{v:.3f}",
                     ha="center", va="bottom", fontsize=9)
    axes[0].set_title(f"Individual models vs Ensemble ({cfg['n_features']}-dim, {cfg['ensemble']})")
    axes[0].set_ylabel("Test Accuracy")
    axes[0].set_ylim(min(vals) - 0.06, 1.0)
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].plot(dims, ens_curve, "o-", color="#C44E52", linewidth=2.5, label="ensemble")
    axes[1].plot(dims, best_curve, "o--", color="#4C72B0", linewidth=2, label="best single")
    axes[1].plot(dims, mean_curve, "o--", color="#999999", linewidth=2, label="average single")
    axes[1].fill_between(dims, mean_curve, ens_curve, color="orange", alpha=0.2,
                         label="ensemble vs average gap")
    axes[1].set_title("Larger input -> bigger ensemble advantage")
    axes[1].set_xlabel("Number of input features")
    axes[1].set_ylabel("Test Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Ensemble: combining models for better, more robust results", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = os.path.join(OUTPUT_DIR, "29_ensemble_experiment.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 그래프 저장됨 -> {save_path}")
    print("\n  💡 해석: 앙상블은 최고 단일 모델과 대등하면서, '평균적인 단일 모델'보다")
    print("     훨씬 낫다. 입력 차원이 커질수록(주황 영역) 그 이점이 더 벌어진다.")


if __name__ == "__main__":
    main()

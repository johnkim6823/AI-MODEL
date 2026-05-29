"""
[비지도학습 - 차원 축소] 주성분 분석 (PCA)
============================================

▶ 차원 축소(Dimensionality Reduction)란?
  - 특성(feature)이 너무 많을 때, 중요한 정보는 최대한 살리면서
    특성 개수를 줄이는 것.
  - 장점: 시각화 가능(2~3D), 계산 속도 향상, 잡음 제거

▶ PCA (Principal Component Analysis, 주성분 분석)
  - 데이터가 가장 많이 퍼져 있는(=정보가 많은) 방향을 찾아
    그 방향(주성분)으로 데이터를 새로 표현한다.
  - 4차원 데이터를 2차원으로 줄여서 그래프로 볼 수 있게 한다.

▶ 사용 데이터: 붓꽃(Iris) 데이터
  - 꽃잎/꽃받침 길이·너비 4개 특성 → 3종류 꽃 분류용 데이터
  - PCA로 4차원을 2차원으로 줄여 시각화한다.
  - (라벨은 색칠 확인용으로만 사용. PCA 자체는 라벨을 쓰지 않는 비지도학습!)
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def main():
    # 1) 데이터 로드 ------------------------------------------------------
    iris = load_iris()
    X = iris.data                 # (150, 4) : 특성 4개
    y = iris.target               # 꽃 종류 (시각화 색칠용으로만 사용)
    target_names = iris.target_names

    # 2) 표준화 (PCA 전 필수에 가까운 단계) ------------------------------
    #    특성마다 단위/범위가 다르므로 평균0·분산1로 맞춰준다.
    X_scaled = StandardScaler().fit_transform(X)

    # 3) PCA로 4차원 -> 2차원 --------------------------------------------
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)

    # 각 주성분이 원본 정보를 얼마나 보존하는지 (설명된 분산 비율)
    explained = pca.explained_variance_ratio_

    print("=" * 55)
    print(" PCA 차원 축소 결과 (4차원 -> 2차원)")
    print("=" * 55)
    print(f"  주성분1(PC1)이 담은 정보량: {explained[0]*100:.1f}%")
    print(f"  주성분2(PC2)이 담은 정보량: {explained[1]*100:.1f}%")
    print(f"  2개 합계: {explained.sum()*100:.1f}% (이만큼의 정보를 2D로 유지)")

    # 4) 시각화: (왼쪽) 2D 산점도, (오른쪽) 정보 보존 비율 ----------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # (왼쪽) PCA 2D 투영 결과 (종류별 색)
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    for i, name in enumerate(target_names):
        axes[0].scatter(X_2d[y == i, 0], X_2d[y == i, 1],
                        color=colors[i], s=40, alpha=0.8, label=name)
    axes[0].set_title("Iris projected to 2D by PCA")
    axes[0].set_xlabel(f"PC1 ({explained[0]*100:.1f}%)")
    axes[0].set_ylabel(f"PC2 ({explained[1]*100:.1f}%)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # (오른쪽) 각 주성분이 보존한 정보량 + 누적
    all_pca = PCA().fit(X_scaled)
    ratios = all_pca.explained_variance_ratio_
    pcs = [f"PC{i+1}" for i in range(len(ratios))]
    axes[1].bar(pcs, ratios, color="#4C72B0", alpha=0.7, label="each")
    axes[1].plot(pcs, np.cumsum(ratios), "o-", color="#C44E52",
                 label="cumulative")
    axes[1].set_title("Explained Variance by Component")
    axes[1].set_ylabel("Explained Variance Ratio")
    axes[1].legend()
    axes[1].grid(True, axis="y", alpha=0.3)

    save_path = os.path.join(OUTPUT_DIR, "05_pca_dimensionality.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 그래프 저장됨 -> {save_path}")


if __name__ == "__main__":
    main()

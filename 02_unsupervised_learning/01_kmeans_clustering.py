"""
[비지도학습 - 군집화] K-평균 (K-Means Clustering)
==================================================

▶ 비지도학습(Unsupervised Learning)이란?
  - "정답(label)이 없는" 데이터에서 숨은 구조/패턴을 찾는 것.
  - 지도학습과 달리 "맞다/틀리다"를 알려주는 정답이 없다.

▶ 군집화(Clustering)
  - 비슷한 데이터끼리 그룹(cluster)으로 묶는 것.
  - 예: 고객을 구매 패턴에 따라 자동으로 그룹화

▶ K-Means 동작 방식 (직관)
  1. 그룹 개수 k를 정한다.
  2. 임의로 중심점 k개를 찍는다.
  3. 각 점을 가장 가까운 중심점에 배정한다.
  4. 각 그룹의 평균 위치로 중심점을 옮긴다.
  5. 3~4를 변화가 없을 때까지 반복한다.

▶ "k는 몇 개가 좋을까?" → 엘보우(Elbow) 기법으로 찾는다.
  - 그룹 수를 늘려가며 'inertia(응집도)'를 측정.
  - 그래프가 팔꿈치처럼 꺾이는 지점이 적절한 k.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def main():
    # 1) 데이터 생성 (정답 라벨은 일부러 사용하지 않는다!) ----------------
    #    실제로는 4개 덩어리지만, 모델에게는 알려주지 않는다.
    X, _ = make_blobs(n_samples=400, centers=4, cluster_std=0.9, random_state=42)

    # 2) 엘보우 기법으로 적절한 k 탐색 -----------------------------------
    k_range = range(1, 9)
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)  # inertia: 각 점과 중심점 거리 제곱의 합

    # 3) 적절한 k로 최종 군집화 ------------------------------------------
    best_k = 4
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_

    print("=" * 55)
    print(" K-Means 군집화 결과")
    print("=" * 55)
    print(f"  사용한 그룹 수 k = {best_k}")
    print(f"  각 그룹의 데이터 개수: {np.bincount(labels)}")
    print(f"  최종 inertia(응집도, 작을수록 조밀): {kmeans.inertia_:.1f}")

    # 4) 시각화: (왼쪽) 엘보우 그래프, (오른쪽) 군집 결과 ------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # (왼쪽) 엘보우 그래프
    axes[0].plot(list(k_range), inertias, "o-", color="#4C72B0", linewidth=2)
    axes[0].axvline(best_k, color="#C44E52", linestyle="--",
                    label=f"chosen k = {best_k}")
    axes[0].set_title("Elbow Method: choosing k")
    axes[0].set_xlabel("Number of clusters (k)")
    axes[0].set_ylabel("Inertia (lower = tighter)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # (오른쪽) 군집화 결과 + 중심점
    scatter = axes[1].scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis",
                              s=30, alpha=0.7)
    axes[1].scatter(centers[:, 0], centers[:, 1], c="red", marker="X",
                    s=250, edgecolor="k", label="Centroids")
    axes[1].set_title(f"K-Means Result (k={best_k})")
    axes[1].set_xlabel("Feature 1")
    axes[1].set_ylabel("Feature 2")
    axes[1].legend()

    save_path = os.path.join(OUTPUT_DIR, "04_kmeans_clustering.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 그래프 저장됨 -> {save_path}")


if __name__ == "__main__":
    main()

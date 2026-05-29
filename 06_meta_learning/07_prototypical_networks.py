"""
[메타러닝] Prototypical Networks — 거리로 분류하는 few-shot ⭐
==============================================================

▶ 핵심 아이디어
  - 입력을 '임베딩 공간'으로 보내는 신경망을 학습한다.
  - 각 클래스의 서포트 예시들의 임베딩 '평균'을 그 클래스의 '프로토타입'으로.
  - 쿼리는 '가장 가까운 프로토타입'의 클래스로 분류한다 (유클리드 거리).
  - 학습에 쓰지 않은 '새 클래스'도 예시 몇 개로 프로토타입만 만들면 바로 분류!

▶ 손글씨 숫자로 3-way 5-shot
  - 학습 클래스: 0~5,  테스트(새) 클래스: 6~9 (학습에 안 본 숫자)

▶ 결과
  (왼쪽) 학습 곡선 (에피소드 정확도)
  (오른쪽) 새 클래스 한 태스크의 임베딩을 2D(PCA)로 투영
          → 같은 클래스끼리 뭉치고, 쿼리는 가까운 프로토타입으로 분류됨
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

import metric_core as mc

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def main():
    train_tasks = mc.FewShotTasks(classes=[0, 1, 2, 3, 4, 5])
    test_tasks = mc.FewShotTasks(classes=[6, 7, 8, 9])
    n_way, k_shot = 3, 5

    print("=" * 55)
    print(" Prototypical Networks (3-way 5-shot, 새 클래스 6~9)")
    print("=" * 55)
    embedder, hist = mc.train(mc.protonet_loss, train_tasks, n_way=n_way,
                              k_shot=k_shot, q_query=5, n_iters=1000, seed=0)
    acc = mc.evaluate(embedder, mc.protonet_loss, test_tasks, n_way=n_way,
                      k_shot=k_shot, q_query=5, n_tasks=300)
    print(f"  새 클래스 few-shot 정확도: {acc:.3f}  (3-way 무작위 추측 = 0.33)")

    # 새 클래스 한 태스크의 임베딩을 PCA로 2D 투영 -----------------------
    rng = np.random.default_rng(123)
    Xs, ys, Xq, yq = test_tasks.sample(n_way, k_shot, 12, rng)
    n_s = Xs.shape[0]
    emb_all = embedder.forward(np.vstack([Xs, Xq]))
    emb_s, emb_q = emb_all[:n_s], emb_all[n_s:]
    protos = np.stack([emb_s[ys == k].mean(axis=0) for k in range(n_way)])

    pca = PCA(n_components=2).fit(emb_all)
    q2d = pca.transform(emb_q)
    p2d = pca.transform(protos)

    # 시각화 --------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sm = mc.moving_average(hist, 50)
    axes[0].plot(range(len(sm)), sm, color="#55A868", linewidth=2)
    axes[0].set_title("ProtoNet training accuracy (episodes)")
    axes[0].set_xlabel("Meta-iteration")
    axes[0].set_ylabel("Query accuracy (smoothed)")
    axes[0].set_ylim(0, 1.02)
    axes[0].grid(True, alpha=0.3)

    colors = ["#4C72B0", "#C44E52", "#55A868"]
    for k in range(n_way):
        axes[1].scatter(q2d[yq == k, 0], q2d[yq == k, 1], color=colors[k],
                        s=35, alpha=0.7, label=f"query class {k}")
        axes[1].scatter(p2d[k, 0], p2d[k, 1], color=colors[k], marker="*",
                        s=400, edgecolor="k", zorder=5)
    axes[1].set_title(f"Embedding space (PCA) — novel classes (acc={acc:.2f})")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].legend(fontsize=9)
    axes[1].text(0.02, 0.98, "★ = class prototype", transform=axes[1].transAxes,
                 fontsize=10, va="top")

    fig.suptitle("Prototypical Networks: classify by nearest prototype", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = os.path.join(OUTPUT_DIR, "24_prototypical_networks.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 그래프 저장됨 -> {save_path}")
    print("\n  💡 해석:")
    print("     - 학습 클래스로 '임베딩 만드는 법'을 배운다.")
    print("     - 새 클래스도 임베딩하면 같은 클래스끼리 뭉쳐, 가까운 프로토타입으로")
    print("       분류된다(무작위 0.33보다 훨씬 높음). 경사하강 적응이 필요 없다.")


if __name__ == "__main__":
    main()

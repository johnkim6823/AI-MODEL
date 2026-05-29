"""
[메타러닝] Matching Networks — 유사도 가중 투표 few-shot ⭐
============================================================

▶ 핵심 아이디어 (Prototypical Networks와 비교)
  - ProtoNet: 클래스 '평균(프로토타입)'까지의 거리로 분류.
  - Matching : 쿼리와 '각 서포트 예시'의 코사인 유사도로 어텐션(주목도)을 만들고,
               유사한 예시일수록 그 예시의 라벨에 더 많이 투표한다.
               (일종의 '학습된 가중 최근접 이웃')

▶ 손글씨 숫자로 3-way 5-shot (학습 0~5 / 테스트 새 클래스 6~9)

▶ 결과
  (왼쪽) 어텐션 히트맵: 한 테스트 태스크에서 쿼리(행)가 어떤 서포트(열)에
        주목하는지 → 같은 클래스 블록이 밝다(같은 클래스끼리 주목).
  (오른쪽) 새 클래스 정확도: Matching vs ProtoNet 비교
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import metric_core as mc

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def main():
    train_tasks = mc.FewShotTasks(classes=[0, 1, 2, 3, 4, 5])
    test_tasks = mc.FewShotTasks(classes=[6, 7, 8, 9])
    n_way, k_shot = 3, 5

    print("=" * 55)
    print(" Matching Networks (3-way 5-shot, 새 클래스 6~9)")
    print("=" * 55)

    # Matching Networks 와 ProtoNet 둘 다 학습해 비교
    emb_match, _ = mc.train(mc.matching_loss, train_tasks, n_way=n_way,
                            k_shot=k_shot, q_query=5, n_iters=1000, seed=0)
    emb_proto, _ = mc.train(mc.protonet_loss, train_tasks, n_way=n_way,
                            k_shot=k_shot, q_query=5, n_iters=1000, seed=0)

    acc_match = mc.evaluate(emb_match, mc.matching_loss, test_tasks, n_way,
                            k_shot, 5, n_tasks=300)
    acc_proto = mc.evaluate(emb_proto, mc.protonet_loss, test_tasks, n_way,
                            k_shot, 5, n_tasks=300)
    print(f"  새 클래스 정확도 — Matching {acc_match:.3f}  vs  ProtoNet {acc_proto:.3f}")
    print(f"  (3-way 무작위 추측 = 0.33)")

    # 어텐션 히트맵용: 한 태스크 (클래스별로 정렬) -----------------------
    rng = np.random.default_rng(7)
    Xs, ys, Xq, yq = test_tasks.sample(n_way, k_shot, 3, rng)
    n_s = Xs.shape[0]
    emb_all = emb_match.forward(np.vstack([Xs, Xq]))
    emb_s, emb_q = emb_all[:n_s], emb_all[n_s:]
    Sn = emb_s / (np.linalg.norm(emb_s, axis=1, keepdims=True) + 1e-8)
    Qn = emb_q / (np.linalg.norm(emb_q, axis=1, keepdims=True) + 1e-8)
    attn = mc._softmax(Qn @ Sn.T, axis=1)        # (Nq, Ns)

    # 시각화 --------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    im = axes[0].imshow(attn, cmap="viridis", aspect="auto")
    axes[0].set_title("Attention: query (rows) -> support (cols)")
    axes[0].set_xlabel(f"support examples (grouped by class, {k_shot} each)")
    axes[0].set_ylabel("query examples (grouped by class)")
    # 클래스 경계선
    for b in range(1, n_way):
        axes[0].axvline(b * k_shot - 0.5, color="white", linewidth=1)
        axes[0].axhline(b * 3 - 0.5, color="white", linewidth=1)
    fig.colorbar(im, ax=axes[0], fraction=0.046, label="attention weight")

    bars = axes[1].bar(["Matching\nNetworks", "Prototypical\nNetworks"],
                       [acc_match, acc_proto], color=["#8172B3", "#55A868"])
    axes[1].axhline(1.0 / n_way, color="#C44E52", linestyle="--",
                    label="random guess (0.33)")
    for bar, a in zip(bars, [acc_match, acc_proto]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, a + 0.01,
                     f"{a:.3f}", ha="center", va="bottom", fontsize=11)
    axes[1].set_title("Novel-class few-shot accuracy")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.02)
    axes[1].legend()

    fig.suptitle("Matching Networks: similarity-weighted voting", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = os.path.join(OUTPUT_DIR, "25_matching_networks.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 그래프 저장됨 -> {save_path}")
    print("\n  💡 해석:")
    print("     - 어텐션 히트맵의 '같은 클래스 블록'이 밝다 = 쿼리가 같은 클래스")
    print("       서포트에 더 주목해 그 라벨로 분류한다.")
    print("     - 두 방법 모두 무작위(0.33)보다 훨씬 잘 맞힌다.")


if __name__ == "__main__":
    main()

"""
모든 예제를 한 번에 실행하는 스크립트
======================================

사용법:
    python run_all.py

각 예제를 순서대로 실행하고, 그래프는 모두 outputs/ 폴더에 저장된다.
"""

import os
import runpy
import sys

# 실행할 스크립트 목록 (폴더, 파일명) — 학습 순서대로
SCRIPTS = [
    ("01_supervised_learning", "01_linear_regression.py"),
    ("01_supervised_learning", "02_logistic_regression.py"),
    ("01_supervised_learning", "03_model_comparison.py"),
    ("02_unsupervised_learning", "01_kmeans_clustering.py"),
    ("02_unsupervised_learning", "02_pca_dimensionality.py"),
    ("03_hyperparameter_experiments", "00_gradient_descent_intuition.py"),
    ("03_hyperparameter_experiments", "01_learning_rate_experiment.py"),
    ("03_hyperparameter_experiments", "02_epochs_experiment.py"),
    ("03_hyperparameter_experiments", "03_grid_search_heatmap.py"),
]

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    for folder, filename in SCRIPTS:
        path = os.path.join(ROOT, folder, filename)
        print("\n" + "#" * 65)
        print(f"# 실행: {folder}/{filename}")
        print("#" * 65)

        # 각 스크립트가 같은 폴더의 model.py 등을 import 할 수 있도록
        # 해당 폴더를 import 경로(sys.path) 맨 앞에 넣어준다.
        script_dir = os.path.join(ROOT, folder)
        sys.path.insert(0, script_dir)
        try:
            runpy.run_path(path, run_name="__main__")
        except Exception as e:  # 한 스크립트가 실패해도 나머지는 계속 실행
            print(f"  ⚠️  실행 중 오류: {e}")
        finally:
            sys.path.remove(script_dir)

    print("\n" + "=" * 65)
    print(" ✅ 전체 실행 완료! outputs/ 폴더에서 그래프를 확인하세요.")
    print("=" * 65)


if __name__ == "__main__":
    main()

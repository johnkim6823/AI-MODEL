"""
실험 설정 헬퍼: config 딕셔너리 + 커맨드라인 인자 (둘 다 지원)
=============================================================

각 실험 스크립트는 상단에 기본 CONFIG 딕셔너리를 둔다.
  - 그냥 실행하면 CONFIG 기본값 사용.
  - 또는 커맨드라인으로 값을 덮어쓸 수 있다.
      예) python 01_ensemble_experiment.py --n_features 80 --ensemble stacking

사용법:
    from exp_config import build_config
    CONFIG = {"n_features": 30, "models": ["logreg", "tree"], "ensemble": "soft"}
    cfg = build_config(CONFIG, "앙상블 실험")
    print(cfg["n_features"])
"""

import argparse


def build_config(defaults, description=""):
    """기본값 딕셔너리로부터 커맨드라인 파서를 자동 생성하고, 병합된 설정을 반환."""
    p = argparse.ArgumentParser(description=description)
    for key, val in defaults.items():
        if isinstance(val, bool):
            p.add_argument(f"--{key}", default=val,
                           type=lambda x: str(x).lower() in ("1", "true", "yes", "y"))
        elif isinstance(val, list):
            # 리스트는 콤마로 구분된 문자열로 받는다. 예) --models logreg,tree,svm
            p.add_argument(f"--{key}", default=",".join(map(str, val)), type=str)
        else:
            p.add_argument(f"--{key}", default=val, type=type(val))

    args, _ = p.parse_known_args()   # 알 수 없는 인자는 무시 (run_all 등에서 안전)
    cfg = dict(defaults)
    for key, val in defaults.items():
        new = getattr(args, key)
        if isinstance(val, list) and isinstance(new, str):
            new = [x.strip() for x in new.split(",") if x.strip()]
        cfg[key] = new
    return cfg


def print_config(cfg):
    print("  [설정값]")
    for k, v in cfg.items():
        print(f"    {k} = {v}")

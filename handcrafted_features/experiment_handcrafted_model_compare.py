# -*- coding: utf-8 -*-
"""
在「CGM+BGM合併訓練、只用BGM評估」Normal vs High（全時段，使用者指示先不處理晝夜
confound）上，比較幾種模型/超參數設定，並用5個不同的fold切分seed檢查穩健度。

比較對象：
  logreg_C0.01 / logreg_C0.1(目前用的) / logreg_C1.0 : LogisticRegression 三種正規化強度
  random_forest                                        : RandomForestClassifier
  xgboost                                              : XGBClassifier
  svm_C0.01 / svm_C0.1 / svm_C1.0                      : SVC(kernel="rbf") 三種正規化強度

每個模型 x 每個seed，都跑一次完整的5-fold merged訓練/BGM評估流程(raw + centered_6兩種)，
記錄per-subject AUC。5個seed算mean/std，看結果對「切哪5折」穩不穩。

**重要提醒（如果之後要用這個結果做決策）**：n=18人，多個模型/設定一起比較，本身就
有「多重比較挑到運氣好的設定」的風險——結果僅供參考誰的方向合理，不要只因為某個
設定的mean AUC最高0.02~0.03就直接採用它,要連同std/seed間的變動一起看。

用法：
  conda run -n ml python kfold/handcrafted_features/experiment_handcrafted_model_compare.py
"""

import json
import os
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

_KFOLD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in sorted(os.listdir(_KFOLD_ROOT)):
    _p = os.path.join(_KFOLD_ROOT, _d)
    if os.path.isdir(_p):
        sys.path.insert(0, _p)

from experiment_handcrafted_bgm import load_merged, OUT_DIR  # noqa: E402
from experiment_handcrafted_classification import (  # noqa: E402
    K_FOLDS, N_CALIB_EVENTS, per_subject_auc, center_by_subject,
)

POS_LEVEL, NEG_LEVELS = "High", ["Normal"]
CALIB_SEED = 42  # 校正抽樣固定，只變動fold切分seed，這樣才是單純比較「切哪5折」的影響
SPLIT_SEEDS = [1, 2, 3, 4, 5]
OUT_PATH = os.path.join(OUT_DIR, "model_compare_results.json")


def make_model(name):
    if name == "logreg_C0.01":
        return LogisticRegression(max_iter=3000, class_weight="balanced", C=0.01)
    if name == "logreg_C0.1":
        return LogisticRegression(max_iter=3000, class_weight="balanced", C=0.1)
    if name == "logreg_C1.0":
        return LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=300, max_depth=6, class_weight="balanced",
                                       random_state=0, n_jobs=1)
    if name == "xgboost":
        return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                              eval_metric="logloss", n_jobs=1, random_state=0)
    if name == "svm_C0.01":
        return SVC(C=0.01, kernel="rbf", class_weight="balanced")
    if name == "svm_C0.1":
        return SVC(C=0.1, kernel="rbf", class_weight="balanced")
    if name == "svm_C1.0":
        return SVC(C=1.0, kernel="rbf", class_weight="balanced")
    raise ValueError(name)


def decision_score(clf, X):
    """統一介面：LogisticRegression有decision_function，樹模型用predict_proba取正類機率。"""
    if hasattr(clf, "decision_function"):
        return clf.decision_function(X)
    return clf.predict_proba(X)[:, 1]


def run_one(data, feat_cols, model_name, split_seed):
    sub = data[data["BG_Level"].isin([POS_LEVEL] + NEG_LEVELS)].reset_index(drop=True)
    X = sub[feat_cols].to_numpy(dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = (sub["BG_Level"] == POS_LEVEL).astype(int).to_numpy()
    subj = sub["uuid"].to_numpy()
    src = sub["source"].to_numpy()
    uuids = np.array(sorted(set(subj)))

    rng = np.random.RandomState(split_seed)
    order = rng.permutation(len(uuids))
    folds = {u: i % K_FOLDS for i, u in enumerate(uuids[order])}

    rng2 = np.random.RandomState(CALIB_SEED)
    calib_mask = np.zeros(len(y), bool)
    for u in uuids:
        pool = np.where(subj == u)[0]
        pool_bgm = pool[src[pool] == "BGM"]
        pool = pool_bgm if len(pool_bgm) > 0 else pool
        pick = rng2.choice(pool, size=min(N_CALIB_EVENTS, len(pool)), replace=False)
        calib_mask[pick] = True

    modes = ["raw", "centered_6"]
    scores = {m: np.full(len(y), np.nan) for m in modes}
    done = np.zeros(len(y), bool)

    for f in range(K_FOLDS):
        te_u = {u for u in uuids if folds[u] == f}
        tr = ~np.isin(subj, list(te_u))
        te = np.isin(subj, list(te_u)) & (src == "BGM")
        if te.sum() == 0:
            continue
        for mode in modes:
            if mode == "raw":
                Xtr, Xte = X[tr], X[te]
            else:
                Xc = X.copy()
                Xc[tr] = center_by_subject(X[tr], subj[tr])
                Xc[te] = center_by_subject(X[te], subj[te], ref_idx=calib_mask[te])
                Xtr, Xte = Xc[tr], Xc[te]
            sc = StandardScaler().fit(Xtr)
            clf = make_model(model_name)
            clf.fit(sc.transform(Xtr), y[tr])
            scores[mode][te] = decision_score(clf, sc.transform(Xte))
        done[te] = True

    out = {}
    for mode in modes:
        ps, n, _ = per_subject_auc(y[done], scores[mode][done], subj[done])
        out[mode] = {"per_subject_auc": ps, "n_subjects": n}
    return out


def main():
    print("讀取合併資料...")
    data, feat_cols, _ = load_merged()

    model_names = ["logreg_C0.01", "logreg_C0.1", "logreg_C1.0", "random_forest", "xgboost",
                   "svm_C0.01", "svm_C0.1", "svm_C1.0"]
    results = {m: {"raw": [], "centered_6": []} for m in model_names}

    for model_name in model_names:
        print(f"\n{'='*70}\n{model_name}\n{'='*70}")
        for seed in SPLIT_SEEDS:
            r = run_one(data, feat_cols, model_name, seed)
            results[model_name]["raw"].append(r["raw"]["per_subject_auc"])
            results[model_name]["centered_6"].append(r["centered_6"]["per_subject_auc"])
            print(f"  seed={seed}: raw AUC={r['raw']['per_subject_auc']:.4f}  "
                  f"centered_6 AUC={r['centered_6']['per_subject_auc']:.4f}  "
                  f"(n={r['raw']['n_subjects']})")

    print(f"\n{'='*90}\n跨 {len(SPLIT_SEEDS)} 個fold切分seed的穩健度總表\n{'='*90}")
    print(f"{'模型':16}{'raw mean±std':>20}{'raw range':>18}{'centered_6 mean±std':>24}{'centered_6 range':>20}")
    summary = {}
    for m in model_names:
        raw_v = np.array(results[m]["raw"])
        c6_v = np.array(results[m]["centered_6"])
        summary[m] = {
            "raw_mean": float(raw_v.mean()), "raw_std": float(raw_v.std()),
            "raw_min": float(raw_v.min()), "raw_max": float(raw_v.max()),
            "centered_6_mean": float(c6_v.mean()), "centered_6_std": float(c6_v.std()),
            "centered_6_min": float(c6_v.min()), "centered_6_max": float(c6_v.max()),
            "raw_values": raw_v.tolist(), "centered_6_values": c6_v.tolist(),
        }
        s = summary[m]
        raw_mean_std = f"{s['raw_mean']:.4f}±{s['raw_std']:.4f}"
        raw_range = f"[{s['raw_min']:.3f},{s['raw_max']:.3f}]"
        c6_mean_std = f"{s['centered_6_mean']:.4f}±{s['centered_6_std']:.4f}"
        c6_range = f"[{s['centered_6_min']:.3f},{s['centered_6_max']:.3f}]"
        print(f"{m:16}{raw_mean_std:>20}{raw_range:>18}{c6_mean_std:>24}{c6_range:>20}")

    with open(OUT_PATH, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
    print(f"\n寫入 {OUT_PATH}")


if __name__ == "__main__":
    main()

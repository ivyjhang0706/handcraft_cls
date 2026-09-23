# -*- coding: utf-8 -*-
"""
測試「校正事件的次數」對績效的影響——維持隨機挑選(現況策略、BGM優先)，只改
6 個變成幾個。

動機：experiment_calib_composition.py 已經確認「改挑哪 6 個」沒有幫助(改挑 Normal
反而稍微變差)，但那支腳本刻意「不動校正次數」。既然 centering 是目前唯一有效的槓桿，
自然要問：基準線用更多筆事件估，會不會更準。

三個設定：
  centered_6（現況）
  centered_12
  centered_18

訓練端跟 calib_composition 的「random（現況）」策略一致：用該人全部事件算基準線，
只有測試端的校正事件數量改變。

評估上的處理跟 calib_composition 一樣：排除掉三種設定校正事件的聯集，讓三個數字都在
完全相同的事件集合上被評分才能配對比較。代價是這個數字不能直接跟現有的 0.636 比
(評估集合變小)，所以另外附上「centered_6 + 不排除校正事件」對照現況。

注意：某些受試者的 BGM 事件數可能不到 12 或 18 筆，這種情況會直接把該人「全部」BGM
事件都當校正事件(跟 make_calib 的 min(n_calib, len(pool)) 邏輯一致)，可能讓
可評估人數隨 n_calib 增加而減少，執行時要看印出來的人數診斷。

用法：
  conda run -n ml python kfold/handcrafted_features/experiment_calib_count.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

_KFOLD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in sorted(os.listdir(_KFOLD_ROOT)):
    _p = os.path.join(_KFOLD_ROOT, _d)
    if os.path.isdir(_p):
        sys.path.insert(0, _p)

from experiment_handcrafted_bgm import load_merged  # noqa: E402
from experiment_handcrafted_classification import (  # noqa: E402
    K_FOLDS, SEED, MIN_PER_SIDE, center_by_subject, pick_threshold, OUT_DIR,
)
from binary_metrics import binary_metrics  # noqa: E402

POS_LEVEL, NEG_LEVELS = "High", ["Normal"]
RF_PARAMS = dict(n_estimators=300, max_depth=6, class_weight="balanced",
                 random_state=0, n_jobs=8)
N_BOOT = 10000
CALIB_COUNTS = [6, 12, 18]


def make_calib(subj, src, uuids, n_calib):
    """挑每個人的 n_calib 個校正事件（隨機、BGM優先，跟現況抽樣邏輯一致）。"""
    rng2 = np.random.RandomState(SEED)
    calib = np.zeros(len(subj), bool)
    for u in uuids:
        pool = np.where(subj == u)[0]
        pool_bgm = pool[src[pool] == "BGM"]
        pool = pool_bgm if len(pool_bgm) > 0 else pool
        pick = rng2.choice(pool, size=min(n_calib, len(pool)), replace=False)
        calib[pick] = True
    return calib


def run_strategy(X, y, subj, src, uuids, folds, calib):
    """跑 5 折 centered_N：訓練端維持現況(用該人全部事件算基準線)，
    測試端用 calib 指定的那批事件當校正參考。"""
    score = np.full(len(y), np.nan)
    thr_all = np.full(len(y), np.nan)
    for f in range(K_FOLDS):
        te_u = [u for u in uuids if folds[u] == f]
        tr = ~np.isin(subj, te_u)
        te = np.isin(subj, te_u) & (src == "BGM")   # 只評估 BGM(部署現實)
        if te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue

        Xc = X.copy()
        Xc[tr] = center_by_subject(X[tr], subj[tr])
        Xc[te] = center_by_subject(X[te], subj[te], ref_idx=calib[te])

        sc = StandardScaler().fit(Xc[tr])
        clf = RandomForestClassifier(**RF_PARAMS)
        clf.fit(sc.transform(Xc[tr]), y[tr])
        thr = pick_threshold(clf.predict_proba(sc.transform(Xc[tr]))[:, 1], y[tr])
        score[te] = clf.predict_proba(sc.transform(Xc[te]))[:, 1]
        thr_all[te] = thr
    return score, thr_all


def run_raw(X, y, subj, src, uuids, folds):
    """不做 centering 的對照(跟校正次數無關，僅供參考)。"""
    score = np.full(len(y), np.nan)
    thr_all = np.full(len(y), np.nan)
    for f in range(K_FOLDS):
        te_u = [u for u in uuids if folds[u] == f]
        tr = ~np.isin(subj, te_u)
        te = np.isin(subj, te_u) & (src == "BGM")
        if te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        clf = RandomForestClassifier(**RF_PARAMS)
        clf.fit(sc.transform(X[tr]), y[tr])
        thr_all[te] = pick_threshold(clf.predict_proba(sc.transform(X[tr]))[:, 1], y[tr])
        score[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return score, thr_all


def per_subject_metrics(y, subj, score, thr, eval_mask):
    """事件層級的 per-subject 指標，只看 eval_mask 為 True 的事件。"""
    done = (~np.isnan(score)) & eval_mask
    out = {}
    for u in np.unique(subj[done]):
        m = done & (subj == u)
        if min((y[m] == 0).sum(), (y[m] == 1).sum()) < MIN_PER_SIDE:
            continue
        out[u] = binary_metrics(y[m], score[m], threshold=float(thr[m][0]))
    return out


def paired_bootstrap(base, new, key="auc"):
    common = sorted(set(base) & set(new))
    d = np.array([new[u][key] - base[u][key] for u in common
                  if base[u][key] is not None and new[u][key] is not None])
    if len(d) == 0:
        return None
    rng = np.random.RandomState(0)
    boot = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(N_BOOT)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mean_diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "n": int(len(d)), "significant": bool(lo > 0 or hi < 0)}


def main():
    out_dir = os.path.join(OUT_DIR, "calib_count")
    os.makedirs(out_dir, exist_ok=True)

    ev_all, feat_cols, _ = load_merged()
    ev = ev_all[ev_all["BG_Level"].isin([POS_LEVEL] + NEG_LEVELS)].reset_index(drop=True)
    X = np.nan_to_num(ev[feat_cols].to_numpy(dtype=np.float64),
                      nan=0.0, posinf=0.0, neginf=0.0)
    y = (ev["BG_Level"] == POS_LEVEL).astype(int).to_numpy()
    subj = ev["uuid"].to_numpy()
    src = ev["source"].to_numpy()
    uuids = np.array(sorted(set(subj)))
    print(f"事件數={len(y):,}  正類(High)={y.sum():,}  人數={len(uuids)}")

    rng = np.random.RandomState(SEED)
    order = rng.permutation(len(uuids))
    folds = {u: i % K_FOLDS for i, u in enumerate(uuids[order])}

    calibs = {n: make_calib(subj, src, uuids, n) for n in CALIB_COUNTS}
    for n in CALIB_COUNTS:
        c = calibs[n]
        print(f"n_calib={n:>2}：校正事件 {c.sum()} 個，其中 High {int(y[c].sum())} 個"
              f"（{y[c].mean():.1%} 被污染）")

    # 排除三種校正次數設定的聯集，讓三個數字在同一批事件上被評分
    excl = np.zeros(len(y), bool)
    for c in calibs.values():
        excl |= c
    eval_common = ~excl
    eval_full = np.ones(len(y), bool)
    print(f"排除校正事件聯集 {excl.sum()} 個後，可評估事件 {eval_common.sum():,}/{len(y):,}")

    print("\n跑 raw（無 centering，對照用）...")
    s_raw, t_raw = run_raw(X, y, subj, src, uuids, folds)

    scores = {}
    for n in CALIB_COUNTS:
        print(f"跑 centered_{n} ...")
        scores[n] = run_strategy(X, y, subj, src, uuids, folds, calibs[n])

    rows, metrics = [], {}

    def add(label, score, thr, mask, tag):
        m = per_subject_metrics(y, subj, score, thr, mask)
        metrics[f"{label}|{tag}"] = m
        vals = [m[u]["auc"] for u in m if m[u]["auc"] is not None]
        rec = [m[u]["recall"] for u in m if m[u]["recall"] is not None]
        spe = [m[u]["specificity"] for u in m if m[u]["specificity"] is not None]
        rows.append({"設定": label, "評估集": tag, "人數": len(m),
                     "AUC": float(np.mean(vals)) if vals else None,
                     "recall": float(np.mean(rec)) if rec else None,
                     "specificity": float(np.mean(spe)) if spe else None})

    # 對照現況：centered_6 + 不排除校正事件(可與既有的 0.636 對照)
    add("raw（無centering）", s_raw, t_raw, eval_full, "含校正事件")
    add("centered_6（現況）", *scores[6], eval_full, "含校正事件")
    # 主要比較：三種校正次數，同一個排除聯集後的評估集合
    add("raw（無centering）", s_raw, t_raw, eval_common, "排除校正事件")
    for n in CALIB_COUNTS:
        add(f"centered_{n}", *scores[n], eval_common, "排除校正事件")

    df = pd.DataFrame(rows)
    print("\n" + "=" * 92)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n" + "=" * 92)
    print("配對比較（同一批人，AUC 差值 + bootstrap 95% CI；基準＝centered_6，排除校正事件）")
    print("=" * 92)
    base = metrics["centered_6|排除校正事件"]
    boots = {}
    for n in CALIB_COUNTS[1:]:
        b = paired_bootstrap(base, metrics[f"centered_{n}|排除校正事件"])
        boots[n] = b
        if b is None:
            continue
        verdict = "顯著" if b["significant"] else "與雜訊分不開"
        print(f"  centered_{n:<3} {b['mean_diff']:+.4f}  "
              f"95%CI=[{b['ci_low']:+.4f}, {b['ci_high']:+.4f}]  n={b['n']}  {verdict}")

    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as fp:
        json.dump({"summary": rows, "paired_bootstrap": boots,
                   "calib_contamination": {str(n): float(y[calibs[n]].mean()) for n in CALIB_COUNTS},
                   "per_subject": metrics}, fp, ensure_ascii=False, indent=2)
    print(f"\n寫入 {os.path.join(out_dir, 'results.json')}")


if __name__ == "__main__":
    main()

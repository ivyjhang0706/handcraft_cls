# -*- coding: utf-8 -*-
"""
測試「校正事件的組成」對績效的影響——不動校正次數(維持6次)，只改挑哪6個。

動機：centering 是目前唯一真正有效的環節(raw 0.598 → centered_6 0.636)，而它替
held-out 受試者算基準線時，那 6 個校正事件是**完全隨機**抽的，不看 BG_Level：

    pool = np.where(subj == u)[0]
    pick = rng2.choice(pool, size=min(N_CALIB_EVENTS, len(pool)), replace=False)

但每個人的 Normal/High 比例差很多(2131 是 8% High、2294 是 71% High)，所以隨機
抽出來的基準線，對不同人代表的意義不一樣：有人的基準線是「我的正常狀態」，有人
的是「正常和高血糖的混合」。

注意這不會縮小同一個人內部的類別差距(減常數只是平移，不改變個人內排序)，真正的
問題是**不同人被錨定在不同位置**：甲的 Normal 落在 0、High 落在 +20，乙的 Normal
落在 -13、High 落在 +7。RF 只有一套決策規則，卻要同時處理這兩種錨點。如果大家都
用 Normal 校正，「0 ≈ 我的正常狀態」對每個人才會一致。

三種策略：
  random       現況：6 個隨機 BGM 事件；訓練端用該人全部事件算基準線
  normal_test  測試端改抽 6 個 Normal 事件；訓練端維持不變
  normal_both  測試端抽 6 個 Normal；訓練端也只用 Normal 事件算基準線
               (讓整個特徵空間都錨定在「正常狀態」，是這個想法最完整的版本)

這不是 leakage：校正就是「扎手指量一次並回報數值」，那 6 筆的血糖值本來就已知，
所以「請在正常狀態下校正」在部署時做得到，不需要任何未來資訊。

評估上的處理：校正事件目前也會被拿去評估，若改成只挑 Normal，這幾筆保證是 Normal
的事件會灌水 specificity。所以評估時**排除掉所有策略校正事件的聯集**，三種策略因此
都在完全相同的事件集合上被評分，配對比較才成立。代價是這個數字不能直接跟現有的
0.636 比(評估集合變小了)，所以另外附上「random 策略 + 不排除校正事件」的結果，
用來對照現況並看出自我參照偏差有多大。

用法：
  conda run -n ml python kfold/handcrafted_features/experiment_calib_composition.py
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
    K_FOLDS, SEED, N_CALIB_EVENTS, MIN_PER_SIDE,
    center_by_subject, pick_threshold, OUT_DIR,
)
from evaluate_normal_vs_high_k5 import binary_metrics  # noqa: E402

POS_LEVEL, NEG_LEVELS = "High", ["Normal"]
RF_PARAMS = dict(n_estimators=300, max_depth=6, class_weight="balanced",
                 random_state=0, n_jobs=8)
N_BOOT = 10000


def make_calib(subj, src, y, uuids, normal_only):
    """挑每個人的 6 個校正事件。

    normal_only=False 時逐行等同 plot_rf_report.build_predictions 的抽樣邏輯
    (同樣的 SEED、同樣的順序)，所以會抽到完全相同的那 6 個事件，可以當對照組。
    """
    rng2 = np.random.RandomState(SEED)
    calib = np.zeros(len(subj), bool)
    for u in uuids:
        pool = np.where(subj == u)[0]
        pool_bgm = pool[src[pool] == "BGM"]
        pool = pool_bgm if len(pool_bgm) > 0 else pool
        if normal_only:
            pool_n = pool[y[pool] == 0]
            if len(pool_n) > 0:
                pool = pool_n
        pick = rng2.choice(pool, size=min(N_CALIB_EVENTS, len(pool)), replace=False)
        calib[pick] = True
    return calib


def run_strategy(X, y, subj, src, uuids, folds, calib, train_normal_only):
    """跑 5 折，回傳每個事件的 centered_6 分數與門檻。

    train_normal_only=True 時，訓練受試者的基準線只用他們的 Normal 事件來算
    (測試端一律用 calib 指定的那 6 筆)。
    """
    score = np.full(len(y), np.nan)
    thr_all = np.full(len(y), np.nan)
    for f in range(K_FOLDS):
        te_u = [u for u in uuids if folds[u] == f]
        tr = ~np.isin(subj, te_u)
        te = np.isin(subj, te_u) & (src == "BGM")   # 只評估 BGM(部署現實)
        if te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue

        Xc = X.copy()
        tr_ref = (y == 0) if train_normal_only else None
        Xc[tr] = center_by_subject(X[tr], subj[tr],
                                   ref_idx=None if tr_ref is None else tr_ref[tr])
        Xc[te] = center_by_subject(X[te], subj[te], ref_idx=calib[te])

        sc = StandardScaler().fit(Xc[tr])
        clf = RandomForestClassifier(**RF_PARAMS)
        clf.fit(sc.transform(Xc[tr]), y[tr])
        thr = pick_threshold(clf.predict_proba(sc.transform(Xc[tr]))[:, 1], y[tr])
        score[te] = clf.predict_proba(sc.transform(Xc[te]))[:, 1]
        thr_all[te] = thr
    return score, thr_all


def run_raw(X, y, subj, src, uuids, folds):
    """不做 centering 的對照(跟校正策略無關，僅供參考)。"""
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
    out_dir = os.path.join(OUT_DIR, "calib_composition")
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

    calib_rand = make_calib(subj, src, y, uuids, normal_only=False)
    calib_norm = make_calib(subj, src, y, uuids, normal_only=True)
    print(f"隨機校正事件 {calib_rand.sum()} 個，其中 High {int(y[calib_rand].sum())} 個"
          f"（{y[calib_rand].mean():.1%} 被污染）")
    print(f"Normal校正事件 {calib_norm.sum()} 個，其中 High {int(y[calib_norm].sum())} 個")

    # 評估集合排除兩組校正事件的聯集，讓所有策略在同一批事件上被評分
    excl = calib_rand | calib_norm
    eval_common = ~excl
    eval_full = np.ones(len(y), bool)
    print(f"排除校正事件聯集 {excl.sum()} 個後，可評估事件 {eval_common.sum():,}/{len(y):,}")

    print("\n跑 raw（無 centering，對照用）...")
    s_raw, t_raw = run_raw(X, y, subj, src, uuids, folds)

    strategies = {
        "random（現況）": (calib_rand, False),
        "normal_test（測試端改用Normal校正）": (calib_norm, False),
        "normal_both（訓練端也只用Normal錨定）": (calib_norm, True),
    }
    scores = {}
    for name, (calib, tn) in strategies.items():
        print(f"跑 centered_6 / {name} ...")
        scores[name] = run_strategy(X, y, subj, src, uuids, folds, calib, tn)

    rows, metrics = [], {}
    def add(label, score, thr, mask, tag):
        m = per_subject_metrics(y, subj, score, thr, mask)
        metrics[f"{label}|{tag}"] = m
        vals = [m[u]["auc"] for u in m if m[u]["auc"] is not None]
        rec = [m[u]["recall"] for u in m if m[u]["recall"] is not None]
        spe = [m[u]["specificity"] for u in m if m[u]["specificity"] is not None]
        rows.append({"策略": label, "評估集": tag, "人數": len(m),
                     "AUC": float(np.mean(vals)) if vals else None,
                     "recall": float(np.mean(rec)) if rec else None,
                     "specificity": float(np.mean(spe)) if spe else None})

    # 對照現況：random + 不排除校正事件（可與既有的 0.598/0.636 對照）
    add("raw（無centering）", s_raw, t_raw, eval_full, "含校正事件")
    add("random（現況）", *scores["random（現況）"], eval_full, "含校正事件")
    # 主要比較：三種策略，同一個排除聯集後的評估集合
    add("raw（無centering）", s_raw, t_raw, eval_common, "排除校正事件")
    for name in strategies:
        add(name, *scores[name], eval_common, "排除校正事件")

    df = pd.DataFrame(rows)
    print("\n" + "=" * 92)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n" + "=" * 92)
    print("配對比較（同一批人，AUC 差值 + bootstrap 95% CI；基準＝random 策略，排除校正事件）")
    print("=" * 92)
    base = metrics["random（現況）|排除校正事件"]
    boots = {}
    for name in list(strategies)[1:]:
        b = paired_bootstrap(base, metrics[f"{name}|排除校正事件"])
        boots[name] = b
        if b is None:
            continue
        verdict = "顯著" if b["significant"] else "與雜訊分不開"
        print(f"  {name:40} {b['mean_diff']:+.4f}  "
              f"95%CI=[{b['ci_low']:+.4f}, {b['ci_high']:+.4f}]  n={b['n']}  {verdict}")

    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as fp:
        json.dump({"summary": rows, "paired_bootstrap": boots,
                   "calib_contamination": {
                       "random_high_ratio": float(y[calib_rand].mean()),
                       "normal_high_ratio": float(y[calib_norm].mean())},
                   "per_subject": metrics}, fp, ensure_ascii=False, indent=2)
    print(f"\n寫入 {os.path.join(out_dir, 'results.json')}")


if __name__ == "__main__":
    main()

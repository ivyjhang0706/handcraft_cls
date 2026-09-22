# -*- coding: utf-8 -*-
"""
把 experiment_handcrafted_classification.py 的方法套到「真正要部署用的資料」上：
`data/Regression_Features_BGM_updown_latest`（BGM 指尖採血，54人，「上下」校正過的
新版前處理），取代先前用的 `Regression_Features_new`（CGM，62人）。

背景：先前的 0.6214（Low, centered）跟 0.666/0.715（Normal vs High, centered）都是在
CGM 資料上測的。CGM 是連續15分鐘一筆，BGM 是指尖採血、一天通常只有幾筆、幾乎不會在
半夜量測——這代表：(a) 這是兩個不同的資料集，CGM 上的結論不能直接套用到 BGM；
(b) BGM 因為量測節奏本身偏白天，晝夜 confound 的嚴重程度可能天差地遠，需要重新驗證。

跑三組：
  A. BGM-only：跟 CGM 版一模一樣的方法（raw/centered_all/centered_6），純 BGM 資料，
     54人，subject-wise 5-fold。順便印晝夜分布、跑白天限定對照（Low 與 Normal vs High
     都做，補齊先前 exp.md 「尚未驗證事項」的 0a/0c）。
  B. CGM+BGM 合併訓練、只用 BGM 評估：population 訓練用兩種來源全部事件（54人BGM+CGM
     都有，另外8人只有CGM，只能當訓練），但**評估/校正永遠只看 BGM 事件**——這才符合
     部署現實：使用者的檢測/校正手段就是指尖採血，CGM 不是產品會用到的輸入來源，只是
     拿來加大訓練組的資料量。held-out 受試者的 CGM 事件在該人被抽中當測試對象的那一折
     裡完全不使用（不算訓練、也不算評估，避免用同一人的 CGM 洩漏到自己的測試）。

用法：
  conda run -n ml python kfold/handcrafted_features/experiment_handcrafted_bgm.py
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

_KFOLD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in sorted(os.listdir(_KFOLD_ROOT)):
    _p = os.path.join(_KFOLD_ROOT, _d)
    if os.path.isdir(_p):
        sys.path.insert(0, _p)

from experiment_handcrafted_classification import (  # noqa: E402
    META_COLS, K_FOLDS, SEED, N_CALIB_EVENTS, per_subject_auc, center_by_subject, OUT_DIR,
)
from experiment_handcrafted_daytime import hour_of_timestamp, DAY_START, DAY_END  # noqa: E402
from config import CGM_ROOT, BGM_ROOT  # noqa: E402

OUT_PATH = os.path.join(OUT_DIR, "bgm_and_merged_results.json")


def load_source(root, source_tag):
    """跟 experiment_handcrafted_classification.load_event_level 邏輯一致，多加 source 欄位。"""
    frames = []
    feat_cols = None
    for f in sorted(glob.glob(os.path.join(root, "*", "*_regression_features.csv"))):
        u = os.path.basename(os.path.dirname(f))
        df = pd.read_csv(f)
        cols = [c for c in df.columns if c not in META_COLS]
        if feat_cols is None:
            feat_cols = cols
        num = df[cols].apply(pd.to_numeric, errors="coerce")
        num["uuid"] = str(u)
        num["timestamp"] = df["timestamp"].values
        num["glucose"] = df["glucose"].values
        num["BG_Level"] = df["BG_Level"].values
        agg = num.groupby(["uuid", "timestamp"], as_index=False).agg(
            {**{c: "mean" for c in cols}, "glucose": "first", "BG_Level": "first"})
        frames.append(agg)
    data = pd.concat(frames, ignore_index=True)
    data["source"] = source_tag
    print(f"  {source_tag}: {data['uuid'].nunique()} 人，{len(data):,} 事件")
    return data, feat_cols


def load_merged():
    bgm, feat_cols = load_source(BGM_ROOT, "BGM")
    cgm, feat_cols2 = load_source(CGM_ROOT, "CGM")
    assert feat_cols == feat_cols2, "CGM/BGM 特徵欄位不一致"
    merged = pd.concat([bgm, cgm], ignore_index=True)
    before = len(merged)
    merged = merged.drop_duplicates(subset=["uuid", "timestamp"], keep="first")
    if len(merged) != before:
        print(f"  合併後移除 {before - len(merged)} 筆 uuid+timestamp 重複事件（BGM 優先保留）")
    return merged, feat_cols, bgm


def print_day_night_table(data, label):
    hours = hour_of_timestamp(data["timestamp"].to_numpy())
    daytime = (hours >= DAY_START) & (hours <= DAY_END)
    print(f"\n[{label}] 晝夜事件分布：")
    print(f"{'類別':8}{'事件數':>10}{'夜間/清晨(00-07)':>20}{'白天(08-19)':>16}")
    for level in ["Low", "Normal", "High"]:
        m = (data["BG_Level"] == level).to_numpy()
        n = m.sum()
        if n == 0:
            continue
        n_night = int((~daytime[m]).sum())
        n_day = int(daytime[m].sum())
        print(f"{level:8}{n:>10,}{n_night:>13,} ({n_night/n:>5.1%}){n_day:>10,} ({n_day/n:>5.1%})")
    return daytime


def run_task(data, feat_cols, task_name, pos_level, neg_levels,
             eval_source=None, calib_source=None, calib_seed=None, verbose=True):
    """
    跟 experiment_handcrafted_classification.run_task 相同邏輯，多三個參數：
      eval_source : 若給定，held-out 受試者只有這個來源的事件會被拿來評估
                    （另一來源的事件整批丟棄，不訓練也不評估）。None = 不限制(單一來源時用)。
      calib_source: centered_6 抽 6 筆校正事件時，只從這個來源抽（None = 不限制）。
      calib_seed  : 只控制「抽哪6筆當校正事件」的隨機性，不影響 fold 切分（fold 切分
                    永遠用固定的 SEED，這樣才能重複抽校正事件、同時保證每次比較的是
                    同一組 train/test 切分）。None = 用預設 SEED。
    """
    if verbose:
        print(f"\n{'=' * 76}\n{task_name}\n{'=' * 76}")
    sub = data[data["BG_Level"].isin([pos_level] + neg_levels)].reset_index(drop=True)
    X = sub[feat_cols].to_numpy(dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = (sub["BG_Level"] == pos_level).astype(int).to_numpy()
    subj = sub["uuid"].to_numpy()
    src = sub["source"].to_numpy()
    uuids = np.array(sorted(set(subj)))
    if verbose:
        print(f"事件數 = {len(y):,}，正類({pos_level}) = {y.sum():,}，人數 = {len(uuids)}")
        if eval_source is not None:
            print(f"  評估限定來源 = {eval_source}（held-out 受試者的其他來源事件會被排除）")

    rng = np.random.RandomState(SEED)
    order = rng.permutation(len(uuids))
    folds = {u: i % K_FOLDS for i, u in enumerate(uuids[order])}

    modes = ["raw", "centered_all", "centered_6"]
    results = {k: {"score": np.full(len(y), np.nan), "done": np.zeros(len(y), bool)} for k in modes}

    rng2 = np.random.RandomState(SEED if calib_seed is None else calib_seed)
    calib_mask = np.zeros(len(y), bool)
    for u in uuids:
        pool = np.where(subj == u)[0]
        if calib_source is not None:
            pool_c = pool[src[pool] == calib_source]
            pool = pool_c if len(pool_c) > 0 else pool
        pick = rng2.choice(pool, size=min(N_CALIB_EVENTS, len(pool)), replace=False)
        calib_mask[pick] = True

    for f in range(K_FOLDS):
        te_u = {u for u in uuids if folds[u] == f}
        tr = ~np.isin(subj, list(te_u))
        te = np.isin(subj, list(te_u))
        if eval_source is not None:
            te = te & (src == eval_source)
        for mode in modes:
            if mode == "raw":
                Xtr, Xte = X[tr], X[te]
            elif mode == "centered_all":
                Xc = center_by_subject(X, subj)
                Xtr, Xte = Xc[tr], Xc[te]
            else:  # centered_6
                Xc = X.copy()
                Xc[tr] = center_by_subject(X[tr], subj[tr])
                Xc[te] = center_by_subject(X[te], subj[te], ref_idx=calib_mask[te])
                Xtr, Xte = Xc[tr], Xc[te]
            if len(np.unique(y[tr])) < 2 or te.sum() == 0:
                continue
            sc = StandardScaler().fit(Xtr)
            clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.1)
            clf.fit(sc.transform(Xtr), y[tr])
            results[mode]["score"][te] = clf.decision_function(sc.transform(Xte))
            results[mode]["done"][te] = True
        if verbose:
            print(f"  fold {f}: 訓練 {tr.sum():,} 事件 / 評估 {te.sum():,} 事件（{len(te_u)} 人待評估）")

    summary = {}
    if verbose:
        print(f"\n{'做法':16}{'per-subject AUC':>18}{'可評估人數':>12}{'pooled AUC':>13}")
    for mode in modes:
        done = results[mode]["done"]
        s, yy, ss = results[mode]["score"][done], y[done], subj[done]
        if done.sum() == 0 or len(np.unique(yy)) < 2:
            summary[mode] = {"per_subject_auc": None, "n_subjects": 0, "pooled_auc": None,
                              "per_subject_list": []}
            if verbose:
                print(f"{mode:16}{'N/A':>18}{0:>12}{'N/A':>13}")
            continue
        ps, n, per = per_subject_auc(yy, s, ss)
        pooled = roc_auc_score(yy, s)
        summary[mode] = {"per_subject_auc": ps, "n_subjects": n, "pooled_auc": float(pooled),
                          "per_subject_list": per}
        if verbose:
            ps_s = f"{ps:.4f}" if ps is not None and not np.isnan(ps) else "N/A"
            print(f"{mode:16}{ps_s:>18}{n:>12}{pooled:>13.4f}")
    return summary


def main():
    print("讀取 BGM-only 事件層級特徵...")
    bgm_data, feat_cols = load_source(BGM_ROOT, "BGM")
    print(f"\n總計 {len(bgm_data):,} 事件，{bgm_data['uuid'].nunique()} 人，{len(feat_cols)} 個特徵")
    print(f"類別分布:\n{bgm_data['BG_Level'].value_counts()}")

    res = {}

    print(f"\n{'#'*76}\n# A. BGM-only 分類\n{'#'*76}")
    res["bgm_only"] = {}
    res["bgm_only"]["Normal_vs_High"] = run_task(
        bgm_data, feat_cols, "[BGM-only] Normal vs High", "High", ["Normal"])
    res["bgm_only"]["Low_vs_rest"] = run_task(
        bgm_data, feat_cols, "[BGM-only] Low vs 其餘", "Low", ["Normal", "High"])

    print(f"\n{'#'*76}\n# A2. BGM-only 晝夜 confound 檢查\n{'#'*76}")
    daytime = print_day_night_table(bgm_data, "BGM-only")
    bgm_day = bgm_data[daytime].reset_index(drop=True)
    print(f"篩選白天(08-19)後：{len(bgm_day):,} / {len(bgm_data):,} 事件，"
          f"{bgm_day['uuid'].nunique()} / {bgm_data['uuid'].nunique()} 人")
    res["bgm_only_daytime"] = {}
    res["bgm_only_daytime"]["Normal_vs_High"] = run_task(
        bgm_day, feat_cols, "[BGM-only, 白天限定] Normal vs High", "High", ["Normal"])
    res["bgm_only_daytime"]["Low_vs_rest"] = run_task(
        bgm_day, feat_cols, "[BGM-only, 白天限定] Low vs 其餘", "Low", ["Normal", "High"])

    print(f"\n{'#'*76}\n# B. CGM+BGM 合併訓練，只用 BGM 評估\n{'#'*76}")
    print("讀取合併資料...")
    merged_data, feat_cols_m, _ = load_merged()
    assert feat_cols_m == feat_cols
    print(f"合併後：{merged_data['uuid'].nunique()} 人，{len(merged_data):,} 事件"
          f"（BGM {int((merged_data['source']=='BGM').sum()):,} + "
          f"CGM {int((merged_data['source']=='CGM').sum()):,}）")
    res["merged_train_bgm_test"] = {}
    res["merged_train_bgm_test"]["Normal_vs_High"] = run_task(
        merged_data, feat_cols, "[CGM+BGM訓練/BGM評估] Normal vs High", "High", ["Normal"],
        eval_source="BGM", calib_source="BGM")
    res["merged_train_bgm_test"]["Low_vs_rest"] = run_task(
        merged_data, feat_cols, "[CGM+BGM訓練/BGM評估] Low vs 其餘", "Low", ["Normal", "High"],
        eval_source="BGM", calib_source="BGM")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fp:
        json.dump(res, fp, ensure_ascii=False, indent=2)
    print(f"\n寫入 {OUT_PATH}")

    print(f"\n{'='*90}\n總對照表（per-subject AUC，centered_all 欄位）\n{'='*90}")
    print(f"{'情境':30}{'Normal vs High':>18}{'Low vs 其餘':>18}")
    for key, label in [("bgm_only", "BGM-only（全時段）"),
                        ("bgm_only_daytime", "BGM-only（白天限定）"),
                        ("merged_train_bgm_test", "CGM+BGM訓練/BGM評估")]:
        nh = res[key]["Normal_vs_High"]["centered_all"]["per_subject_auc"]
        lr = res[key]["Low_vs_rest"]["centered_all"]["per_subject_auc"]
        nh_s = f"{nh:.4f}" if nh is not None else "N/A"
        lr_s = f"{lr:.4f}" if lr is not None else "N/A"
        print(f"{label:30}{nh_s:>18}{lr_s:>18}")
    print(f"{'CGM-only（先前結果，對照）':30}{'0.6664~0.7147':>18}{'0.6214':>18}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
測試「手工形態學特徵能不能做跨人分類」——這是先前迴歸測試沒有回答的問題。

為什麼要分開測：先前的迴歸測試是「已知類別後預測精確值」，類別內 CV 只有 9-14%，
trivial baseline 已達 10.35% MARD，先天沒有多少發揮空間。分類是完全不同難度的任務
（72 vs 216 mg/dL，差3倍），特徵分不出 110/125 完全不代表分不出 72/216。

資料：data/Regression_Features_new/Regression_Features/{uuid}/{uuid}_regression_features.csv
      每個 timestamp（一次血糖讀數）底下約 28 個 ECG 片段，聚合成事件層級再用。

**關鍵：用跨人切分**（CSV 裡現成的 split 欄位是個人內部的 70/10/20，不能拿來評估
通用模型，會嚴重高估）。這裡自己做 subject-wise K-fold。

比較四種做法：
  raw          : 原始特徵直接訓練 population 模型
  centered_all : 每人減掉自己的平均（用全部事件算）——理想上限，部署時拿不到
  centered_6   : 每人減掉自己的平均（只用6筆校正事件算）——真實部署情境
  raw + 6筆個人化 : 原始特徵 + 用6筆做個人化擬合

評估：per-subject AUC（產品真正的問題）＋ pooled AUC（對照，會被人際差異灌水）

用法：
  conda run -n ECGFounder python kfold/handcrafted_features/experiment_handcrafted_classification.py
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ROOT, OUT_DIR, K_FOLDS, SEED, N_CALIB_EVENTS, MIN_PER_SIDE  # noqa: E402

META_COLS = ["source_file", "uuid", "timestamp", "type", "index", "glucose",
             "split", "BG_Level", "group_id", "split_method", "status"]
N_REPEATS = 20


def load_event_level():
    """讀全部 uuid，聚合成事件層級（同一 timestamp 的片段取平均）。"""
    frames = []
    for d in sorted(glob.glob(os.path.join(ROOT, "*"))):
        u = os.path.basename(d)
        f = os.path.join(d, f"{u}_regression_features.csv")
        if not os.path.isfile(f):
            continue
        df = pd.read_csv(f)
        feat_cols = [c for c in df.columns if c not in META_COLS]
        num = df[feat_cols].apply(pd.to_numeric, errors="coerce")
        num["uuid"] = str(u)
        num["timestamp"] = df["timestamp"].values
        num["glucose"] = df["glucose"].values
        num["BG_Level"] = df["BG_Level"].values
        # 事件層級：同一 timestamp 的片段取平均
        agg = num.groupby(["uuid", "timestamp"], as_index=False).agg(
            {**{c: "mean" for c in feat_cols}, "glucose": "first", "BG_Level": "first"})
        frames.append(agg)
        print(f"  {u}: {len(df)} 片段 → {len(agg)} 事件")
    data = pd.concat(frames, ignore_index=True)
    return data, feat_cols


def per_subject_auc(y_true, score, subj):
    """每個人各自算 AUC 再平均（只算兩類都有的人）。"""
    out = []
    for u in np.unique(subj):
        m = subj == u
        if len(np.unique(y_true[m])) < 2:
            continue
        if min((y_true[m] == 0).sum(), (y_true[m] == 1).sum()) < MIN_PER_SIDE:
            continue
        out.append(roc_auc_score(y_true[m], score[m]))
    return (float(np.mean(out)) if out else np.nan), len(out), out


def pick_threshold(score, y, objective="balanced_accuracy"):
    """在訓練組(population)的分數上，搜尋讓 balanced_accuracy 最大的二元化門檻，
    取代寫死的 0.0(decision_function)/0.5(predict_proba)。跟 kfold/model.py 的
    shift_decision_thresholds() 同樣精神(每類別 recall 平均，不被多數類別 Normal
    主導)，只是這裡是二元、單一標量門檻，窮舉訓練組分數的所有唯一值當候選即可。

    只能用訓練組(population)資料呼叫，不能碰待評估的受試者，否則是洩漏。
    """
    if objective != "balanced_accuracy":
        raise ValueError(f"未知的 objective: {objective}")
    y = np.asarray(y)
    candidates = np.unique(score)
    best_t, best_score = 0.0, -1.0
    for t in candidates:
        pred = (score > t).astype(int)
        recalls = [np.mean(pred[y == c] == c) for c in (0, 1) if (y == c).any()]
        s = float(np.mean(recalls))
        if s > best_score:
            best_score, best_t = s, float(t)
    return best_t


def center_by_subject(X, subj, ref_idx=None):
    """每人減掉自己的平均。ref_idx 指定用哪些列算平均（模擬只有6筆校正）。"""
    Xc = X.copy()
    for u in np.unique(subj):
        m = subj == u
        if ref_idx is None:
            mu = X[m].mean(axis=0)
        else:
            mm = m & ref_idx
            mu = X[mm].mean(axis=0) if mm.sum() > 0 else X[m].mean(axis=0)
        Xc[m] = X[m] - mu
    return Xc


def run_task(data, feat_cols, task_name, pos_level, neg_levels):
    print(f"\n{'=' * 76}\n{task_name}\n{'=' * 76}")
    sub = data[data["BG_Level"].isin([pos_level] + neg_levels)].reset_index(drop=True)
    X = sub[feat_cols].to_numpy(dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = (sub["BG_Level"] == pos_level).astype(int).to_numpy()
    subj = sub["uuid"].to_numpy()
    uuids = np.array(sorted(set(subj)))
    print(f"事件數 = {len(y):,}，正類({pos_level}) = {y.sum():,}，人數 = {len(uuids)}")

    rng = np.random.RandomState(SEED)
    order = rng.permutation(len(uuids))
    folds = {u: i % K_FOLDS for i, u in enumerate(uuids[order])}

    results = {k: {"score": np.zeros(len(y)), "done": np.zeros(len(y), bool)}
               for k in ["raw", "centered_all", "centered_6"]}
    rng2 = np.random.RandomState(SEED)
    # 為每個人固定抽一次「6筆校正事件」(不分層，當中性基準線)
    calib_mask = np.zeros(len(y), bool)
    for u in uuids:
        idx = np.where(subj == u)[0]
        pick = rng2.choice(idx, size=min(N_CALIB_EVENTS, len(idx)), replace=False)
        calib_mask[pick] = True

    for f in range(K_FOLDS):
        te_u = {u for u in uuids if folds[u] == f}
        te = np.isin(subj, list(te_u)); tr = ~te
        for mode in ["raw", "centered_all", "centered_6"]:
            if mode == "raw":
                Xtr, Xte = X[tr], X[te]
            elif mode == "centered_all":
                Xc = center_by_subject(X, subj)
                Xtr, Xte = Xc[tr], Xc[te]
            else:  # centered_6：測試者只能用自己的6筆算基準線
                Xc = X.copy()
                Xc[tr] = center_by_subject(X[tr], subj[tr])
                Xc[te] = center_by_subject(X[te], subj[te], ref_idx=calib_mask[te])
                Xtr, Xte = Xc[tr], Xc[te]
            sc = StandardScaler().fit(Xtr)
            clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.1)
            clf.fit(sc.transform(Xtr), y[tr])
            results[mode]["score"][te] = clf.decision_function(sc.transform(Xte))
            results[mode]["done"][te] = True
        print(f"  fold {f}: 測試 {len(te_u)} 人 / {te.sum():,} 事件")

    summary = {}
    print(f"\n{'做法':16}{'per-subject AUC':>18}{'可評估人數':>12}{'pooled AUC':>13}")
    for mode in ["raw", "centered_all", "centered_6"]:
        s = results[mode]["score"]
        ps, n, per = per_subject_auc(y, s, subj)
        pooled = roc_auc_score(y, s)
        summary[mode] = {"per_subject_auc": ps, "n_subjects": n, "pooled_auc": float(pooled),
                          "per_subject_list": per}
        print(f"{mode:16}{ps:>18.4f}{n:>12}{pooled:>13.4f}")
    return summary


def main():
    print("讀取事件層級特徵...")
    data, feat_cols = load_event_level()
    print(f"\n總計 {len(data):,} 事件，{data['uuid'].nunique()} 人，{len(feat_cols)} 個特徵")
    print(f"類別分布:\n{data['BG_Level'].value_counts()}")

    res = {}
    res["Normal_vs_High"] = run_task(data, feat_cols, "Normal vs High（手工特徵，跨人切分）",
                                      "High", ["Normal"])
    res["Low_vs_rest"] = run_task(data, feat_cols, "Low vs 其餘（手工特徵，跨人切分）",
                                   "Low", ["Normal", "High"])

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as fp:
        json.dump(res, fp, ensure_ascii=False, indent=2)
    print(f"\n寫入 {OUT_DIR}/results.json")
    print("\n判讀：per-subject AUC 明顯 >0.5 才代表這些特徵對跨人分類有用；"
          "\n      pooled 遠高於 per-subject 則代表又是靠人際差異。")


if __name__ == "__main__":
    main()

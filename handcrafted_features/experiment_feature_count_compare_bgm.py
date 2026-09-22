# -*- coding: utf-8 -*-
"""
跟 experiment_feature_count_compare.py 完全同樣的設計（17個篩選特徵 vs 完整~96個
特徵，RandomForest，raw/centered_all/centered_6 三種模式 + baseline vs centered_6
完整指標報表），但資料換成「CGM+BGM 合併訓練、只用 BGM 評估」的場景（跟
experiment_handcrafted_bgm.py 的 scenario B、plot_rf_report.py 一致）：

  訓練：用 CGM+BGM 兩來源全部事件（62人：54人有BGM+CGM，8人只有CGM）擴大訓練樣本。
  評估/校正：held-out 受試者永遠只看 BGM 事件——這才符合部署現實（產品只會有
             BGM扎手指資料，CGM不是產品輸入來源，只是用來加大訓練量）。

用法：
  conda run -n ml python kfold/handcrafted_features/experiment_feature_count_compare_bgm.py --features all
  conda run -n ml python kfold/handcrafted_features/experiment_feature_count_compare_bgm.py --features 17

輸出（避免跟 CGM-only 版本的 17/96 資料夾撞名，另外放一層 rf_cgm+bgm/）：
  output/handcrafted_classification/rf_cgm+bgm/{96,17}/
    results.json / performance_summary.xlsx+png                    （同前，AUC對照）
    {task}_sens_auc_merged.png / _per_subject_merged.png / _per_subject_sensitivity.png
    rf_metrics_summary.xlsx      （同前，完整指標多分頁+逐人明細）
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiment_handcrafted_classification import (  # noqa: E402
    K_FOLDS, SEED, N_CALIB_EVENTS, MIN_PER_SIDE, per_subject_auc, center_by_subject,
    pick_threshold, OUT_DIR,
)
from experiment_handcrafted_bgm import load_merged  # noqa: E402
from experiment_feature_count_compare import (  # noqa: E402
    RF_PARAMS, MODES, COLOR_BASE, COLOR_PERS, SELECTED_17, TASKS as ALL_TASKS,
    per_subject_metrics, plot_task_reports, write_full_excel,
)

BGM_EVAL_OUT_DIR = os.path.join(OUT_DIR, "rf_cgm+bgm")

# Low vs 其餘在 BGM 評估下 Low 事件太稀疏(先前 LogReg benchmark 只有 n=2 可評估，
# 不可信)，這個場景只跑 Normal vs High。
TASKS = [t for t in ALL_TASKS if t[0] == "Normal_vs_High"]


def run_task(data, feat_cols, task_name, pos_level, neg_levels):
    """跟 experiment_feature_count_compare.run_task 邏輯一致，多加 eval_source="BGM"、
    calib_source="BGM"：held-out 受試者只評估/校正 BGM 事件，訓練仍用 CGM+BGM 全部。"""
    print(f"\n{'=' * 76}\n{task_name}\n{'=' * 76}")
    sub = data[data["BG_Level"].isin([pos_level] + neg_levels)].reset_index(drop=True)
    X = sub[feat_cols].to_numpy(dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = (sub["BG_Level"] == pos_level).astype(int).to_numpy()
    subj = sub["uuid"].to_numpy()
    src = sub["source"].to_numpy()
    uuids = np.array(sorted(set(subj)))
    print(f"事件數 = {len(y):,}，正類({pos_level}) = {y.sum():,}，人數 = {len(uuids)}"
          f"（評估限定 BGM 事件）")

    rng = np.random.RandomState(SEED)
    order = rng.permutation(len(uuids))
    folds = {u: i % K_FOLDS for i, u in enumerate(uuids[order])}

    rng2 = np.random.RandomState(SEED)
    calib_mask = np.zeros(len(y), bool)
    for u in uuids:
        pool = np.where(subj == u)[0]
        pool_bgm = pool[src[pool] == "BGM"]
        pool = pool_bgm if len(pool_bgm) > 0 else pool
        pick = rng2.choice(pool, size=min(N_CALIB_EVENTS, len(pool)), replace=False)
        calib_mask[pick] = True

    results = {k: {"score": np.full(len(y), np.nan), "done": np.zeros(len(y), bool)} for k in MODES}
    thr_raw = np.full(len(y), np.nan)
    thr_c6 = np.full(len(y), np.nan)

    for f in range(K_FOLDS):
        te_u = {u for u in uuids if folds[u] == f}
        tr = ~np.isin(subj, list(te_u))
        te = np.isin(subj, list(te_u)) & (src == "BGM")
        if te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        for mode in MODES:
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
            sc = StandardScaler().fit(Xtr)
            clf = RandomForestClassifier(**RF_PARAMS)
            clf.fit(sc.transform(Xtr), y[tr])
            score_te = clf.predict_proba(sc.transform(Xte))[:, 1]
            results[mode]["score"][te] = score_te
            results[mode]["done"][te] = True
            if mode in ("raw", "centered_6"):
                score_tr = clf.predict_proba(sc.transform(Xtr))[:, 1]
                thr = pick_threshold(score_tr, y[tr])
                (thr_raw if mode == "raw" else thr_c6)[te] = thr
        print(f"  fold {f}: 評估 {te.sum():,} 筆BGM事件（{len(te_u)} 人待評估）")

    summary = {}
    print(f"\n{'做法':16}{'per-subject AUC':>18}{'可評估人數':>12}{'pooled AUC':>13}")
    for mode in MODES:
        done = results[mode]["done"]
        s, yy, ss = results[mode]["score"][done], y[done], subj[done]
        if done.sum() == 0 or len(np.unique(yy)) < 2:
            summary[mode] = {"per_subject_auc": None, "n_subjects": 0, "pooled_auc": None,
                              "per_subject_list": []}
            print(f"{mode:16}{'N/A':>18}{0:>12}{'N/A':>13}")
            continue
        ps, n, per = per_subject_auc(yy, s, ss)
        pooled = roc_auc_score(yy, s)
        summary[mode] = {"per_subject_auc": ps, "n_subjects": n, "pooled_auc": float(pooled),
                          "per_subject_list": per}
        print(f"{mode:16}{ps:>18.4f}{n:>12}{pooled:>13.4f}")

    done_raw, done_c6 = results["raw"]["done"], results["centered_6"]["done"]
    detail = {"y_raw": y[done_raw], "subj_raw": subj[done_raw], "score_raw": results["raw"]["score"][done_raw],
               "thr_raw": thr_raw[done_raw],
               "y_c6": y[done_c6], "subj_c6": subj[done_c6], "score_c6": results["centered_6"]["score"][done_c6],
               "thr_c6": thr_c6[done_c6]}
    return summary, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=["all", "17"], required=True,
                     help="all=完整~96個手工特徵, 17=Regression_Model_Predictor_meta_OOF.py 篩選出的17個")
    args = ap.parse_args()

    print("讀取合併資料（CGM+BGM）...")
    data, all_feat_cols, _ = load_merged()

    if args.features == "17":
        missing = [f for f in SELECTED_17 if f not in all_feat_cols]
        if missing:
            raise ValueError(f"17個特徵裡有些欄位在資料裡不存在: {missing}")
        feat_cols = SELECTED_17
        out_subdir = "17"
    else:
        feat_cols = all_feat_cols
        out_subdir = "96"

    print(f"使用 {len(feat_cols)} 個特徵，輸出到 output/handcrafted_classification/"
          f"rf_cgm+bgm/{out_subdir}/")

    out_dir = os.path.join(BGM_EVAL_OUT_DIR, out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    res = {}
    detail = {}
    for task_key, task_name, pos_level, neg_levels, neg_label, pos_label in TASKS:
        res[task_key], detail[task_key] = run_task(data, feat_cols, task_name, pos_level, neg_levels)

    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as fp:
        json.dump(res, fp, ensure_ascii=False, indent=2)
    print(f"寫入 {os.path.join(out_dir, 'results.json')}")

    # ---------- 績效表/圖（raw vs centered_all vs centered_6，只看 AUC） ----------
    rows = []
    for task_key, *_ in TASKS:
        for mode in MODES:
            m = res[task_key][mode]
            rows.append({"task": task_key, "mode": mode, "n_features": len(feat_cols),
                         "per_subject_auc": m["per_subject_auc"], "pooled_auc": m["pooled_auc"],
                         "n_subjects": m["n_subjects"]})
    pd.DataFrame(rows).to_excel(os.path.join(out_dir, "performance_summary.xlsx"), index=False)
    print(f"寫入 {os.path.join(out_dir, 'performance_summary.xlsx')}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (task_key, task_name, *_rest) in zip(axes, TASKS):
        vals_ps = [res[task_key][m]["per_subject_auc"] for m in MODES]
        vals_pool = [res[task_key][m]["pooled_auc"] for m in MODES]
        vals_ps = [v if v is not None else 0 for v in vals_ps]
        vals_pool = [v if v is not None else 0 for v in vals_pool]
        x = np.arange(len(MODES))
        ax.bar(x - 0.2, vals_ps, 0.4, label="per-subject AUC", color=COLOR_PERS)
        ax.bar(x + 0.2, vals_pool, 0.4, label="pooled AUC", color=COLOR_BASE)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance (0.5)")
        ax.set_xticks(x)
        ax.set_xticklabels(MODES)
        ax.set_ylim(0, 1.0)
        ax.set_title(f"[RandomForest, {len(feat_cols)}特徵, 合併訓練/BGM評估] {task_name}", fontsize=10)
        ax.legend(fontsize=8)
        for i, (a, b) in enumerate(zip(vals_ps, vals_pool)):
            ax.annotate(f"{a:.3f}", (i - 0.2, a), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
            ax.annotate(f"{b:.3f}", (i + 0.2, b), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "performance_summary.png"), dpi=150)
    plt.close(fig)
    print(f"寫入 {os.path.join(out_dir, 'performance_summary.png')}")

    # ---------- 完整指標報表（baseline vs centered_6）：圖 + summary excel ----------
    task_metrics = {}
    for task_key, task_name, pos_level, neg_levels, neg_label, pos_label in TASKS:
        d = detail[task_key]
        m_raw = per_subject_metrics(d["y_raw"], d["subj_raw"], d["score_raw"], d["thr_raw"])
        m_c6 = per_subject_metrics(d["y_c6"], d["subj_c6"], d["score_c6"], d["thr_c6"])
        common_uuids = sorted(set(m_raw) & set(m_c6))
        task_metrics[task_key] = (m_raw, m_c6, common_uuids)
        if common_uuids:
            plot_task_reports(m_raw, m_c6, common_uuids, out_dir, task_key, pos_label, neg_label)
        print(f"  {task_key}: 完整指標可評估人數 = {len(common_uuids)}")

    excel_path = os.path.join(out_dir, "rf_metrics_summary.xlsx")
    write_full_excel(task_metrics, excel_path)
    print(f"寫入 {excel_path}")


if __name__ == "__main__":
    main()

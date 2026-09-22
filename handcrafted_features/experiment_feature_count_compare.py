# -*- coding: utf-8 -*-
"""
比較「用幾個特徵」對 RandomForest 分類表現的影響：完整 ~96 個手工形態學特徵 vs
只用 Regression_Model_Predictor_meta_OOF.py 特徵選擇挑出的 17 個特徵。

跟 experiment_handcrafted_classification.py 用同一份資料（CGM 62人）、同一套
fold 切分/centering 邏輯，只換兩個變因：
  1. 特徵數量：--features all(~96個) 或 17(篩選過的17個)
  2. 模型：RandomForest（model_compare_results.json 已證實在 centered_6 上是目前
     最好、最穩的模型，之後 handcrafted 特徵實驗預設都用它，不再逐次跑
     LogisticRegression 對照）

用法：
  conda run -n ml python kfold/handcrafted_features/experiment_feature_count_compare.py --features all
  conda run -n ml python kfold/handcrafted_features/experiment_feature_count_compare.py --features 17

輸出（依 --features 分別存到對應資料夾，方便並排比較 96 vs 17）：
  output/handcrafted_classification/{96,17}/
    results.json                   raw/centered_all/centered_6 三種模式的原始數字
    performance_summary.xlsx/png   三種模式的 per-subject AUC / pooled AUC 對照
    {task}_sens_auc_merged.png         baseline vs centered_6：分類別 sensitivity + AUC 平均
    {task}_per_subject_merged.png      baseline vs centered_6：逐人 AUC 對照
    {task}_per_subject_sensitivity.png baseline vs centered_6：逐人 sensitivity 對照
      （task = Normal_vs_High / Low_vs_rest，各存一份）
    rf_metrics_summary.xlsx        baseline vs centered_6 完整指標：
                                      每個指標(AUC/Accuracy/Precision/Recall/
                                      Specificity/F1)各一個分頁(逐task平均)，
                                      最後一個分頁是逐人逐task的完整指標
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
    load_event_level, K_FOLDS, SEED, N_CALIB_EVENTS, MIN_PER_SIDE,
    per_subject_auc, center_by_subject, pick_threshold, OUT_DIR,
)
from binary_metrics import binary_metrics, METRIC_KEYS, METRIC_LABELS  # noqa: E402

RF_PARAMS = dict(n_estimators=300, max_depth=6, class_weight="balanced", random_state=0, n_jobs=1)
MODES = ["raw", "centered_all", "centered_6"]
COLOR_BASE = "#9fb8d1"
COLOR_PERS = "#e0824a"

# Regression_Model_Predictor_meta_OOF.py 的 used_feature_dic 裡 flag=1 的欄位（17個）
SELECTED_17 = [
    "rr_interval", "rt_duration", "pt_duration", "qt_duration", "qt_distances",
    "st_corrections3", "t_left_slope", "t_right_slope", "t_left_sharp", "t_right_sharp",
    "qrs_area", "st_area", "t_wave_cog", "dif_qs_amp/dif_qr_amp", "dif_rt_amp/dif_st_amp",
    "t/r_amp", "s/t_amp",
]

TASKS = [
    # (key,             task_name,  pos_level, neg_levels,    neg_label, pos_label)
    ("Normal_vs_High", "Normal vs High", "High", ["Normal"], "Normal", "High"),
    ("Low_vs_rest",    "Low vs 其餘",    "Low",  ["Normal", "High"], "rest", "Low"),
]


def run_task(data, feat_cols, task_name, pos_level, neg_levels):
    """subject-wise 5-fold，raw/centered_all/centered_6 三種模式都跑。額外把 raw 跟
    centered_6 的逐筆分數/門檻存下來（centered_6 all 只是理論上限，不需要完整指標），
    後面拿去算 accuracy/precision/recall/specificity/f1。"""
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

    results = {k: {"score": np.zeros(len(y)), "done": np.zeros(len(y), bool)} for k in MODES}
    thr_raw = np.zeros(len(y))
    thr_c6 = np.zeros(len(y))
    rng2 = np.random.RandomState(SEED)
    calib_mask = np.zeros(len(y), bool)
    for u in uuids:
        idx = np.where(subj == u)[0]
        pick = rng2.choice(idx, size=min(N_CALIB_EVENTS, len(idx)), replace=False)
        calib_mask[pick] = True

    for f in range(K_FOLDS):
        te_u = {u for u in uuids if folds[u] == f}
        te = np.isin(subj, list(te_u))
        tr = ~te
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
        print(f"  fold {f}: 測試 {len(te_u)} 人 / {te.sum():,} 事件")

    summary = {}
    print(f"\n{'做法':16}{'per-subject AUC':>18}{'可評估人數':>12}{'pooled AUC':>13}")
    for mode in MODES:
        s = results[mode]["score"]
        ps, n, per = per_subject_auc(y, s, subj)
        pooled = roc_auc_score(y, s)
        summary[mode] = {"per_subject_auc": ps, "n_subjects": n, "pooled_auc": float(pooled),
                          "per_subject_list": per}
        print(f"{mode:16}{ps:>18.4f}{n:>12}{pooled:>13.4f}")

    detail = {"y": y, "subj": subj, "score_raw": results["raw"]["score"],
              "score_c6": results["centered_6"]["score"], "thr_raw": thr_raw, "thr_c6": thr_c6}
    return summary, detail


def per_subject_metrics(y, subj, score, thr):
    """回傳 {uuid: {metric: value}}，只保留兩類都 >= MIN_PER_SIDE 的人。"""
    out = {}
    for u in np.unique(subj):
        m = subj == u
        if min((y[m] == 0).sum(), (y[m] == 1).sum()) < MIN_PER_SIDE:
            continue
        out[u] = binary_metrics(y[m], score[m], threshold=float(thr[m][0]))
    return out


def plot_task_reports(m_raw, m_c6, uuids, out_dir, task_key, pos_label, neg_label):
    """畫三張 baseline vs centered_6 對照圖，檔名依 task_key 分開存。"""
    n = len(uuids)

    def overall(metrics_dict, key):
        vals = [metrics_dict[u][key] for u in uuids if metrics_dict[u][key] is not None]
        return float(np.mean(vals)) if vals else None

    # 圖1：分類別 sensitivity + AUC 平均
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    b_vals = [overall(m_raw, "specificity"), overall(m_raw, "recall")]
    c_vals = [overall(m_c6, "specificity"), overall(m_c6, "recall")]
    x = np.arange(2)
    axes[0].bar(x - 0.2, [v or 0 for v in b_vals], 0.4, label="baseline", color=COLOR_BASE)
    axes[0].bar(x + 0.2, [v or 0 for v in c_vals], 0.4, label="centered_6", color=COLOR_PERS)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([neg_label, pos_label])
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("sensitivity (recall)")
    axes[0].set_title(f"[RandomForest] {task_key}, n={n} subjects", fontsize=10)
    axes[0].legend(fontsize=8)
    for i, (b, c) in enumerate(zip(b_vals, c_vals)):
        if b is not None:
            axes[0].annotate(f"{b:.2f}", (i - 0.2, b), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
        if c is not None:
            axes[0].annotate(f"{c:.2f}", (i + 0.2, c), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)

    auc_b, auc_c = overall(m_raw, "auc"), overall(m_c6, "auc")
    axes[1].bar([0], [auc_b or 0], 0.5, color=COLOR_BASE, label="baseline")
    axes[1].bar([1], [auc_c or 0], 0.5, color=COLOR_PERS, label="centered_6")
    axes[1].axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance (0.5)")
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["baseline", "centered_6"])
    axes[1].set_ylim(0, 1.0)
    axes[1].set_title(f"[RandomForest] AUC — {task_key}", fontsize=10)
    axes[1].legend(fontsize=8)
    if auc_b is not None:
        axes[1].annotate(f"{auc_b:.3f}", (0, auc_b), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9)
    if auc_c is not None:
        axes[1].annotate(f"{auc_c:.3f}", (1, auc_c), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{task_key}_sens_auc_merged.png"), dpi=150)
    plt.close(fig)

    # 圖2：逐人 AUC 對照
    aucs_b = np.array([m_raw[u]["auc"] if m_raw[u]["auc"] is not None else np.nan for u in uuids])
    aucs_c = np.array([m_c6[u]["auc"] if m_c6[u]["auc"] is not None else np.nan for u in uuids])
    order_idx = np.argsort(np.nan_to_num(aucs_b, nan=-1))
    xx = np.arange(n)
    fig, ax = plt.subplots(figsize=(max(9, n * 0.4), 5))
    ax.bar(xx - 0.2, aucs_b[order_idx], 0.4, label="baseline", color=COLOR_BASE)
    ax.bar(xx + 0.2, aucs_c[order_idx], 0.4, label="centered_6", color=COLOR_PERS)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance (0.5)")
    ax.set_xticks(xx)
    ax.set_xticklabels([uuids[i] for i in order_idx], rotation=60, ha="right", fontsize=6)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("AUC")
    ax.set_title(f"[RandomForest] Per-subject AUC — {task_key} (sorted by baseline, n={n})", fontsize=10)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{task_key}_per_subject_merged.png"), dpi=150)
    plt.close(fig)

    # 圖3：逐人 sensitivity 對照
    sens_b = np.array([m_raw[u]["recall"] if m_raw[u]["recall"] is not None else np.nan for u in uuids])
    sens_c = np.array([m_c6[u]["recall"] if m_c6[u]["recall"] is not None else np.nan for u in uuids])
    order_idx2 = np.argsort(np.nan_to_num(sens_b, nan=-1))
    fig, ax = plt.subplots(figsize=(max(9, n * 0.4), 5))
    ax.bar(xx - 0.2, sens_b[order_idx2], 0.4, label="baseline", color=COLOR_BASE)
    ax.bar(xx + 0.2, sens_c[order_idx2], 0.4, label="centered_6", color=COLOR_PERS)
    ax.set_xticks(xx)
    ax.set_xticklabels([uuids[i] for i in order_idx2], rotation=60, ha="right", fontsize=6)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(f"sensitivity (recall, {pos_label})")
    ax.set_title(f"[RandomForest] Per-subject sensitivity ({pos_label} recall) — {task_key} (n={n})", fontsize=10)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{task_key}_per_subject_sensitivity.png"), dpi=150)
    plt.close(fig)


def write_full_excel(task_metrics, out_path):
    """task_metrics: {task_key: (m_raw, m_c6, uuids)}。每個指標一個分頁(逐task平均對照)，
    最後一個分頁是逐人逐task的完整指標。"""
    with pd.ExcelWriter(out_path) as writer:
        for k in METRIC_KEYS:
            rows = []
            for task_key, (m_raw, m_c6, uuids) in task_metrics.items():
                b = [m_raw[u][k] for u in uuids if m_raw[u][k] is not None]
                c = [m_c6[u][k] for u in uuids if m_c6[u][k] is not None]
                rows.append({
                    "task": task_key, "n_subjects": len(uuids),
                    "baseline_mean": float(np.mean(b)) if b else None,
                    "baseline_std": float(np.std(b)) if b else None,
                    "centered6_mean": float(np.mean(c)) if c else None,
                    "centered6_std": float(np.std(c)) if c else None,
                    "delta(centered6-baseline)": (float(np.mean(c)) - float(np.mean(b))) if b and c else None,
                })
            pd.DataFrame(rows).to_excel(writer, sheet_name=METRIC_LABELS[k], index=False)

        rows = []
        for task_key, (m_raw, m_c6, uuids) in task_metrics.items():
            for u in uuids:
                row = {"uuid": u, "task": task_key}
                for k in METRIC_KEYS:
                    row[f"baseline_{METRIC_LABELS[k]}"] = m_raw[u][k]
                for k in METRIC_KEYS:
                    row[f"centered6_{METRIC_LABELS[k]}"] = m_c6[u][k]
                rows.append(row)
        pd.DataFrame(rows).to_excel(writer, sheet_name="per_subject", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=["all", "17"], required=True,
                     help="all=完整~96個手工特徵, 17=Regression_Model_Predictor_meta_OOF.py 篩選出的17個")
    args = ap.parse_args()

    print("讀取事件層級特徵...")
    data, all_feat_cols = load_event_level()

    if args.features == "17":
        missing = [f for f in SELECTED_17 if f not in all_feat_cols]
        if missing:
            raise ValueError(f"17個特徵裡有些欄位在資料裡不存在: {missing}")
        feat_cols = SELECTED_17
        out_subdir = "17"
    else:
        feat_cols = all_feat_cols
        out_subdir = "96"

    print(f"使用 {len(feat_cols)} 個特徵（{'全部' if args.features == 'all' else '篩選過的17個'}），"
          f"輸出到 output/handcrafted_classification/{out_subdir}/")

    out_dir = os.path.join(OUT_DIR, out_subdir)
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
        x = np.arange(len(MODES))
        ax.bar(x - 0.2, vals_ps, 0.4, label="per-subject AUC", color=COLOR_PERS)
        ax.bar(x + 0.2, vals_pool, 0.4, label="pooled AUC", color=COLOR_BASE)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance (0.5)")
        ax.set_xticks(x)
        ax.set_xticklabels(MODES)
        ax.set_ylim(0, 1.0)
        ax.set_title(f"[RandomForest, {len(feat_cols)}特徵] {task_name}", fontsize=10)
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
        m_raw = per_subject_metrics(d["y"], d["subj"], d["score_raw"], d["thr_raw"])
        m_c6 = per_subject_metrics(d["y"], d["subj"], d["score_c6"], d["thr_c6"])
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

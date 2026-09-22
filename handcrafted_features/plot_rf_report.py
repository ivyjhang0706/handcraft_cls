# -*- coding: utf-8 -*-
"""
Random Forest 版報表整合腳本——取代原本分散的做法（一張圖一支程式：
plot_handcrafted_merged_report_rf.py 只出 sens_auc_merged_rf.png / per_subject_merged_rf.png，
per_subject_sensitivity_rf.png 跟 rf_per_subject_metrics.xlsx 另外用別的程式產生），
現在統一從同一份 5-fold 預測結果算出全部圖表跟 Excel，只需跑這一支。

背景：CGM+BGM 合併訓練、只用 BGM 評估（部署現實），RandomForest（跟
experiment_handcrafted_model_compare.py 的 make_model("random_forest") 同參數：
n_estimators=300, max_depth=6, class_weight="balanced", random_state=0），
比較 baseline(raw) vs centered_6 兩種做法。

另外依 32 位已確診糖尿病(DM)受試者名單（使用者提供），把可評估受試者拆成
DM / Normal 兩組，各自再輸出一份同樣的圖表組合，方便比較兩組的表現是否不同。

輸出全部收在 output/handcrafted_classification/rf/ 底下，不再散落在上層：
  rf/
    sens_auc_merged_rf.png             全體：Normal/High sensitivity + AUC 平均
    per_subject_merged_rf.png          全體：逐人 AUC 對照
    per_subject_sensitivity_rf.png     全體：逐人 sensitivity(High recall) 對照
    rf_per_subject_metrics.xlsx        逐人指標(per_subject分頁，含 group 欄位) + 平均對照(summary分頁)
    merged_report_metrics_rf.json      逐人原始指標 + 分組結果
    DM/       同樣三張圖，只用 DM 組受試者
    normal/   同樣三張圖，只用 Normal(非DM) 組受試者

用法：
  conda run -n ml python kfold/handcrafted_features/plot_rf_report.py
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

_KFOLD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in sorted(os.listdir(_KFOLD_ROOT)):
    _p = os.path.join(_KFOLD_ROOT, _d)
    if os.path.isdir(_p):
        sys.path.insert(0, _p)

from experiment_handcrafted_bgm import load_merged, OUT_DIR  # noqa: E402
from experiment_handcrafted_classification import (  # noqa: E402
    K_FOLDS, SEED, N_CALIB_EVENTS, center_by_subject, pick_threshold,
)
from evaluate_normal_vs_high_k5 import binary_metrics, METRIC_KEYS, METRIC_LABELS  # noqa: E402

POS_LEVEL, NEG_LEVELS = "High", ["Normal"]
MIN_PER_SIDE = 3
COLOR_BASE = "#9fb8d1"
COLOR_PERS = "#e0824a"
RF_OUT_DIR = os.path.join(OUT_DIR, "rf")

# 已確診糖尿病(DM)受試者 uuid，共32人（使用者提供）；其餘可評估受試者視為 Normal 組。
DM_UUIDS = {
    "2200", "2218", "2227", "2235", "2248", "2249", "2251", "2253", "2261", "2262",
    "2257", "2279", "2276", "2278", "2281", "2286", "2285", "2294", "2293", "2295",
    "2291", "2298", "2300", "2299", "2304", "2305", "2306", "2322", "2329", "2352",
    "2378", "2397",
}


def make_rf():
    return RandomForestClassifier(n_estimators=300, max_depth=6, class_weight="balanced",
                                   random_state=0, n_jobs=1)


def group_of(uuid):
    return "DM" if str(uuid) in DM_UUIDS else "Normal"


def build_predictions():
    """跑5折，蒐集每個測試row的 raw / centered_6 分數（RandomForest predict_proba 正類機率）。"""
    data, feat_cols, _ = load_merged()
    sub = data[data["BG_Level"].isin([POS_LEVEL] + NEG_LEVELS)].reset_index(drop=True)
    X = sub[feat_cols].to_numpy(dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = (sub["BG_Level"] == POS_LEVEL).astype(int).to_numpy()
    subj = sub["uuid"].to_numpy()
    src = sub["source"].to_numpy()
    uuids = np.array(sorted(set(subj)))
    print(f"事件數={len(y):,}  正類(High)={y.sum():,}  人數={len(uuids)}")

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

    score_raw = np.full(len(y), np.nan)
    score_c6 = np.full(len(y), np.nan)
    thr_raw = np.full(len(y), np.nan)
    thr_c6 = np.full(len(y), np.nan)
    evaluated = np.zeros(len(y), bool)

    for f in range(K_FOLDS):
        te_u = {u for u in uuids if folds[u] == f}
        tr = ~np.isin(subj, list(te_u))
        te = np.isin(subj, list(te_u)) & (src == "BGM")  # 只評估BGM(部署現實)

        sc = StandardScaler().fit(X[tr])
        clf = make_rf()
        clf.fit(sc.transform(X[tr]), y[tr])
        score_raw[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
        t_raw = pick_threshold(clf.predict_proba(sc.transform(X[tr]))[:, 1], y[tr])
        thr_raw[te] = t_raw

        Xc = X.copy()
        Xc[tr] = center_by_subject(X[tr], subj[tr])
        Xc[te] = center_by_subject(X[te], subj[te], ref_idx=calib_mask[te])
        sc2 = StandardScaler().fit(Xc[tr])
        clf2 = make_rf()
        clf2.fit(sc2.transform(Xc[tr]), y[tr])
        score_c6[te] = clf2.predict_proba(sc2.transform(Xc[te]))[:, 1]
        t_c6 = pick_threshold(clf2.predict_proba(sc2.transform(Xc[tr]))[:, 1], y[tr])
        thr_c6[te] = t_c6

        evaluated[te] = True
        print(f"  fold {f}: 評估 {te.sum()} 筆事件（{len(te_u)} 人待評估）"
              f"  threshold(raw)={t_raw:.3f}  threshold(centered_6)={t_c6:.3f}")

    return (y[evaluated], subj[evaluated], score_raw[evaluated], score_c6[evaluated],
            thr_raw[evaluated], thr_c6[evaluated])


def per_subject_metrics(y, subj, score, thr):
    """回傳 {uuid: {metric: value}}，只保留兩類都 >= MIN_PER_SIDE 的人。"""
    out = {}
    for u in np.unique(subj):
        m = subj == u
        if min((y[m] == 0).sum(), (y[m] == 1).sum()) < MIN_PER_SIDE:
            continue
        out[u] = binary_metrics(y[m], score[m], threshold=float(thr[m][0]))
    return out


def plot_reports(m_raw, m_c6, uuids, out_dir, label_suffix=""):
    """把三張對照圖畫進 out_dir：sens_auc / per-subject AUC / per-subject sensitivity。"""
    os.makedirs(out_dir, exist_ok=True)
    n = len(uuids)

    def overall(metrics_dict, key):
        vals = [metrics_dict[u][key] for u in uuids if metrics_dict[u][key] is not None]
        return float(np.mean(vals)) if vals else None

    # ---------- 圖1：Normal/High sensitivity + AUC ----------
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    b_vals = [overall(m_raw, "specificity"), overall(m_raw, "recall")]
    c_vals = [overall(m_c6, "specificity"), overall(m_c6, "recall")]
    x = np.arange(2)
    axes[0].bar(x - 0.2, [v or 0 for v in b_vals], 0.4, label="baseline", color=COLOR_BASE)
    axes[0].bar(x + 0.2, [v or 0 for v in c_vals], 0.4, label="centered_6", color=COLOR_PERS)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["Normal", "High"])
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("sensitivity (recall)")
    axes[0].set_title(f"[Random Forest]{label_suffix} Merged train (CGM+BGM) / BGM-only eval\n"
                       f"Normal vs High, n={n} subjects", fontsize=10)
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
    axes[1].set_title(f"[Random Forest]{label_suffix} AUC (discrimination ability)", fontsize=10)
    axes[1].legend(fontsize=8)
    if auc_b is not None:
        axes[1].annotate(f"{auc_b:.3f}", (0, auc_b), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9)
    if auc_c is not None:
        axes[1].annotate(f"{auc_c:.3f}", (1, auc_c), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "sens_auc_merged_rf.png"), dpi=150)
    plt.close(fig)

    # ---------- 圖2：per-subject AUC 對照 ----------
    aucs_b = np.array([m_raw[u]["auc"] if m_raw[u]["auc"] is not None else np.nan for u in uuids])
    aucs_c = np.array([m_c6[u]["auc"] if m_c6[u]["auc"] is not None else np.nan for u in uuids])
    order_idx = np.argsort(np.nan_to_num(aucs_b, nan=-1))
    xx = np.arange(n)
    fig, ax = plt.subplots(figsize=(max(9, n * 0.5), 5))
    ax.bar(xx - 0.2, aucs_b[order_idx], 0.4, label="baseline", color=COLOR_BASE)
    ax.bar(xx + 0.2, aucs_c[order_idx], 0.4, label="centered_6", color=COLOR_PERS)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance (0.5)")
    ax.set_xticks(xx)
    ax.set_xticklabels([uuids[i] for i in order_idx], rotation=60, ha="right", fontsize=7)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("AUC")
    ax.set_title(f"[Random Forest]{label_suffix} Per-subject AUC: baseline vs centered_6 "
                 f"(sorted by baseline, n={n})", fontsize=10)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "per_subject_merged_rf.png"), dpi=150)
    plt.close(fig)

    # ---------- 圖3：per-subject sensitivity(High recall) 對照 ----------
    sens_b = np.array([m_raw[u]["recall"] if m_raw[u]["recall"] is not None else np.nan for u in uuids])
    sens_c = np.array([m_c6[u]["recall"] if m_c6[u]["recall"] is not None else np.nan for u in uuids])
    order_idx2 = np.argsort(np.nan_to_num(sens_b, nan=-1))
    fig, ax = plt.subplots(figsize=(max(9, n * 0.5), 5))
    ax.bar(xx - 0.2, sens_b[order_idx2], 0.4, label="baseline", color=COLOR_BASE)
    ax.bar(xx + 0.2, sens_c[order_idx2], 0.4, label="centered_6", color=COLOR_PERS)
    ax.set_xticks(xx)
    ax.set_xticklabels([uuids[i] for i in order_idx2], rotation=60, ha="right", fontsize=7)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("sensitivity (recall, High)")
    ax.set_title(f"[Random Forest]{label_suffix} Per-subject sensitivity (High recall, tuned threshold): "
                 f"baseline vs centered_6 (sorted by baseline, n={n})", fontsize=10)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "per_subject_sensitivity_rf.png"), dpi=150)
    plt.close(fig)


def write_excel(m_raw, m_c6, uuids, out_path):
    rows = []
    for u in uuids:
        row = {"uuid": u, "group": group_of(u)}
        for k in METRIC_KEYS:
            row[f"baseline_{METRIC_LABELS[k]}"] = m_raw[u][k]
        for k in METRIC_KEYS:
            row[f"centered6_{METRIC_LABELS[k]}"] = m_c6[u][k]
        rows.append(row)
    per_subject_df = pd.DataFrame(rows)

    summary_rows = []
    for k in METRIC_KEYS:
        b = [m_raw[u][k] for u in uuids if m_raw[u][k] is not None]
        c = [m_c6[u][k] for u in uuids if m_c6[u][k] is not None]
        summary_rows.append({
            "指標": METRIC_LABELS[k],
            "baseline_mean": float(np.mean(b)) if b else None,
            "baseline_std": float(np.std(b)) if b else None,
            "centered6_mean": float(np.mean(c)) if c else None,
            "centered6_std": float(np.std(c)) if c else None,
            "delta(centered6-baseline)": (float(np.mean(c)) - float(np.mean(b))) if b and c else None,
        })
    summary_df = pd.DataFrame(summary_rows)

    with pd.ExcelWriter(out_path) as writer:
        per_subject_df.to_excel(writer, sheet_name="per_subject", index=False)
        summary_df.to_excel(writer, sheet_name="summary", index=False)


def main():
    print("讀取合併資料、跑5折預測(Random Forest, raw + centered_6)...")
    y, subj, score_raw, score_c6, thr_raw, thr_c6 = build_predictions()

    m_raw = per_subject_metrics(y, subj, score_raw, thr_raw)
    m_c6 = per_subject_metrics(y, subj, score_c6, thr_c6)
    common_uuids = sorted(set(m_raw) & set(m_c6))
    groups = {u: group_of(u) for u in common_uuids}
    n_dm = sum(1 for g in groups.values() if g == "DM")
    print(f"\n可評估人數(Normal/High各>=3筆) = {len(common_uuids)}"
          f"（DM組 n={n_dm}，Normal組 n={len(common_uuids) - n_dm}）")

    os.makedirs(RF_OUT_DIR, exist_ok=True)

    plot_reports(m_raw, m_c6, common_uuids, RF_OUT_DIR)
    write_excel(m_raw, m_c6, common_uuids, os.path.join(RF_OUT_DIR, "rf_per_subject_metrics.xlsx"))
    with open(os.path.join(RF_OUT_DIR, "merged_report_metrics_rf.json"), "w", encoding="utf-8") as fp:
        json.dump({"baseline": {u: m_raw[u] for u in common_uuids},
                    "centered_6": {u: m_c6[u] for u in common_uuids},
                    "group": groups}, fp, ensure_ascii=False, indent=2)
    print(f"寫入 {RF_OUT_DIR}（全體 n={len(common_uuids)}）")

    for group_name, subdir in [("DM", "DM"), ("Normal", "normal")]:
        g_uuids = [u for u in common_uuids if groups[u] == group_name]
        if not g_uuids:
            print(f"  {group_name} 組沒有可評估的人，略過")
            continue
        g_dir = os.path.join(RF_OUT_DIR, subdir)
        plot_reports(m_raw, m_c6, g_uuids, g_dir, label_suffix=f" [{group_name}]")
        print(f"寫入 {g_dir}（{group_name} n={len(g_uuids)}）")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
把「CGM+BGM合併訓練、只用BGM評估」Normal vs High(全時段，不排除晝夜——使用者指示先
不管這個confound，避免過度排除把真訊號也濾掉)的 raw(baseline，無個人化) vs
centered_6(用這個人自己6筆BGM校正事件做個人化) 結果，畫成跟舊版深度特徵報告
(`kfold/evaluate_normal_vs_high_k5.py` 的 `normal_vs_high_k5.png`)同樣風格的圖，
方便直接跟先前的深度特徵結果對照。

輸出三張圖到 output/handcrafted_classification/：
  1. sens_auc_merged.png      ：左＝Normal/High各自的recall(sensitivity)，右＝AUC
  2. per_subject_merged.png   ：每個人一組(baseline vs centered_6)的AUC長條圖
  3. all_metrics_merged.png   ：AUC/Accuracy/Precision/Recall/Specificity/F1 全指標對照

用法：
  conda run -n ml python kfold/handcrafted_features/plot_handcrafted_merged_report.py
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
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


def build_predictions():
    """跑5折，蒐集每個測試row的 raw / centered_6 分數（不只是彙總的AUC）。"""
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

        # raw
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.1)
        clf.fit(sc.transform(X[tr]), y[tr])
        score_raw[te] = clf.decision_function(sc.transform(X[te]))
        # 門檻只能用訓練組(population)自己的分數選，不能碰待評估的人
        t_raw = pick_threshold(clf.decision_function(sc.transform(X[tr])), y[tr])
        thr_raw[te] = t_raw

        # centered_6
        Xc = X.copy()
        Xc[tr] = center_by_subject(X[tr], subj[tr])
        Xc[te] = center_by_subject(X[te], subj[te], ref_idx=calib_mask[te])
        sc2 = StandardScaler().fit(Xc[tr])
        clf2 = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.1)
        clf2.fit(sc2.transform(Xc[tr]), y[tr])
        score_c6[te] = clf2.decision_function(sc2.transform(Xc[te]))
        t_c6 = pick_threshold(clf2.decision_function(sc2.transform(Xc[tr])), y[tr])
        thr_c6[te] = t_c6

        evaluated[te] = True
        print(f"  fold {f}: 評估 {te.sum()} 筆事件（{len(te_u)} 人待評估）"
              f"  threshold(raw)={t_raw:.3f}  threshold(centered_6)={t_c6:.3f}")

    return (y[evaluated], subj[evaluated], score_raw[evaluated], score_c6[evaluated],
            thr_raw[evaluated], thr_c6[evaluated])


def bootstrap_subject_auc_ci(y_sub, score_sub, n_boot=2000, seed=0):
    """對單一受試者的事件，用bootstrap重抽估計AUC的95% CI（反映這個人自己事件數
    有限帶來的不確定性，不是跨人變異）。回傳 (lo, hi)；抽到單一類別的重複樣本直接跳過。"""
    rng = np.random.RandomState(seed)
    n = len(y_sub)
    boot_aucs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yy = y_sub[idx]
        if len(np.unique(yy)) < 2:
            continue
        boot_aucs.append(roc_auc_score(yy, score_sub[idx]))
    if len(boot_aucs) < n_boot * 0.5:
        return None, None
    lo, hi = np.percentile(boot_aucs, [2.5, 97.5])
    return float(lo), float(hi)


def per_subject_metrics(y, subj, score, thr):
    """回傳 {uuid: {metric: value}}，只保留兩類都 >= MIN_PER_SIDE 的人。
    thr：每筆事件對應的門檻(同一人的事件都屬於同一折，門檻是同一個值)，
    用訓練組資料選出，取代寫死的 0.0。"""
    out = {}
    for u in np.unique(subj):
        m = subj == u
        if min((y[m] == 0).sum(), (y[m] == 1).sum()) < MIN_PER_SIDE:
            continue
        out[u] = binary_metrics(y[m], score[m], threshold=float(thr[m][0]))
    return out


def main():
    print("讀取合併資料、跑5折預測(raw + centered_6)...")
    y, subj, score_raw, score_c6, thr_raw, thr_c6 = build_predictions()

    m_raw = per_subject_metrics(y, subj, score_raw, thr_raw)
    m_c6 = per_subject_metrics(y, subj, score_c6, thr_c6)
    common_uuids = sorted(set(m_raw) & set(m_c6))
    print(f"\n可評估人數(Normal/High各>=3筆) = {len(common_uuids)}")

    def overall(metrics_dict, key):
        vals = [metrics_dict[u][key] for u in common_uuids if metrics_dict[u][key] is not None]
        return float(np.mean(vals)) if vals else None

    print(f"\n{'指標':16}{'baseline':>12}{'centered_6':>14}")
    for k in METRIC_KEYS:
        b, c = overall(m_raw, k), overall(m_c6, k)
        f_ = lambda v: f"{v:.4f}" if v is not None else "   N/A"
        print(f"{METRIC_LABELS[k]:16}{f_(b):>12}{f_(c):>14}")

    os.makedirs(OUT_DIR, exist_ok=True)

    # ---------- 圖1：Normal/High sensitivity + AUC ----------
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    normal_recall_b = overall(m_raw, "specificity")   # Normal正確率 = High視角下的specificity
    normal_recall_c = overall(m_c6, "specificity")
    high_recall_b = overall(m_raw, "recall")          # High正確率 = sensitivity
    high_recall_c = overall(m_c6, "recall")
    x = np.arange(2)
    b_vals = [normal_recall_b, high_recall_b]
    c_vals = [normal_recall_c, high_recall_c]
    axes[0].bar(x - 0.2, b_vals, 0.4, label="baseline", color=COLOR_BASE)
    axes[0].bar(x + 0.2, c_vals, 0.4, label="centered_6", color=COLOR_PERS)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["Normal", "High"])
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("sensitivity (recall)")
    axes[0].set_title(f"Merged train (CGM+BGM) / BGM-only eval\nNormal vs High, n={len(common_uuids)} subjects", fontsize=10)
    axes[0].legend(fontsize=8)
    for i, (b, c) in enumerate(zip(b_vals, c_vals)):
        axes[0].annotate(f"{b:.2f}", (i - 0.2, b), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
        axes[0].annotate(f"{c:.2f}", (i + 0.2, c), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)

    auc_b, auc_c = overall(m_raw, "auc"), overall(m_c6, "auc")
    axes[1].bar([0], [auc_b], 0.5, color=COLOR_BASE, label="baseline")
    axes[1].bar([1], [auc_c], 0.5, color=COLOR_PERS, label="centered_6")
    axes[1].axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance (0.5)")
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["baseline", "centered_6"])
    axes[1].set_ylim(0, 1.0)
    axes[1].set_title("AUC (discrimination ability)", fontsize=10)
    axes[1].legend(fontsize=8)
    axes[1].annotate(f"{auc_b:.3f}", (0, auc_b), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9)
    axes[1].annotate(f"{auc_c:.3f}", (1, auc_c), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9)
    plt.tight_layout()
    p1 = os.path.join(OUT_DIR, "sens_auc_merged.png")
    plt.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"寫入 {p1}")

    # ---------- 圖2：per-subject AUC 對照 ----------
    aucs_b = np.array([m_raw[u]["auc"] for u in common_uuids])
    aucs_c = np.array([m_c6[u]["auc"] for u in common_uuids])
    order_idx = np.argsort(aucs_b)
    fig, ax = plt.subplots(figsize=(max(9, len(common_uuids) * 0.5), 5))
    xx = np.arange(len(common_uuids))
    ax.bar(xx - 0.2, aucs_b[order_idx], 0.4, label="baseline", color=COLOR_BASE)
    ax.bar(xx + 0.2, aucs_c[order_idx], 0.4, label="centered_6", color=COLOR_PERS)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance (0.5)")
    ax.set_xticks(xx)
    ax.set_xticklabels([common_uuids[i] for i in order_idx], rotation=60, ha="right", fontsize=7)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("AUC")
    ax.set_title(f"Per-subject AUC: baseline vs centered_6 (sorted by baseline, n={len(common_uuids)})", fontsize=10)
    ax.legend(fontsize=8)
    plt.tight_layout()
    p2 = os.path.join(OUT_DIR, "per_subject_merged.png")
    plt.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"寫入 {p2}")

    # ---------- 圖3：全指標對照 ----------
    fig, ax = plt.subplots(figsize=(9, 5))
    xk = np.arange(len(METRIC_KEYS))
    bvals = [overall(m_raw, k) for k in METRIC_KEYS]
    cvals = [overall(m_c6, k) for k in METRIC_KEYS]
    ax.bar(xk - 0.2, bvals, 0.4, label="baseline", color=COLOR_BASE)
    ax.bar(xk + 0.2, cvals, 0.4, label="centered_6", color=COLOR_PERS)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance (0.5)")
    ax.set_xticks(xk)
    ax.set_xticklabels([METRIC_LABELS[k] for k in METRIC_KEYS], rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"Merged train (CGM+BGM) / BGM-only eval\nNormal vs High, per-subject mean (n={len(common_uuids)})", fontsize=10)
    ax.legend(fontsize=8)
    for i, (b, c) in enumerate(zip(bvals, cvals)):
        if b is not None:
            ax.annotate(f"{b:.2f}", (i - 0.2, b), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7)
        if c is not None:
            ax.annotate(f"{c:.2f}", (i + 0.2, c), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7)
    plt.tight_layout()
    p3 = os.path.join(OUT_DIR, "all_metrics_merged.png")
    plt.savefig(p3, dpi=150)
    plt.close(fig)
    print(f"寫入 {p3}")

    with open(os.path.join(OUT_DIR, "merged_report_metrics.json"), "w", encoding="utf-8") as fp:
        json.dump({"baseline": {u: m_raw[u] for u in common_uuids},
                    "centered_6": {u: m_c6[u] for u in common_uuids}}, fp, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Learning curve：per-subject AUC 對「訓練受試者人數」的曲線。

回答的問題：現在的績效是被「受試者人數不夠」卡住，還是早就飽和了？
這決定了「去處理外部資料集的 domain gap」值不值得花力氣——在投入時間下載/對齊外部
資料之前，先用現有的 62 人問出答案。

  曲線到最大人數還在往上爬  → 人數是瓶頸，外部資料值得追
  曲線早就打平             → 再多人也沒用，該回頭質疑訊號源本身

為什麼不是用 burst augmentation 那次的結果來回答：那次把訓練列數從 3.6 萬拉到 26 萬
(7倍)完全沒有改善，但那些列全部來自同樣的 62 個人，同一事件切出來的 burst 高度相關，
不是新資訊。這支腳本變動的是「**不同的人**」，是質性不同的維度。

實驗設計（關鍵在評估端必須完全固定，否則等於同時改了兩件事）：
  - fold 切分、held-out 受試者、評估事件集合，在所有訓練人數設定下**完全相同**
  - 每一折裡，訓練池 = 不在該折的受試者(約 50 人)，從中隨機抽 K 個人來訓練
  - 只有「抽幾個人」在變，其他一切不變 → 曲線上的差異可歸因於訓練人數

  每個 K 值重複抽樣多次(--repeats)取平均，因為 K 小的時候「抽到哪幾個人」影響很大；
  K = 訓練池全滿時不需要重複(每次都是同一批人)，自動只跑一次。

已知限制：K 變大時訓練「事件數」也跟著變多，所以「人數」與「資料量」兩個因素是綁在
一起的，這支腳本不拆開它們(要拆的話需要在增加人數時同步降採樣事件數，讓總事件數固定)。
輸出有印出各設定的訓練事件數，判讀時要意識到這個綁定。

評估一律排除校正事件(跟 README 對外報的 0.614 同一個口徑)。

用法：
  conda run -n ml python kfold/handcrafted_features/experiment_learning_curve.py
  conda run -n ml python kfold/handcrafted_features/experiment_learning_curve.py --sizes 10,20,30,40,50 --repeats 5
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

_KFOLD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in sorted(os.listdir(_KFOLD_ROOT)):
    _p = os.path.join(_KFOLD_ROOT, _d)
    if os.path.isdir(_p):
        sys.path.insert(0, _p)

from experiment_handcrafted_bgm import load_merged  # noqa: E402
from experiment_handcrafted_classification import (  # noqa: E402
    center_by_subject, per_subject_auc,
)
from config import K_FOLDS, SEED, N_CALIB_EVENTS, MIN_PER_SIDE, OUT_DIR  # noqa: E402

POS_LEVEL, NEG_LEVELS = "High", ["Normal"]
RF_PARAMS = dict(n_estimators=300, max_depth=6, class_weight="balanced",
                 random_state=0, n_jobs=8)
MODES = ["raw", "centered_6"]


def make_calib(subj, src, uuids):
    """每個人 N_CALIB_EVENTS 個隨機校正事件(BGM優先)，跟現行管線的抽樣邏輯一致。
    這批事件不隨訓練人數改變，所以不影響曲線上各點的可比性。"""
    rng = np.random.RandomState(SEED)
    calib = np.zeros(len(subj), bool)
    for u in uuids:
        pool = np.where(subj == u)[0]
        pool_bgm = pool[src[pool] == "BGM"]
        pool = pool_bgm if len(pool_bgm) > 0 else pool
        pick = rng.choice(pool, size=min(N_CALIB_EVENTS, len(pool)), replace=False)
        calib[pick] = True
    return calib


def run_once(X, y, subj, src, uuids, folds, calib, n_train, rep, mode):
    """跑完整 5 折，每折從訓練池隨機抽 n_train 個人來訓練。

    n_train=None 代表用訓練池全部的人（曲線最右邊那一點，等同現行做法）。
    回傳 (score, n_train_subjects_used, n_train_events_used)。
    """
    score = np.full(len(y), np.nan)
    used_subjects, used_events = [], []
    rng = np.random.RandomState(SEED + 1000 * rep + (n_train or 0))

    for f in range(K_FOLDS):
        te_u = [u for u in uuids if folds[u] == f]
        pool_u = np.array([u for u in uuids if folds[u] != f])

        if n_train is None or n_train >= len(pool_u):
            tr_u = pool_u
        else:
            tr_u = rng.choice(pool_u, size=n_train, replace=False)

        tr = np.isin(subj, tr_u)
        te = np.isin(subj, te_u) & (src == "BGM")   # 只評估 BGM(部署現實)
        if te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        used_subjects.append(len(tr_u))
        used_events.append(int(tr.sum()))

        if mode == "raw":
            Xtr, Xte = X[tr], X[te]
        else:  # centered_6
            Xc = X.copy()
            Xc[tr] = center_by_subject(X[tr], subj[tr])
            Xc[te] = center_by_subject(X[te], subj[te], ref_idx=calib[te])
            Xtr, Xte = Xc[tr], Xc[te]

        sc = StandardScaler().fit(Xtr)
        clf = RandomForestClassifier(**RF_PARAMS)
        clf.fit(sc.transform(Xtr), y[tr])
        score[te] = clf.predict_proba(sc.transform(Xte))[:, 1]

    return score, float(np.mean(used_subjects)), float(np.mean(used_events))


def evaluate(y, subj, score, eval_mask):
    """回傳 (per_subject_auc, n_subjects, pooled_auc)，只看 eval_mask 內的事件。"""
    done = (~np.isnan(score)) & eval_mask
    yy, ss, uu = y[done], score[done], subj[done]
    if len(np.unique(yy)) < 2:
        return None, 0, None
    ps, n, _ = per_subject_auc(yy, ss, uu)
    pooled = float(roc_auc_score(yy, ss))
    return ps, n, pooled


def write_summary(path, rows, args, meta):
    """照專案慣例寫一份表格化的 summary.txt。"""
    lines = []
    lines.append("experiment_learning_curve.py")
    lines.append(f"CGM_ROOT / BGM_ROOT 見 config.py；repeats={args.repeats}，"
                 f"sizes={args.sizes}")
    lines.append("")
    lines.append("問題：per-subject AUC 對「訓練受試者人數」的曲線，還在爬還是已經飽和。")
    lines.append("評估端在所有設定下完全固定（同樣的 fold、held-out 受試者、評估事件），")
    lines.append("只有訓練池抽幾個人在變。評估排除校正事件。")
    lines.append("")
    lines.append(f"總事件數={meta['n_events']:,}，正類(High)={meta['n_pos']:,}，"
                 f"總人數={meta['n_subjects']}")
    lines.append(f"排除校正事件 {meta['n_calib']} 個後，可評估事件 "
                 f"{meta['n_eval_events']:,}/{meta['n_events']:,}")
    lines.append("")

    df = pd.DataFrame(rows)
    for mode in MODES:
        sub = df[df["mode"] == mode]
        if sub.empty:
            continue
        lines.append(f"[{mode}]")
        lines.append("")
        lines.append("| 訓練人數 | 平均訓練事件數 | per-subject AUC (mean±std) | pooled AUC | 可評估人數 | 重複次數 |")
        lines.append("|----------|----------------|----------------------------|------------|------------|----------|")
        for _, r in sub.iterrows():
            auc_s = (f"{r['auc_mean']:.4f} ± {r['auc_std']:.4f}"
                     if r["auc_mean"] is not None else "N/A")
            pool_s = f"{r['pooled_mean']:.4f}" if r["pooled_mean"] is not None else "N/A"
            lines.append(f"| {r['n_train_subjects']:.0f} | {r['n_train_events']:,.0f} | "
                         f"{auc_s} | {pool_s} | {r['n_eval_subjects']:.0f} | {r['n_reps']} |")
        lines.append("")

    # 自動判讀：比較最後兩點的差距與重複抽樣的雜訊水準
    lines.append("判讀：")
    for mode in MODES:
        sub = df[df["mode"] == mode].sort_values("n_train_subjects")
        if len(sub) < 2:
            continue
        last, prev = sub.iloc[-1], sub.iloc[-2]
        if last["auc_mean"] is None or prev["auc_mean"] is None:
            continue
        delta = last["auc_mean"] - prev["auc_mean"]
        noise = max(float(prev["auc_std"] or 0), 0.005)
        verdict = ("最後一段仍在上升，人數可能還是瓶頸"
                   if delta > noise else
                   "最後一段已經打平（漲幅小於重複抽樣的雜訊），再加人預期幫助有限")
        lines.append(f"  [{mode}] {prev['n_train_subjects']:.0f}人 → "
                     f"{last['n_train_subjects']:.0f}人：{delta:+.4f}"
                     f"（重複抽樣 std≈{prev['auc_std'] or 0:.4f}）→ {verdict}")
    lines.append("")
    lines.append("注意：訓練人數與訓練事件數是綁在一起變動的，這支腳本沒有拆開兩者。")

    with open(path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="10,20,30,40,50",
                    help="訓練人數清單，逗號分隔；訓練池全滿那一點會自動補上")
    ap.add_argument("--repeats", type=int, default=3,
                    help="每個人數設定重複抽樣幾次取平均（全滿那點自動只跑1次）")
    args = ap.parse_args()

    out_dir = os.path.join(OUT_DIR, "learning_curve")
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

    calib = make_calib(subj, src, uuids)
    eval_mask = ~calib
    print(f"排除校正事件 {calib.sum()} 個後，可評估事件 {eval_mask.sum():,}/{len(y):,}")

    pool_sizes = [len([u for u in uuids if folds[u] != f]) for f in range(K_FOLDS)]
    max_pool = min(pool_sizes)
    sizes = sorted({int(s) for s in args.sizes.split(",") if int(s) < max_pool})
    sizes.append(None)   # None = 訓練池全滿（現行做法）
    print(f"訓練池大小(各折) = {pool_sizes}，曲線取樣點 = "
          f"{[s if s is not None else max_pool for s in sizes]}")

    rows, raw_records = [], []
    for mode in MODES:
        print(f"\n===== mode = {mode} =====")
        for n_train in sizes:
            n_reps = 1 if n_train is None else args.repeats
            aucs, pooleds, n_evals, n_subj_used, n_ev_used = [], [], [], [], []
            for rep in range(n_reps):
                score, n_su, n_ee = run_once(X, y, subj, src, uuids, folds,
                                             calib, n_train, rep, mode)
                ps, n_eval, pooled = evaluate(y, subj, score, eval_mask)
                if ps is None:
                    continue
                aucs.append(ps)
                pooleds.append(pooled)
                n_evals.append(n_eval)
                n_subj_used.append(n_su)
                n_ev_used.append(n_ee)
                raw_records.append({"mode": mode, "n_train": n_train or max_pool,
                                    "rep": rep, "per_subject_auc": ps,
                                    "pooled_auc": pooled, "n_eval_subjects": n_eval})
                print(f"  n_train={n_su:5.1f} 人 (rep {rep}): "
                      f"per-subject AUC={ps:.4f}  pooled={pooled:.4f}  "
                      f"可評估 {n_eval} 人  訓練事件 {n_ee:,.0f}")

            rows.append({
                "mode": mode,
                "n_train_subjects": float(np.mean(n_subj_used)) if n_subj_used else np.nan,
                "n_train_events": float(np.mean(n_ev_used)) if n_ev_used else np.nan,
                "auc_mean": float(np.mean(aucs)) if aucs else None,
                "auc_std": float(np.std(aucs)) if aucs else None,
                "pooled_mean": float(np.mean(pooleds)) if pooleds else None,
                "n_eval_subjects": float(np.mean(n_evals)) if n_evals else np.nan,
                "n_reps": len(aucs),
            })

    df = pd.DataFrame(rows)
    print("\n" + "=" * 96)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---------- 曲線圖（英文標籤，避免 CJK 缺字警告） ----------
    fig, ax = plt.subplots(figsize=(8, 5))
    for mode, color in zip(MODES, ["#9fb8d1", "#e0824a"]):
        sub = df[df["mode"] == mode].sort_values("n_train_subjects")
        if sub.empty:
            continue
        xx = sub["n_train_subjects"].to_numpy()
        yy = np.array([v if v is not None else np.nan for v in sub["auc_mean"]])
        ee = np.array([v if v is not None else 0.0 for v in sub["auc_std"]])
        ax.errorbar(xx, yy, yerr=ee, marker="o", capsize=3, label=mode, color=color)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance (0.5)")
    ax.set_xlabel("Number of training subjects")
    ax.set_ylabel("per-subject AUC")
    ax.set_title("Learning curve: does adding subjects still help?", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "learning_curve.png")
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\n寫入 {plot_path}")

    meta = {"n_events": int(len(y)), "n_pos": int(y.sum()), "n_subjects": int(len(uuids)),
            "n_calib": int(calib.sum()), "n_eval_events": int(eval_mask.sum())}
    json_path = os.path.join(out_dir, "results.json")
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump({"meta": meta, "summary": rows, "raw_runs": raw_records,
                   "args": {"sizes": args.sizes, "repeats": args.repeats}},
                  fp, ensure_ascii=False, indent=2)
    print(f"寫入 {json_path}")

    summary_path = os.path.join(out_dir, "summary.txt")
    write_summary(summary_path, rows, args, meta)
    print(f"寫入 {summary_path}")


if __name__ == "__main__":
    main()

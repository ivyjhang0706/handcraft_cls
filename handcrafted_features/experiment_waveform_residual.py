# -*- coding: utf-8 -*-
"""
測試「殘差波形」：用個人正常模板相減後的 150 點波形當輸入，對照現行手工特徵。

動機：手工特徵把 150 點平均心拍壓成 ~96 個摘要統計量（時距、振幅、比值），任何不被
這些統計量捕捉的形態變化，現行管線完全看不見。前置的視覺檢查算出配對訊噪比 1.217
（16人平均，95%CI [1.061, 1.408]，虛無期望值=1），代表 High 事件偏離個人正常模板的
幅度比隨機 Normal 事件多約 22%，殘差裡有非零的系統性成分，值得用模型測一次。

⚠️ 資料版本（踩過兩次坑，動手前務必看懂）：

  BGM 特徵有兩版，值完全不同（同一筆 (timestamp,index) 的 96 個特徵沒有一個相同，
  相對差異 0.6~2.2，保留的 burst 數差 2.7 倍）：
    Regression_Features_BGM_latest（09-16）        ← 現行主線用這版，raw AUC ~0.598
    Model_bgm/70_10_20/Regression_Features_qc_mode_backup（09-21）  ← raw AUC 只有 ~0.478
  波形只有 09-18 那一版（Dataset 與 70_10_20/GlucoseData 都是同一次跑的），
  它跟 qc_mode 版事件 100% 對齊、跟 BGM_latest 只有 56% 對齊。

  這裡選擇「BGM_latest 特徵 ∩ Dataset 波形」的交集（1,316 事件、17 人可評估），
  理由是寧可用好的特徵當 baseline、犧牲事件數，也不要用明顯較差的 qc_mode 版
  （用 qc_mode 版跑過一次，所有組別都掉到 0.40~0.48，baseline 本身就壞掉，
  比較沒有意義）。代價與已知限制：同一個事件的波形與特徵來自不同次 QC 設定，
  不是同一批 burst 算出來的；兩者描述同一個血糖事件、標籤相同，但不完全等價。

五個對照組，全部跑在完全相同的事件、fold、校正事件上：
  feat_c6      手工特徵 + 錨定（baseline，對應現行 centered_6 的做法）
  feat_raw     手工特徵，未錨定（控制組）
  wave_raw     原始波形，未相減（控制組：區分「波形資訊」與「模板相減」的貢獻）
  resid        波形 − 個人正常模板
  resid_norm   振幅正規化後再相減（事件間振幅 CV 達 0.28~1.28，殘差可能被電極
               接觸品質主導；正規化與否在訊噪比上分不出勝負，兩個都跑）
  feat_c6+resid  手工特徵 ⊕ 殘差（最實用：殘差能不能在現有特徵之上再加分）

錨定規則與現行管線一致（這點第一版寫錯過，會讓 baseline 失真）：
  訓練端的人用他全部 Normal 事件算模板（訓練階段本來就拿得到完整資料）
  測試端的人只用 6 筆 Normal 校正事件（部署現實）
  → 在每一折內各自計算，特徵與波形走完全相同的錨定機制

其他：模板用 Normal 事件（校正時血糖值本來就已知，不是 leakage）；校正事件排除在
評估之外（先前實測這個自我參照偏差約 0.022 AUC）。因為事件集合換了，數字不能直接
跟現有的 0.636 比，要看本腳本自己算的 feat_c6。

用法：
  conda run -n ml python kfold/handcrafted_features/experiment_waveform_residual.py
"""

import glob
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

from experiment_handcrafted_classification import (  # noqa: E402
    META_COLS, K_FOLDS, SEED, N_CALIB_EVENTS, MIN_PER_SIDE,
    center_by_subject, pick_threshold, OUT_DIR,
)
from binary_metrics import binary_metrics  # noqa: E402
from config import BGM_WAVE_ROOT as BGM_WAVE, CGM_WAVE_ROOT as CGM_WAVE  # noqa: E402
from config import BGM_ROOT as BGM_FEAT, CGM_ROOT as CGM_FEAT  # noqa: E402

POS_LEVEL, NEG_LEVELS = "High", ["Normal"]
WCOLS = [f"ecg_{i}" for i in range(150)]
RF_PARAMS = dict(n_estimators=300, max_depth=6, class_weight="balanced",
                 random_state=0, n_jobs=8)
N_BOOT = 10000


def _event_mean(df, value_cols):
    """同一事件底下多個 burst 先平均成事件層級（與分類管線慣例一致）。"""
    return df.groupby(["uuid", "timestamp"], as_index=False).agg(
        {**{c: "mean" for c in value_cols}, "BG_Level": "first"})


def load_waveforms():
    out = []
    for tag, pattern in [("BGM", f"{BGM_WAVE}/*/*/*.csv"),
                         ("CGM", f"{CGM_WAVE}/*/*_ecg_waveforms.csv")]:
        frames = []
        for f in sorted(glob.glob(pattern)):
            if "removed" in os.path.basename(f):
                continue
            df = pd.read_csv(f, usecols=["uuid", "timestamp", "BG_Level"] + WCOLS)
            df = df[df["BG_Level"].isin([POS_LEVEL] + NEG_LEVELS)]
            if df.empty:
                continue
            df["uuid"] = df["uuid"].astype(str)
            df["timestamp"] = df["timestamp"].astype(str)
            frames.append(_event_mean(df, WCOLS))
        d = pd.concat(frames, ignore_index=True)
        d["source"] = tag
        print(f"  波形 {tag}: {d['uuid'].nunique()} 人，{len(d):,} 事件")
        out.append(d)
    return pd.concat(out, ignore_index=True)


def load_features():
    out, feat_cols = [], None
    for tag, root in [("BGM", BGM_FEAT), ("CGM", CGM_FEAT)]:
        frames = []
        for f in sorted(glob.glob(os.path.join(root, "*", "*_regression_features.csv"))):
            if "removed" in os.path.basename(f):
                continue
            u = os.path.basename(os.path.dirname(f))
            df = pd.read_csv(f)
            cols = [c for c in df.columns if c not in META_COLS]
            if feat_cols is None:
                feat_cols = cols
            assert cols == feat_cols, f"特徵欄位不一致: {f}"
            num = df[cols].apply(pd.to_numeric, errors="coerce")
            num["uuid"] = str(u)
            num["timestamp"] = df["timestamp"].astype(str).values
            num["BG_Level"] = df["BG_Level"].values
            num = num[num["BG_Level"].isin([POS_LEVEL] + NEG_LEVELS)]
            if num.empty:
                continue
            frames.append(_event_mean(num, cols))
        d = pd.concat(frames, ignore_index=True)
        d["source"] = tag
        print(f"  特徵 {tag}: {d['uuid'].nunique()} 人，{len(d):,} 事件")
        out.append(d)
    return pd.concat(out, ignore_index=True), list(feat_cols)


def build_table():
    wav = load_waveforms()
    fea, feat_cols = load_features()
    df = fea.merge(wav[["uuid", "timestamp", "source"] + WCOLS],
                   on=["uuid", "timestamp", "source"], how="inner")
    # 同事件兩來源都有時留 BGM（比照 load_merged 的 BGM 優先）
    df["_pri"] = (df["source"] == "CGM").astype(int)
    df = df.sort_values(["uuid", "timestamp", "_pri"]).drop_duplicates(
        subset=["uuid", "timestamp"], keep="first").drop(columns="_pri").reset_index(drop=True)
    n_bgm = int((df["source"] == "BGM").sum())
    print(f"\n特徵∩波形後：{len(df):,} 事件（BGM {n_bgm:,} / CGM {len(df)-n_bgm:,}），"
          f"{df['uuid'].nunique()} 人")
    return df, feat_cols


def make_calib(subj, src, y, uuids):
    """每人 6 個 Normal 校正事件（BGM 優先）。"""
    rng = np.random.RandomState(SEED)
    calib = np.zeros(len(subj), bool)
    for u in uuids:
        pool = np.where(subj == u)[0]
        pool_bgm = pool[src[pool] == "BGM"]
        pool = pool_bgm if len(pool_bgm) > 0 else pool
        pool_n = pool[y[pool] == 0]
        if len(pool_n) > 0:
            pool = pool_n
        pick = rng.choice(pool, size=min(N_CALIB_EVENTS, len(pool)), replace=False)
        calib[pick] = True
    return calib


def _amp_norm(X):
    span = X.max(axis=1) - X.min(axis=1)
    span[span == 0] = 1.0
    return X / span[:, None]


def run_arm(parts, y, subj, src, uuids, folds, calib):
    """parts: [(矩陣, anchored: bool, normalize: bool), ...]，折內各自處理後 hstack。

    anchored=True 時的錨定規則與現行管線一致：訓練端用該人全部 Normal 事件算模板、
    測試端只用 6 筆校正事件。錨定必須在折內做，因為誰是訓練端/測試端每折都不同。
    """
    score = np.full(len(y), np.nan)
    thr = np.full(len(y), np.nan)
    is_normal = (y == 0)
    for f in range(K_FOLDS):
        te_u = [u for u in uuids if folds[u] == f]
        tr = ~np.isin(subj, te_u)
        te = np.isin(subj, te_u) & (src == "BGM")   # 只評估 BGM（部署現實）
        if te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        tr_blocks, te_blocks = [], []
        for X0, anchored, normalize in parts:
            X = _amp_norm(X0) if normalize else X0
            if not anchored:
                tr_blocks.append(X[tr])
                te_blocks.append(X[te])
                continue
            tr_blocks.append(center_by_subject(X[tr], subj[tr], ref_idx=is_normal[tr]))
            te_blocks.append(center_by_subject(X[te], subj[te], ref_idx=calib[te]))
        Xtr, Xte = np.hstack(tr_blocks), np.hstack(te_blocks)
        sc = StandardScaler().fit(Xtr)
        clf = RandomForestClassifier(**RF_PARAMS)
        clf.fit(sc.transform(Xtr), y[tr])
        thr[te] = pick_threshold(clf.predict_proba(sc.transform(Xtr))[:, 1], y[tr])
        score[te] = clf.predict_proba(sc.transform(Xte))[:, 1]
    return score, thr


def per_subject_metrics(y, subj, score, thr, eval_mask):
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
    out_dir = os.path.join(OUT_DIR, "waveform_residual")
    os.makedirs(out_dir, exist_ok=True)

    df, feat_cols = build_table()
    y = (df["BG_Level"] == POS_LEVEL).astype(int).to_numpy()
    subj = df["uuid"].to_numpy()
    src = df["source"].to_numpy()
    F = np.nan_to_num(df[feat_cols].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
    W = np.nan_to_num(df[WCOLS].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
    uuids = np.array(sorted(set(subj)))

    rng = np.random.RandomState(SEED)
    order = rng.permutation(len(uuids))
    folds = {u: i % K_FOLDS for i, u in enumerate(uuids[order])}
    calib = make_calib(subj, src, y, uuids)
    eval_mask = ~calib
    print(f"校正事件 {calib.sum()} 個（High {int(y[calib].sum())} 個）")
    print(f"評估排除校正事件，可評估 {eval_mask.sum():,}/{len(y):,} 事件\n")

    arms = {
        "feat_c6（特徵+錨定，baseline）": [(F, True, False)],
        "feat_raw（特徵，未錨定）": [(F, False, False)],
        "wave_raw（原始波形，未相減）": [(W, False, False)],
        "resid（波形−正常模板）": [(W, True, False)],
        "resid_norm（振幅正規化後相減）": [(W, True, True)],
        "feat_c6+resid（特徵⊕殘差）": [(F, True, False), (W, True, False)],
    }

    rows, metrics = [], {}
    for name, parts in arms.items():
        dim = sum(p[0].shape[1] for p in parts)
        print(f"跑 {name}  (dim={dim})")
        s, t = run_arm(parts, y, subj, src, uuids, folds, calib)
        m = per_subject_metrics(y, subj, s, t, eval_mask)
        metrics[name] = m
        auc = [m[u]["auc"] for u in m if m[u]["auc"] is not None]
        rec = [m[u]["recall"] for u in m if m[u]["recall"] is not None]
        spe = [m[u]["specificity"] for u in m if m[u]["specificity"] is not None]
        rows.append({"arm": name, "dim": dim, "人數": len(m),
                     "AUC": float(np.mean(auc)) if auc else None,
                     "recall": float(np.mean(rec)) if rec else None,
                     "specificity": float(np.mean(spe)) if spe else None})

    print("\n" + "=" * 88)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n" + "=" * 88)
    print("配對比較（同一批人，AUC 差值 + bootstrap 95% CI；基準＝feat_c6）")
    print("=" * 88)
    base = metrics["feat_c6（特徵+錨定，baseline）"]
    boots = {}
    for name in list(arms)[1:]:
        b = paired_bootstrap(base, metrics[name])
        boots[name] = b
        if b is None:
            continue
        verdict = "顯著" if b["significant"] else "與雜訊分不開"
        print(f"  {name:30} {b['mean_diff']:+.4f}  "
              f"95%CI=[{b['ci_low']:+.4f}, {b['ci_high']:+.4f}]  n={b['n']}  {verdict}")

    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as fp:
        json.dump({"summary": rows, "paired_bootstrap": boots, "per_subject": metrics},
                  fp, ensure_ascii=False, indent=2)
    print(f"\n寫入 {os.path.join(out_dir, 'results.json')}")


if __name__ == "__main__":
    main()

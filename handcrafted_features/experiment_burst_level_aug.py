# -*- coding: utf-8 -*-
"""
測試「同一次血糖事件底下的多個 burst 當成獨立訓練樣本」對績效的影響。

背景：裝置每 10 秒連續錄一個 burst，一次血糖量測會在 ±5/7/10 分鐘視窗內涵蓋
十幾到上百個 burst，每個 burst 各自算出一組手工特徵（CSV 裡的 index 欄）。
但 experiment_handcrafted_bgm.load_source() 會把同一事件的所有 index 取平均、
壓成 1 筆才送進訓練，等於 burst 之間的真實變異從來沒有進到模型裡。

本實驗比較兩種訓練輸入（評估方式完全相同）：
  A  每事件 1 筆（全部 burst 的特徵平均）—— 現況，等同 plot_rf_report.py
  C  A 那 1 筆 + 每事件最多 CAP 筆 burst 列（在時間軸上均勻取）

為什麼是「A 那筆 + burst 列」而不是「只有 burst 列」：
事件平均那筆是用「全部」burst 算的，帶有被 CAP 丟掉的那些 burst 的資訊，所以不是
保留下來那幾筆的線性組合。保留它，A→C 就只多做了一件事（加入 burst 多樣性），
結果變好變壞都能歸因；若直接用 burst 列取代平均列，就同時改了兩件事。

三個容易踩到的坑，這裡都處理了：
  1. load_merged() 的 drop_duplicates(subset=["uuid","timestamp"]) 在 burst 級別
     會把每個事件砍到只剩 1 筆。改成在事件層級判斷、整組 burst 一起去留。
  2. 取樣視窗長度是血糖值的函數（>=250 用 ±10 分、>=200 用 ±7 分、其餘 ±5 分），
     所以 High 事件的 burst 數平均是 Normal 的 1.7 倍。不設上限直接展開，會把
     row 層級的 High 佔比從 0.25 推到 0.29（BGM 單看是 0.29→0.41），等於偷改了
     類別先驗。設 CAP 之後這個偏差會自動收斂回事件層級的比例。
  3. 上限不能取前 N 筆（High 事件視窗是 Normal 的兩倍長，取前 N 筆會讓兩類的
     取樣位置系統性不同），要在時間軸上均勻取。

評估一律回到事件層級：同一事件的預測分數先平均回去，再算 per-subject 指標，
所以 MIN_PER_SIDE、recall/specificity/AUC 的意義跟 A 完全一致，數字可直接對比。
fold 切分與 6 次校正事件直接沿用 A 的結果，確保 A/C 是配對比較。

用法：
  conda run -n ml python kfold/handcrafted_features/experiment_burst_level_aug.py --cap 10
"""

import argparse
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

from experiment_handcrafted_bgm import load_merged  # noqa: E402
from experiment_handcrafted_classification import META_COLS, center_by_subject, pick_threshold  # noqa: E402
from config import CGM_ROOT, BGM_ROOT, K_FOLDS, SEED, N_CALIB_EVENTS, MIN_PER_SIDE, OUT_DIR  # noqa: E402
from binary_metrics import binary_metrics, METRIC_KEYS, METRIC_LABELS  # noqa: E402

POS_LEVEL, NEG_LEVELS = "High", ["Normal"]
RF_PARAMS = dict(n_estimators=300, max_depth=6, class_weight="balanced",
                 random_state=0, n_jobs=8)
N_BOOT = 10000


def load_burst_level():
    """讀 CGM+BGM 兩個來源，保持 burst(CSV 的 index 欄) 級別，不做事件內聚合。"""
    frames = {}
    feat_cols = None
    for root, tag in [(BGM_ROOT, "BGM"), (CGM_ROOT, "CGM")]:
        per_source = []
        for f in sorted(glob.glob(os.path.join(root, "*", "*_regression_features.csv"))):
            u = os.path.basename(os.path.dirname(f))
            df = pd.read_csv(f)
            cols = [c for c in df.columns if c not in META_COLS]
            if feat_cols is None:
                feat_cols = cols
            num = df[cols].apply(pd.to_numeric, errors="coerce")
            num["uuid"] = str(u)
            num["timestamp"] = df["timestamp"].astype(str).values
            num["burst"] = df["index"].values
            num["BG_Level"] = df["BG_Level"].values
            # (uuid,timestamp,burst) 實測是唯一鍵，這個 groupby 只是防同鍵重複
            num = num.groupby(["uuid", "timestamp", "burst"], as_index=False).agg(
                {**{c: "mean" for c in cols}, "BG_Level": "first"})
            per_source.append(num)
        data = pd.concat(per_source, ignore_index=True)
        data["source"] = tag
        frames[tag] = data
        print(f"  {tag}: {data['uuid'].nunique()} 人，{len(data):,} 筆 burst")
    return frames["BGM"], frames["CGM"], feat_cols


def build_tables(cap):
    """回傳 (ev, rows, feat_cols)。

    ev   事件層級表，直接用原本的 load_merged() 產生 —— 這就是 A 的輸入，
         一個位元都沒改，fold 切分與校正抽樣才能跟現況完全一致。
    rows C 的訓練輸入：ev 的每一列(is_avg=True) + 每事件最多 cap 筆 burst 列。
         兩者都帶 ev_pos，用來把預測聚合回事件。
    """
    ev_all, feat_cols, _ = load_merged()
    ev = ev_all[ev_all["BG_Level"].isin([POS_LEVEL] + NEG_LEVELS)].reset_index(drop=True)
    ev["timestamp"] = ev["timestamp"].astype(str)
    ev["ev_pos"] = np.arange(len(ev))
    print(f"事件層級(A 的輸入)：{len(ev):,} 個事件，{ev['uuid'].nunique()} 人")

    bgm, cgm, bfeat = load_burst_level()
    assert list(bfeat) == list(feat_cols), "burst 級別與事件級別的特徵欄位不一致"

    # 坑 1：來源去重要在「事件層級」做，整組 burst 一起去留（BGM 優先），
    # 不能用 drop_duplicates(["uuid","timestamp"])，那會把每事件砍到剩 1 筆。
    burst = pd.concat([bgm, cgm], ignore_index=True)
    burst = burst[burst["BG_Level"].isin([POS_LEVEL] + NEG_LEVELS)]
    # 只保留 ev 選中的那個來源的 burst：用 (uuid,timestamp,source) 三鍵 inner join，
    # 同時完成「去掉 A 沒選到的來源」與「去掉 A 沒有的事件」。
    burst = burst.merge(ev[["uuid", "timestamp", "source", "ev_pos"]],
                        on=["uuid", "timestamp", "source"], how="inner")

    # 坑 3：在時間軸上均勻取 cap 筆（burst 編號本身就是時間先後）
    burst = burst.sort_values(["ev_pos", "burst"]).reset_index(drop=True)
    keep = []
    for _, g in burst.groupby("ev_pos", sort=False):
        n = len(g)
        sel = np.unique(np.linspace(0, n - 1, min(cap, n)).round().astype(int))
        keep.append(g.index.to_numpy()[sel])
    burst = burst.loc[np.concatenate(keep)].reset_index(drop=True)

    keep_cols = list(feat_cols) + ["uuid", "timestamp", "BG_Level", "source", "ev_pos"]
    ev_rows = ev[keep_cols].copy()
    ev_rows["is_avg"] = True
    b_rows = burst[keep_cols].copy()
    b_rows["is_avg"] = False
    rows = pd.concat([ev_rows, b_rows], ignore_index=True)

    n_hit = int((burst.groupby("ev_pos").size() >= cap).sum())
    print(f"burst 級別(cap={cap})：{len(burst):,} 筆，{n_hit:,}/{len(ev):,} 個事件觸到上限")
    print(f"C 的訓練輸入合計：{len(rows):,} 筆（A 是 {len(ev):,} 筆）")
    return ev, rows, list(feat_cols)


def make_fold_and_calib(ev):
    """fold 切分與 6 次校正事件。

    這段刻意跟 plot_rf_report.build_predictions() 逐行相同（同樣的 SEED、同樣的
    抽樣順序、同樣作用在事件層級的表上），A 與 C 才會拿到完全一樣的 fold 與
    完全一樣的那 6 個校正事件，比較才是配對的。
    """
    subj = ev["uuid"].to_numpy()
    src = ev["source"].to_numpy()
    uuids = np.array(sorted(set(subj)))

    rng = np.random.RandomState(SEED)
    order = rng.permutation(len(uuids))
    folds = {u: i % K_FOLDS for i, u in enumerate(uuids[order])}

    rng2 = np.random.RandomState(SEED)
    calib_ev = np.zeros(len(ev), bool)
    for u in uuids:
        pool = np.where(subj == u)[0]
        pool_bgm = pool[src[pool] == "BGM"]
        pool = pool_bgm if len(pool_bgm) > 0 else pool
        pick = rng2.choice(pool, size=min(N_CALIB_EVENTS, len(pool)), replace=False)
        calib_ev[pick] = True
    return uuids, folds, calib_ev


def _event_mean(values, ev_pos, n_ev):
    """把 row 級別的數值按事件取平均，回傳長度 n_ev 的陣列（沒有資料的填 nan）。"""
    out = np.full(n_ev, np.nan)
    s = pd.Series(values).groupby(ev_pos).mean()
    out[s.index.to_numpy()] = s.to_numpy()
    return out


def run_variant(rows, feat_cols, ev, uuids, folds, calib_ev, use_burst):
    """跑 5 折，回傳事件層級的分數與門檻。

    use_burst=False 只用事件平均列（= A）；True 則平均列與 burst 列都進訓練（= C）。
    回傳的 score 有兩種聚合：
      avg  只看事件平均那一列的預測（跟 A 的評估方式一模一樣）
      pool 該事件全部列的預測取平均（等於 test-time ensembling）
    兩者用同一個模型，用來拆解「訓練資料變多」與「測試時聚合」各自的貢獻。
    """
    sel = rows["is_avg"].to_numpy() if not use_burst else np.ones(len(rows), bool)
    sub = rows[sel]
    X = np.nan_to_num(sub[feat_cols].to_numpy(dtype=np.float64),
                      nan=0.0, posinf=0.0, neginf=0.0)
    y = (sub["BG_Level"] == POS_LEVEL).astype(int).to_numpy()
    subj = sub["uuid"].to_numpy()
    src = sub["source"].to_numpy()
    ev_pos = sub["ev_pos"].to_numpy()
    is_avg = sub["is_avg"].to_numpy()
    n_ev = len(ev)

    # 校正參考只用「校正事件的平均列」，跟 A 用的參考完全相同。
    # 這樣 A→C 唯一的差別就是訓練列數，centering 這個變因被固定住。
    ref_all = is_avg & calib_ev[ev_pos]

    out = {}
    for mode in ["raw", "centered_6"]:
        s_avg = np.full(n_ev, np.nan)
        s_pool = np.full(n_ev, np.nan)
        thr_ev = np.full(n_ev, np.nan)
        for f in range(K_FOLDS):
            te_u = [u for u in uuids if folds[u] == f]
            tr = ~np.isin(subj, te_u)
            te = np.isin(subj, te_u) & (src == "BGM")   # 只評估 BGM（部署現實）
            if te.sum() == 0 or len(np.unique(y[tr])) < 2:
                continue

            if mode == "raw":
                Xtr, Xte = X[tr], X[te]
            else:
                Xc = X.copy()
                Xc[tr] = center_by_subject(X[tr], subj[tr])
                Xc[te] = center_by_subject(X[te], subj[te], ref_idx=ref_all[te])
                Xtr, Xte = Xc[tr], Xc[te]

            sc = StandardScaler().fit(Xtr)
            clf = RandomForestClassifier(**RF_PARAMS)
            clf.fit(sc.transform(Xtr), y[tr])

            # 坑 2 的延伸：門檻也要在事件層級選，否則 burst 多的事件會主導門檻搜尋
            s_tr = clf.predict_proba(sc.transform(Xtr))[:, 1]
            g = pd.DataFrame({"p": ev_pos[tr], "s": s_tr, "y": y[tr]}).groupby("p").agg(
                s=("s", "mean"), y=("y", "first"))
            thr = pick_threshold(g["s"].to_numpy(), g["y"].to_numpy())

            s_te = clf.predict_proba(sc.transform(Xte))[:, 1]
            pos_te, avg_te = ev_pos[te], is_avg[te]
            s_pool[np.unique(pos_te)] = _event_mean(s_te, pos_te, n_ev)[np.unique(pos_te)]
            if avg_te.any():
                m = avg_te
                s_avg[np.unique(pos_te[m])] = _event_mean(
                    s_te[m], pos_te[m], n_ev)[np.unique(pos_te[m])]
            thr_ev[np.unique(pos_te)] = thr
            print(f"    [{mode}] fold {f}: 訓練 {tr.sum():,} 列 / 評估 {te.sum():,} 列"
                  f"（{len(te_u)} 人，{len(np.unique(pos_te)):,} 個事件） threshold={thr:.3f}")
        out[mode] = {"avg": s_avg, "pool": s_pool, "thr": thr_ev}
    return out


def per_subject_metrics(ev, score, thr):
    """事件層級的 per-subject 指標，跟 plot_rf_report 的定義一致。"""
    y = (ev["BG_Level"] == POS_LEVEL).astype(int).to_numpy()
    subj = ev["uuid"].to_numpy()
    done = ~np.isnan(score)
    out = {}
    for u in np.unique(subj[done]):
        m = done & (subj == u)
        if min((y[m] == 0).sum(), (y[m] == 1).sum()) < MIN_PER_SIDE:
            continue
        out[u] = binary_metrics(y[m], score[m], threshold=float(thr[m][0]))
    return out


def paired_bootstrap(base, new, uuids, key="auc"):
    """對同一批人做配對 bootstrap。

    單獨看兩個平均值沒用（18 人的 CI 寬度約 0.13，任何小幅改善都會被雜訊蓋掉），
    但配對之後只要看「每個人的差值」，靈敏度高很多。
    """
    d = np.array([new[u][key] - base[u][key] for u in uuids
                  if base[u][key] is not None and new[u][key] is not None])
    if len(d) == 0:
        return None
    rng = np.random.RandomState(0)
    boot = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(N_BOOT)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mean_diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "n": int(len(d)), "significant": bool(lo > 0 or hi < 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=10, help="每個事件最多保留幾筆 burst")
    args = ap.parse_args()

    out_dir = os.path.join(OUT_DIR, "burst_level_aug", f"cap{args.cap}")
    os.makedirs(out_dir, exist_ok=True)

    ev, rows, feat_cols = build_tables(args.cap)
    uuids, folds, calib_ev = make_fold_and_calib(ev)

    print("\n[A] 每事件 1 筆（現況）")
    res_a = run_variant(rows, feat_cols, ev, uuids, folds, calib_ev, use_burst=False)
    print(f"\n[C] 每事件 1 筆平均 + 最多 {args.cap} 筆 burst")
    res_c = run_variant(rows, feat_cols, ev, uuids, folds, calib_ev, use_burst=True)

    # A 只有平均列，pool 與 avg 相同；C 兩種聚合都看，用來拆解效果來源
    variants = {
        "A_現況": (res_a, "avg"),
        "C_訓練加burst_評估只用平均列": (res_c, "avg"),
        "C_訓練加burst_評估用全部列平均": (res_c, "pool"),
    }

    metrics, summary = {}, []
    for name, (res, agg) in variants.items():
        metrics[name] = {}
        for mode in ["raw", "centered_6"]:
            m = per_subject_metrics(ev, res[mode][agg], res[mode]["thr"])
            metrics[name][mode] = m
            vals = [m[u]["auc"] for u in m if m[u]["auc"] is not None]
            summary.append({"variant": name, "mode": mode, "n_subjects": len(m),
                            "per_subject_auc": float(np.mean(vals)) if vals else None})

    print("\n" + "=" * 78)
    print(f"{'變體':40}{'模式':14}{'人數':>6}{'per-subject AUC':>18}")
    print("=" * 78)
    for r in summary:
        auc = f"{r['per_subject_auc']:.4f}" if r["per_subject_auc"] is not None else "N/A"
        print(f"{r['variant']:40}{r['mode']:14}{r['n_subjects']:>6}{auc:>18}")

    print("\n" + "=" * 78)
    print("配對比較（每個人的 AUC 差值 + bootstrap 95% CI）")
    print("=" * 78)
    boots = {}
    base_name = "A_現況"
    for name in list(variants)[1:]:
        for mode in ["raw", "centered_6"]:
            base, new = metrics[base_name][mode], metrics[name][mode]
            common = sorted(set(base) & set(new))
            b = paired_bootstrap(base, new, common)
            boots[f"{name}|{mode}"] = b
            if b is None:
                continue
            verdict = "顯著" if b["significant"] else "與雜訊分不開"
            print(f"  {name:44}{mode:12} "
                  f"{b['mean_diff']:+.4f}  95%CI=[{b['ci_low']:+.4f}, {b['ci_high']:+.4f}]  {verdict}")

    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as fp:
        json.dump({"cap": args.cap, "summary": summary, "paired_bootstrap": boots,
                   "per_subject": {k: {m: v[m] for m in v} for k, v in metrics.items()}},
                  fp, ensure_ascii=False, indent=2)
    print(f"\n寫入 {os.path.join(out_dir, 'results.json')}")


if __name__ == "__main__":
    main()

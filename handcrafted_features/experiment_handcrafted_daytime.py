# -*- coding: utf-8 -*-
"""
Priority 0a：驗證 exp.md 的 Low vs 其餘結果（AUC 0.6214,
centered）是不是跟深度特徵一樣，其實大部分是晝夜/睡眠 confound。

背景：exp.md 發現 Low 事件 48.9% 落在半夜/清晨(00-07)，是
Normal/High(12-13%)的4倍，且全部來自 CGM；深度特徵的 Low AUC 控制時段後從 0.573
掉回 0.509(亂猜)。手工形態學特徵目前還沒做這個驗證(exp.md
「尚未驗證的事項」第1點)，這支腳本補上。

做法：
  1. 用跟 experiment_handcrafted_classification.py 完全一樣的資料/模型/切分邏輯，
     只是把事件先篩成白天(08-19)才丟進去，兩者用一樣的 raw/centered_all/centered_6
     三種做法比較。
  2. 順便印出 Low/Normal/High 各自的晝夜事件分布，跟 exp.md
     的表核對是否一致(同一批資料，數字應該相近)。

用法：
  conda run -n ml python kfold/handcrafted_features/experiment_handcrafted_daytime.py
"""

import json
import os
import sys

import numpy as np

_KFOLD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in sorted(os.listdir(_KFOLD_ROOT)):
    _p = os.path.join(_KFOLD_ROOT, _d)
    if os.path.isdir(_p):
        sys.path.insert(0, _p)

from experiment_handcrafted_classification import load_event_level, run_task, OUT_DIR  # noqa: E402

DAY_START, DAY_END = 8, 19  # 含頭尾，跟 evaluate_normal_vs_high_k5.py 的 daytime_all 定義一致


def hour_of_timestamp(ts):
    """timestamp 格式是 YYYYMMDDHHMMSS(int64)，例如 20250226100400 → 10 點。"""
    return (np.asarray(ts, dtype=np.int64) // 10000) % 100


def print_day_night_table(data):
    hours = hour_of_timestamp(data["timestamp"].to_numpy())
    daytime = (hours >= DAY_START) & (hours <= DAY_END)
    print(f"\n{'類別':8}{'事件數':>10}{'夜間/清晨(00-07)':>20}{'白天(08-19)':>16}")
    for level in ["Low", "Normal", "High"]:
        m = (data["BG_Level"] == level).to_numpy()
        n = m.sum()
        if n == 0:
            continue
        n_night = int((~daytime[m]).sum())
        n_day = int(daytime[m].sum())
        print(f"{level:8}{n:>10,}{n_night:>13,} ({n_night/n:>5.1%}){n_day:>10,} ({n_day/n:>5.1%})")
    return hours, daytime


def main():
    print("讀取事件層級特徵...")
    data, feat_cols = load_event_level()
    print(f"\n總計 {len(data):,} 事件，{data['uuid'].nunique()} 人，{len(feat_cols)} 個特徵")

    print("\n晝夜事件分布（核對是否跟 exp.md 一致）：")
    hours, daytime = print_day_night_table(data)

    data_day = data[daytime].reset_index(drop=True)
    print(f"\n篩選白天(08-19)後：{len(data_day):,} 事件（原本 {len(data):,}），"
          f"{data_day['uuid'].nunique()} 人（原本 {data['uuid'].nunique()}）")

    res = {}
    res["Low_vs_rest_daytime_only"] = run_task(
        data_day, feat_cols, "Low vs 其餘（手工特徵，僅白天 08-19，跨人切分）",
        "Low", ["Normal", "High"])

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "daytime_only_results.json")
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(res, fp, ensure_ascii=False, indent=2)
    print(f"\n寫入 {out_path}")

    full_path = os.path.join(OUT_DIR, "results.json")
    if os.path.isfile(full_path):
        with open(full_path, encoding="utf-8") as fp:
            full = json.load(fp)
        print(f"\n{'='*76}\n對照：全時段 vs 白天限定（Low vs 其餘）\n{'='*76}")
        print(f"{'做法':16}{'全時段 per-subject AUC':>24}{'白天限定 per-subject AUC':>26}")
        for mode in ["raw", "centered_all", "centered_6"]:
            full_auc = full.get("Low_vs_rest", {}).get(mode, {}).get("per_subject_auc")
            day_auc = res["Low_vs_rest_daytime_only"][mode]["per_subject_auc"]
            full_s = f"{full_auc:.4f}" if full_auc is not None else "N/A"
            day_s = f"{day_auc:.4f}" if day_auc is not None and not np.isnan(day_auc) else "N/A"
            print(f"{mode:16}{full_s:>24}{day_s:>26}")
        print("\n判讀：如果白天限定後 AUC 掉回 ~0.5，代表 Low 的訊號跟深度特徵一樣主要是"
              "\n      晝夜/睡眠 confound，先前 0.6214 的結論需要收回。若仍明顯 >0.5，"
              "\n      才能說手工特徵的 Low 訊號站得住腳。")
    else:
        print(f"\n找不到 {full_path}，無法對照全時段結果，僅輸出白天限定結果。")


if __name__ == "__main__":
    main()

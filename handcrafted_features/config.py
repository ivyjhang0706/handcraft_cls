# -*- coding: utf-8 -*-
"""
集中管理資料路徑與輸出路徑。


CGM_ROOT / BGM_ROOT 底下的目錄結構要是 {uuid}/{uuid}_regression_features.csv。
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# CGM 事件特徵（62人）。experiment_handcrafted_classification.py 的 CGM-only 分析、
# experiment_handcrafted_bgm.py 的 CGM+BGM 合併訓練都會用到。
CGM_ROOT = os.path.join(PROJECT_ROOT, "Model_CGM", "70_10_20", "Regression_Features")

# BGM 指尖採血事件特徵。09-22 起改用 server_program_2.1.3 全量62人重跑的版本
# （Regression_Features，61人）：修掉 SRJ 原地覆寫污染 + 用 zip 還原乾淨來源 +
# FEATURE_MEAN_MODE 維持預設 qc，取代先前的 09-16 版(Regression_Features_BGM_latest，
# 54人，來源快照不明)與 09-21 的 qc_mode_backup(mean_mode 被關掉 QC，raw AUC 明顯較差
# ~0.478，見 experiment_waveform_residual.py 開頭說明，兩者都不要再用)。
BGM_ROOT = os.path.join(PROJECT_ROOT, "Model_BGM", "70_10_20", "Regression_Features")

# experiment_handcrafted_classification.py 歷史上只認得 CGM 資料，這裡讓 ROOT 沿用 CGM_ROOT。
ROOT = CGM_ROOT

# 原始 ECG 波形（150點/burst），只有 experiment_waveform_residual.py 用得到。
# 注意 BGM 跟 CGM 的目錄結構不同，load_waveforms() 對兩者用不同的 glob pattern：
#   BGM 讀 Dataset/{uuid}/{High,Low,Normal}/{uuid}_{category}.csv （三層，含分類資料夾）
#   CGM 讀 70_10_20/GlucoseData/{uuid}/{uuid}_ecg_waveforms.csv    （兩層，扁平）
# CGM_WAVE_ROOT 不能指到 Model_CGM/Dataset —— 那邊是跟 BGM 一樣的三層分類結構，
# 沒有 *_ecg_waveforms.csv，glob 會抓到 0 個檔案。
BGM_WAVE_ROOT = os.path.join(PROJECT_ROOT, "Model_BGM", "Dataset")
CGM_WAVE_ROOT = os.path.join(PROJECT_ROOT, "Model_CGM", "70_10_20", "GlucoseData")

# 所有實驗腳本的輸出根目錄（results.json / *.xlsx / *.png）。
OUT_DIR = os.path.join(PROJECT_ROOT, "output", "handcrafted_classification")

# subject-wise K-fold 與校正事件抽樣的共用超參數。
K_FOLDS = 5
SEED = 42
N_CALIB_EVENTS = 6
MIN_PER_SIDE = 3  # 每個人測試集裡每一類至少要幾個事件才算可評估

# handcraft_cls 專案慣例

跨專案的通用原則見 `~/.claude/CLAUDE.md`（指標集中管理／變數走 config／長跑指令用
tmux）。這份文件只記這個專案裡「原則要對應到哪個具體檔案」。

## 指標定義

分類指標一律從 [handcrafted_features/binary_metrics.py](handcrafted_features/binary_metrics.py)
import：`binary_metrics()`、`METRIC_KEYS`、`METRIC_LABELS`。寫新實驗腳本時看到有腳本
import 別的模組（例如已刪除的 `evaluate_normal_vs_high_k5`）取得同樣的東西，那是舊路徑
殘留，要改成從 `binary_metrics` import。

## Config

路徑、fold 數、seed、閾值等共用參數都在
[handcrafted_features/config.py](handcrafted_features/config.py)：
`CGM_ROOT` / `BGM_ROOT`（Regression_Features 根目錄）、`BGM_WAVE_ROOT` /
`CGM_WAVE_ROOT`（原始波形，指到 server_program_2.1.3）、`OUT_DIR`、
`K_FOLDS` / `SEED` / `N_CALIB_EVENTS` / `MIN_PER_SIDE`。新腳本要用到這些值一律
`from config import ...`，不要重新寫死路徑或常數。

## 實驗輸出慣例

每支 `experiment_*.py` 跑完，輸出都在 `output/handcrafted_classification/<實驗名>/`
底下（`config.py` 的 `OUT_DIR`），包含腳本自己寫的 `results.json`。

照全域規則第4條，同類型的實驗要彙總進同一個 Excel、每次跑開一個新分頁，檔案放在
`output/handcrafted_classification/` 底下（不是個別實驗的子資料夾）：

| 類型（測試的維度）           | Excel 檔案                    | 目前有的分頁                        |
|-------------------------------|--------------------------------|--------------------------------------|
| 校正事件的組成/數量           | `summary_calibration.xlsx`    | `calib_composition`、`calib_count`   |
| 訓練資料擴增方式              | `summary_augmentation.xlsx`   | `burst_level_aug_cap10`              |
| 訓練受試者人數（learning curve）| `summary_sample_size.xlsx`   | `learning_curve`                     |

新實驗屬於既有類型，就加一個分頁到對應檔案；屬於全新維度（例如之後的
`experiment_handcrafted_daytime.py`、`experiment_waveform_residual.py`），就開一個新的
Excel 檔案，並在上面這張表補一列，讓下一個人知道要併到哪裡。

## 執行環境

- 所有 `handcrafted_features/*.py` 腳本都要在 `ml` 這個 conda 環境跑：
  `conda run -n ml python <script>.py ...`，且要在 `handcrafted_features/` 目錄下執行
  （部分腳本的 sys.path 設定依賴這個相對關係）。
- 訓練/多 fold 實驗跑超過幾分鐘，照全域規則開 tmux session 執行。


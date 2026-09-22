# handcrafted_features/

手工形態學特徵分類——**目前唯一證實有訊號的主線**（見 repo 根目錄 `exp.md` 的「階段二：手工形態學特徵」），推翻了 `../signal_diagnostics/` 早期「ECG 訊號沒用」的部分結論。核心發現：Normal vs High per-subject AUC raw=0.666、subject-wise centering 後 0.715（對照 ECGFounder 深度特徵只有 0.498，亂猜等級）。

| 檔案 | 做什麼 |
|---|---|
| `experiment_handcrafted_classification.py` | 主線核心：測試手工形態學特徵能不能做跨人分類，自己做 subject-wise K-fold（CSV 裡現成的 split 欄位是個人內部切分，不能拿來評估通用模型）。共用工具 `per_subject_auc()`、`center_by_subject()`、`pick_threshold()`（用訓練組資料選 balanced_accuracy 最佳二元化門檻，取代寫死的 0.5）都定義在這裡，被本資料夾其他腳本 import。 |
| `experiment_handcrafted_bgm.py` | 把上面的方法套到真正要部署用的資料（BGM 指尖採血，54人）上，並提供 `load_merged()`：合併 CGM(62人)+BGM(54人) 訓練、只用 BGM 評估（部署現實）。 |
| `experiment_handcrafted_daytime.py` | 驗證 Low vs 其餘的結果是不是晝夜/睡眠 confound 造成的（優先層級：P0a）。 |
| `experiment_handcrafted_meal.py` | 測試飯前/飯後資訊（只有 BGM 有）能不能再拉高跨人分類表現。 |
| `experiment_handcrafted_merged_followup.py` | 針對「CGM+BGM合併訓練/BGM評估」Normal vs High 這個最有希望的方向，做兩項守門測試。 |
| `experiment_handcrafted_model_compare.py` | 比較 LogisticRegression(3種C)/RandomForest/XGBoost/SVC(3種C) 幾種模型設定，5個fold切分seed檢查穩健度。**結論：所有模型都落在 AUC 0.53~0.65，模型選擇不是瓶頸**，random_forest 目前最穩定（raw std 最小）。 |
| `plot_handcrafted_merged_report.py` | LogisticRegression 版報表：把 raw vs centered_6 結果畫成跟舊版深度特徵報告同樣風格的圖，方便對照。 |
| `plot_rf_report.py` | Random Forest 版報表整合腳本（取代早期分散的多支繪圖程式），額外依已確診糖尿病(DM)名單拆 DM/Normal 兩組分別出圖，輸出全部收在 `output/handcrafted_classification/rf/`。 |

## 呼叫關係
`experiment_handcrafted_bgm.py` 跟兩支 `plot_*.py` 都 import `experiment_handcrafted_classification.py` 的共用工具；兩支 `plot_*.py` 另外 import `../deep_model_kfold/evaluate_normal_vs_high_k5.py` 的 `binary_metrics()`/`METRIC_KEYS`（通用二元指標，不是深度模型專屬，只是歷史上先在那支腳本裡定義）。

## 現在該跑哪支
`plot_rf_report.py` 是目前最新、最完整的報表腳本（一支跑完出全部圖+Excel+DM/Normal分組）。`experiment_handcrafted_model_compare.py` 用於模型選型穩健度檢查，跑一次要將近 2 小時（SVC RBF kernel 特別慢），不需要頻繁重跑。

# -*- coding: utf-8 -*-
"""
從 bgm_pretriain_cls/kfold/deep_model_kfold/evaluate_normal_vs_high_k5.py 抽出來的
二元分類指標函式，只留 experiment_feature_count_compare.py 需要的部分（原檔案其他
部分依賴 torch/深度模型的 model.py、splits.py，跟手工特徵實驗無關，不搬過來）。
"""
import numpy as np
from sklearn.metrics import roc_auc_score

METRIC_KEYS = ["auc", "accuracy", "precision", "recall", "specificity", "f1"]
METRIC_LABELS = {"auc": "AUC", "accuracy": "Accuracy", "precision": "Precision",
                  "recall": "Recall(Sens)", "specificity": "Specificity", "f1": "F1"}


def binary_metrics(y_true, score, threshold=0.0):
    y_true = np.asarray(y_true)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {k: None for k in METRIC_KEYS}
    pred = (np.asarray(score) > threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"auc": float(roc_auc_score(y_true, score)), "accuracy": (tp + tn) / len(y_true),
            "precision": precision, "recall": recall, "specificity": specificity, "f1": f1}

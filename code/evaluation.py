"""지표, 예측 목록, 혼동행렬과 클래스별 결과 저장."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score)
from data import CLASS_NAMES


def predict(model, data_loader, device):
    model.eval()
    truth, preds = [], []
    with torch.no_grad():
        for xb, yb in data_loader:
            pred = model(xb.to(device, non_blocking=True)).argmax(1)
            truth.extend(yb.tolist())
            preds.extend(pred.cpu().tolist())
    return np.array(truth), np.array(preds)


def metrics(truth, preds):
    labels = list(range(len(CLASS_NAMES)))
    return {"Accuracy": accuracy_score(truth, preds),
            "Precision": precision_score(truth, preds, labels=labels, average="macro", zero_division=0),
            "Recall": recall_score(truth, preds, labels=labels, average="macro", zero_division=0),
            "F1-score": f1_score(truth, preds, labels=labels, average="macro", zero_division=0)}


def analyze_confusion_matrix(cm):
    # 클래스별 표본 수가 달라도 비교 가능한 Recall로 가장 좋은/나쁜 클래스를 정한다.
    support = cm.sum(1)
    recall = np.divide(cm.diagonal(), support, out=np.zeros(5, dtype=float), where=support != 0)
    wrong = cm.copy()
    np.fill_diagonal(wrong, 0)
    count = int(wrong.max())
    pairs = np.argwhere(wrong == count) if count else []
    return {"Best Class": ", ".join(np.array(CLASS_NAMES)[recall == recall.max()]),
            "Best Class Recall": float(recall.max()),
            "Worst Class": ", ".join(np.array(CLASS_NAMES)[recall == recall.min()]),
            "Worst Class Recall": float(recall.min()),
            "Most Confused Pair": ", ".join(f"{CLASS_NAMES[i]} -> {CLASS_NAMES[j]}" for i, j in pairs) or "없음",
            "Most Confused Count": count}


def evaluate_model(model, test_loader, device, model_name, result_dir=None, sample_rows=None):
    # CV에서는 저장 경로를 주지 않아 폴드별 이미지/세부 CSV를 만들지 않는다.
    truth, preds = predict(model, test_loader, device)
    result = {"Model": model_name, **metrics(truth, preds)}
    cm = confusion_matrix(truth, preds, labels=range(5))
    if result_dir is not None:
        result_dir = Path(result_dir)
        result_dir.mkdir(parents=True, exist_ok=True)
        # 제출용으로 그림과 해석 가능한 숫자 행렬만 함께 저장한다.
        pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(result_dir / f"{model_name}_confusion_matrix.csv")
        disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
        disp.plot(cmap="Blues", values_format="d")
        plt.title(f"{model_name} Confusion Matrix")
        plt.tight_layout()
        plt.savefig(result_dir / f"{model_name}_confusion_matrix.png", dpi=150)
        plt.close()
    print(classification_report(truth, preds, labels=range(5), target_names=CLASS_NAMES,
                                digits=3, zero_division=0), flush=True)
    prediction_df = pd.DataFrame(sample_rows) if sample_rows is not None else pd.DataFrame()
    prediction_df["True Index"] = truth
    prediction_df["Predicted Index"] = preds
    prediction_df["Predicted Label"] = [CLASS_NAMES[p] for p in preds]
    if sample_rows is not None:
        votes = prediction_df.groupby("File", sort=False)["Predicted Index"].agg(
            lambda values: int(np.bincount(values, minlength=5).argmax()))
        trial_truth = prediction_df.groupby("File", sort=False)["True Index"].first()
        result["Trial Accuracy"] = accuracy_score(trial_truth, votes)
    return result, cm, analyze_confusion_matrix(cm)

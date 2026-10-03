"""강의 기반 3모델 비교. python -u code/main.py (학습은 사용자가 실행)."""
import argparse
from datetime import datetime
import os
from pathlib import Path

# CUDA 결정성과 그래프 캐시 위치는 라이브러리 import 전에 지정한다.
ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from data import find_csv_files, label_of, split_files, build_dataset, create_loader
from models import create_model
from train import set_seed, train_model, measure_inference_ms
from evaluation import evaluate_model
from settings import (SEED, TEST_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE, DROPOUT,
                      CPU_THREADS, CV_FOLDS, INFERENCE_WARMUP, INFERENCE_REPEATS, MODEL_NAMES)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "results/submission")
    parser.add_argument("--raw-data", action="store_true", help="미필터링 원신호에만 필터 적용")
    parser.add_argument("--cv", action="store_true", help="DenseNet161 5-fold를 추가 실행")
    parser.add_argument("--cv-models", nargs="+", choices=MODEL_NAMES, default=["DenseNet161"])
    return parser.parse_args()


def print_comparison(frame):
    """학습 도중 완료된 모델과 전체 최종 결과를 같은 형식으로 표시한다."""
    view = frame[["Model", "Accuracy", "Precision", "Recall", "F1-score"]].copy()
    for column in view.columns[1:]:
        view[column] = view[column].map(lambda value: f"{value:.2%}")
    print("\n=== 모델 성능 비교 (마지막 epoch) ===", flush=True)
    print(view.to_string(index=False), flush=True)


def run(args):
    # 결과가 있으면 새 폴더를 사용한다. 기존 결과를 새 코드의 측정값으로 덮지 않는다.
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        output = output.with_name(output.name + datetime.now().strftime("_%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(CPU_THREADS)
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | 결과: {output}", flush=True)
    files = find_csv_files(args.data.resolve())
    # 같은 시행의 겹치는 윈도우가 학습/테스트에 섞이지 않도록 파일부터 분할한다.
    train_files, test_files = split_files(files, TEST_SIZE, SEED)
    Xtr, ytr, groups, _ = build_dataset(train_files, already_filtered=not args.raw_data)
    Xte, yte, _, test_rows = build_dataset(test_files, already_filtered=not args.raw_data)
    pin = device.type == "cuda"
    test_loader = create_loader(Xte, yte, BATCH_SIZE, pin_memory=pin)
    print(f"Train: {len(train_files)}시행/{len(ytr)}윈도우 | Test: {len(test_files)}시행/{len(yte)}윈도우", flush=True)
    results, analyses = [], []

    for name in MODEL_NAMES:
        print(f"\n=== {name} ===", flush=True)
        # 모델마다 초기화와 배치 순서의 seed를 재설정해 공통 조건을 지킨다.
        set_seed(SEED)
        loader = create_loader(Xtr, ytr, BATCH_SIZE, True, SEED, pin)
        model = create_model(name, dropout=DROPOUT).to(device)
        model, _, seconds = train_model(model, loader, device, EPOCHS, LEARNING_RATE,
                                        test_loader=test_loader)
        result, cm, analysis = evaluate_model(model, test_loader, device, name, output, test_rows)
        result.update({"Parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                       "Training Seconds": seconds,
                       "Inference ms/sample": measure_inference_ms(model, torch.from_numpy(Xte[:1]), device,
                           INFERENCE_WARMUP, INFERENCE_REPEATS)})
        results.append(result)
        analyses.append({"Model": name, **analysis})
        pd.DataFrame(results).to_csv(output / "model_comparison.csv", index=False)
        pd.DataFrame(analyses).to_csv(output / "confusion_analysis.csv", index=False)
        print("Confusion Matrix (행=정답, 열=예측):\n", cm, flush=True)
        print_comparison(pd.DataFrame(results))
        del model, loader
        if pin:
            torch.cuda.empty_cache()
            
    # 3주차 추가 실습. 독립 테스트는 CV에 넣지 않으며 폴드별 상세 파일은 저장하지 않는다.
    if args.cv:
        fold_results = []
        splitter = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED)
        for name in args.cv_models:
            for fold, (tr_idx, va_idx) in enumerate(splitter.split(train_files, [label_of(f) for f in train_files]), 1):
                print(f"\n{name} CV {fold}/{CV_FOLDS}", flush=True)
                tr_mask = np.isin(groups, [str(train_files[i]) for i in tr_idx])
                va_mask = np.isin(groups, [str(train_files[i]) for i in va_idx])
                set_seed(SEED)
                loader = create_loader(Xtr[tr_mask], ytr[tr_mask], BATCH_SIZE, True, SEED, pin)
                validation = create_loader(Xtr[va_mask], ytr[va_mask], BATCH_SIZE, pin_memory=pin)
                model = create_model(name, dropout=DROPOUT).to(device)
                model, _, _ = train_model(model, loader, device, EPOCHS, LEARNING_RATE, val_loader=validation)
                result, _, _ = evaluate_model(model, validation, device, name)
                fold_results.append({"Model": name, "Fold": fold, **{k:v for k,v in result.items() if k != "Model"}})
                pd.DataFrame(fold_results).to_csv(output / "cross_validation.csv", index=False)
                print(f"Fold {fold}: Accuracy={result['Accuracy']:.2%}, F1={result['F1-score']:.2%}", flush=True)
                del model, loader, validation
                if pin:
                    torch.cuda.empty_cache()
    print_comparison(pd.DataFrame(results))
    print(f"\n제출에 필요한 결과 저장 완료: {output}", flush=True)


if __name__ == "__main__":
    run(parse_args())

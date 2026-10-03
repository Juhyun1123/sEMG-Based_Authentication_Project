"""공통 학습 절차. 테스트셋으로 epoch를 고르지 않는다."""
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from settings import ADAM_BETAS, ADAM_EPS, WEIGHT_DECAY, SEED, EPOCHS, LEARNING_RATE, INFERENCE_WARMUP, INFERENCE_REPEATS


def set_seed(seed=SEED):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate_loss_accuracy(model, data_loader, device):
    # 평가 모드에서 전체 배치의 표본 수로 가중 평균한다.
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for xb, yb in data_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            out = model(xb)
            total_loss += criterion(out, yb).item() * len(yb)
            correct += (out.argmax(1) == yb).sum().item()
            total += len(yb)
    return total_loss / total, correct / total


def train_model(model, train_loader, device, epochs=EPOCHS, learning_rate=LEARNING_RATE,
                val_loader=None, test_loader=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=ADAM_BETAS,
                                 eps=ADAM_EPS, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    history = []
    synchronize(device)
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        # 테스트 후 다음 epoch에서는 Dropout/BatchNorm을 학습 모드로 되돌린다.
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        epoch_started = time.perf_counter()
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(yb)
            correct += (out.argmax(1) == yb).sum().item()
            total += len(yb)
        row = {"Epoch": epoch, "Train Loss": loss_sum / total, "Train Accuracy": correct / total}
        message = (f"Epoch {epoch:2d}/{epochs} | Train Loss: {row['Train Loss']:.4f} | "
                   f"Train Accuracy: {row['Train Accuracy']:.2%}")
        # 중간 테스트는 관찰용이다. eval/no_grad로 가중치와 BN 통계를 바꾸지 않는다.
        # 최고 테스트 epoch를 고르지 않으며 매 epoch의 수치를 파일로 내보내지 않는다.
        if test_loader is not None:
            row["Test Loss"], row["Test Accuracy"] = evaluate_loss_accuracy(model, test_loader, device)
            message += (f" | Test Loss: {row['Test Loss']:.4f} | "
                        f"Test Accuracy: {row['Test Accuracy']:.2%}")
        if val_loader is not None:
            row["Validation Loss"], row["Validation Accuracy"] = evaluate_loss_accuracy(model, val_loader, device)
            message += (f" | Validation Loss: {row['Validation Loss']:.4f} | "
                        f"Validation Accuracy: {row['Validation Accuracy']:.2%}")
        synchronize(device)
        row["Epoch Seconds"] = time.perf_counter() - epoch_started
        history.append(row)
        print(message + f" | {row['Epoch Seconds']:.1f}s", flush=True)
    synchronize(device)
    return model, history, time.perf_counter() - started


def measure_inference_ms(model, sample, device, warmup=INFERENCE_WARMUP, repeats=INFERENCE_REPEATS):
    model.eval()
    sample = sample.to(device)
    with torch.no_grad():
        for _ in range(warmup):
            model(sample)
        synchronize(device)
        started = time.perf_counter()
        for _ in range(repeats):
            model(sample)
        synchronize(device)
    return (time.perf_counter() - started) * 1000 / repeats

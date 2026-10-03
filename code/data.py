"""시행 단위 분할 및 공개된 필터링 데이터의 CWT 변환."""
from pathlib import Path

import numpy as np
import pywt
import torch
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from settings import SEED, TEST_SIZE, BATCH_SIZE

FS, WIN, HOP = 1000, 300, 150
SCALES = np.arange(1, 33)
CLASS_NAMES = ["A", "B", "C", "D", "E"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def find_csv_files(data_path):
    files = sorted(Path(data_path).glob("**/*.csv"))
    if not files:
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {data_path}")
    return files


def label_of(file_path):
    subject = Path(file_path).parent.name
    if subject not in CLASS_TO_IDX:
        raise ValueError(f"알 수 없는 클래스 폴더: {subject}")
    return CLASS_TO_IDX[subject]


def load_csv(file_path):
    # 원 CSV 헤더를 건너뛰고 3초/2채널 구조와 결측·무한값을 검증한다.
    x = np.loadtxt(file_path, delimiter=",", skiprows=1, ndmin=2)
    if x.shape[0] < x.shape[1]:
        x = x.T
    if x.shape != (3000, 2) or not np.isfinite(x).all():
        raise ValueError(f"3초/2채널의 유한한 신호가 필요합니다: {file_path}, {x.shape}")
    return x


def preprocess(x, fs=FS):
    """미필터링 원신호에만 사용. 공개 저장소 CSV에는 다시 적용하지 않는다."""
    # 1,000 Hz의 Nyquist가 500 Hz이므로 원신호 옵션의 상한은 499 Hz로 둔다.
    bn, an = iirnotch(60, 30, fs)
    x = filtfilt(bn, an, x, axis=0)
    b, a = butter(4, [20, 499], btype="band", fs=fs)
    return filtfilt(b, a, x, axis=0)


def make_windows(x, win=WIN, hop=HOP):
    if len(x) < win:
        raise ValueError("신호가 윈도우보다 짧습니다.")
    return np.stack([x[start:start + win] for start in range(0, len(x) - win + 1, hop)])


def minmax(windows, eps=1e-8):
    # 강의 예시와 동일: 각 윈도우의 두 채널을 함께 정규화한다.
    mn = windows.min(axis=(1, 2), keepdims=True)
    mx = windows.max(axis=(1, 2), keepdims=True)
    return (windows - mn) / (mx - mn + eps)


def to_cwt(one_window, wavelet="morl"):
    # 2주차 p.14: 두 신호의 CWT 절댓값과 평균 맵을 쌓는다.
    maps = [np.abs(pywt.cwt(one_window[:, ch], SCALES, wavelet)[0]) for ch in range(2)]
    maps.append((maps[0] + maps[1]) / 2)
    return np.stack(maps).astype(np.float32)


def split_files(files, test_size=TEST_SIZE, random_state=SEED):
    """먼저 파일을 Train 80% / Test 20%로 분할. 별도 검증셋을 떼지 않는다."""
    labels = [label_of(f) for f in files]
    train_files, test_files = train_test_split(
        files, test_size=test_size, stratify=labels, random_state=random_state
    )
    assert not set(train_files).intersection(test_files)
    return train_files, test_files


def build_dataset(file_list, already_filtered=True):
    X, y, groups, rows = [], [], [], []
    for i, file_path in enumerate(file_list, 1):
        signal = load_csv(file_path)
        if not already_filtered:
            signal = preprocess(signal)
        for w_idx, window in enumerate(minmax(make_windows(signal))):
            X.append(to_cwt(window))
            y.append(label_of(file_path))
            groups.append(str(file_path))
            rows.append({"File": f"{file_path.parent.parent.name}/{file_path.parent.name}/{file_path.name}",
                         "Window": w_idx, "Start Sample": w_idx * HOP,
                         "True Label": file_path.parent.name})
        if i % 25 == 0 or i == len(file_list):
            print(f"  CWT 전처리: {i}/{len(file_list)} 시행", flush=True)
    return np.stack(X), np.array(y, dtype=np.int64), np.array(groups), rows


def create_loader(X, y, batch_size=BATCH_SIZE, shuffle=False, seed=SEED, pin_memory=False):
    # 각 모델/폴드마다 새 generator를 생성하여 같은 seed의 배치 순서를 보장한다.
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, generator=generator, pin_memory=pin_memory)


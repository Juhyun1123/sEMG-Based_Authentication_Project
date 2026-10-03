"""모든 모델이 공유하는 설정. 실험 값을 바꿀 때는 이 파일에서 변경한다."""
SEED = 37
TEST_SIZE = 0.2
BATCH_SIZE = 16
EPOCHS = 45
LEARNING_RATE = 0.001
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
WEIGHT_DECAY = 0.0
DROPOUT = 0.5  
CPU_THREADS = 4
CV_FOLDS = 5
INFERENCE_WARMUP = 10
INFERENCE_REPEATS = 100
MODEL_NAMES = ["2D_CNN", "ResNet18", "DenseNet161"]


def parameter_rows():
    """설정 확인용 표 데이터를 반환한다. 문서 파일을 자동 작성하지 않는다."""
    return [
        ["분할", f"시행 단위 계층화 Train {1-TEST_SIZE:.0%} / Test {TEST_SIZE:.0%}"],
        ["Seed", SEED], ["Epochs", EPOCHS], ["Batch size", BATCH_SIZE],
        ["Optimizer", "Adam"], ["Learning rate", LEARNING_RATE],
        ["Adam betas", ADAM_BETAS], ["Adam eps", ADAM_EPS],
        ["Weight decay", WEIGHT_DECAY], ["Loss", "CrossEntropyLoss (mean, label_smoothing=0)"],
        ["분류기 Dropout", DROPOUT], ["사전학습", "weights=None"],
        ["Scheduler / 조기종료 / gradient clipping", "미사용"],
        ["최종 평가 모델", f"마지막 {EPOCHS} epoch (테스트로 선택하지 않음)"],
        ["중간 테스트", "매 epoch, eval + no_grad; 콘솔만 출력"],
        ["연산", "FP32, AMP/TF32 미사용"],
        ["결정성", "deterministic=True, benchmark=False, warn_only=True"],
        ["CPU threads", CPU_THREADS], ["DataLoader workers", 0],
        ["학습 shuffle / 테스트 shuffle", "True / False"],
        ["DataLoader drop_last", False], ["pin_memory", "CUDA 사용 시 True"],
        ["데이터", "A–E 5명 × 50시행, 시행당 3,000샘플 × 2채널"],
        ["Sampling rate", "1,000 Hz"], ["Window / hop", "300 / 150샘플 (50% overlap)"],
        ["정규화", "윈도우별 두 채널 공동 min–max, eps=1e-8"],
        ["CWT", "morl, scales=1..32, 절댓값"],
        ["입력", "(3,32,300): 2채널 CWT + 두 CWT 평균"],
        ["기본 필터링", "공개 CSV는 이미 필터링되어 재적용하지 않음"],
        ["--raw-data 필터", "60 Hz notch Q=30 → 4차 20–499 Hz band-pass, filtfilt"],
        ["선택적 CV", f"학습 80% 내부 {CV_FOLDS}-fold, stratified, shuffle=True, seed={SEED}"],
        ["추론 시간", f"batch=1, 예열 {INFERENCE_WARMUP}회, 평균 {INFERENCE_REPEATS}회, CUDA 동기화"],
        ["지표", "Accuracy, macro Precision / Recall / F1, zero_division=0"],
    ]

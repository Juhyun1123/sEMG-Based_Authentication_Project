# 현재 파라미터와 모델 구조

| 공통 설정 | 값 |
| --- | --- |
| 분할 | 시행 단위 계층화 Train 80% / Test 20% |
| Seed | 37 |
| Epochs | 45 |
| Batch size | 16 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Adam betas | (0.9, 0.999) |
| Adam eps | 1e-08 |
| Weight decay | 0.0 |
| Loss | CrossEntropyLoss (mean, label_smoothing=0) |
| 분류기 Dropout | 0.5 |
| 사전학습 | weights=None |
| Scheduler / 조기종료 / gradient clipping | 미사용 |
| 최종 평가 모델 | 마지막 45 epoch (테스트로 선택하지 않음) |
| 중간 테스트 | 매 epoch, eval + no_grad; 콘솔만 출력 |
| 연산 | FP32, AMP/TF32 미사용 |
| 결정성 | deterministic=True, benchmark=False, warn_only=True |
| CPU threads | 4 |
| DataLoader workers | 0 |
| 학습 shuffle / 테스트 shuffle | True / False |
| DataLoader drop_last | False |
| pin_memory | CUDA 사용 시 True |
| 데이터 | A–E 5명 × 50시행, 시행당 3,000샘플 × 2채널 |
| Sampling rate | 1,000 Hz |
| Window / hop | 300 / 150샘플 (50% overlap) |
| 정규화 | 윈도우별 두 채널 공동 min–max, eps=1e-8 |
| CWT | morl, scales=1..32, 절댓값 |
| 입력 | (3,32,300): 2채널 CWT + 두 CWT 평균 |
| 기본 필터링 | 공개 CSV는 이미 필터링되어 재적용하지 않음 |
| --raw-data 필터 | 60 Hz notch Q=30 → 4차 20–499 Hz band-pass, filtfilt |
| 선택적 CV | 학습 80% 내부 5-fold, stratified, shuffle=True, seed=37 |
| 추론 시간 | batch=1, 예열 10회, 평균 100회, CUDA 동기화 |
| 지표 | Accuracy, macro Precision / Recall / F1, zero_division=0 |

| 모델 | 구조 | 학습 파라미터 수 |
| --- | --- | --- |
| 2D CNN | Conv(3→32,k3,p1) / ReLU / MaxPool(k2,s2) / Conv(32→64,k3,p1) / ReLU / GAP / Dropout(0.5) / FC(64→5) | 19,717 |
| ResNet18 | 기본 torchvision: conv7,s2,p3,64채널 / blocks [2,2,2,2] / FC(512→5), 앞 Dropout(0.5) | 11,179,077 |
| DenseNet161 | 기본 torchvision: conv7,s2,p3,96채널 / growth 48 / blocks [6,12,36,24] / compression 0.5 / FC(2208→5), 앞 Dropout(0.5) | 26,483,045 |

Conv bias 및 BatchNorm/초기화는 torchvision 기본값을 사용한다. 간단한 CNN의 Conv는 bias=True, stride=1, dilation=1, groups=1이다. 기본 ResNet/DenseNet BatchNorm eps=1e-5, momentum=0.1, affine=True이며 DenseNet 내부 drop_rate=0, memory_efficient=False다. 구조별 값은 서로 다르지만 외부 학습 설정은 동일하다.

자료 근거: 2주차 p.12–16의 윈도우·정규화·CWT·분할·DenseNet 학습 예시, 3주차 p.11의 CNN/ResNet 예시, 논문추가 자료 p.35의 공통 학습 설정. Dropout 비율은 자료에 정확한 값이 없어 기존 0.5를 유지했다. 공개 CSV는 이미 필터링되어 중복 필터링하지 않는다.

시드가 같아도 구조·전처리·라이브러리·장비·연산 순서가 달라지면 결과는 달라질 수 있다. 중간 테스트는 관찰만 하며 최적 epoch/하이퍼파라미터 선택에 사용하지 않는다.


현재 프로젝트 `.venv`의 설치 환경은 다음과 같다. `requirements.txt`도 이 버전에 맞췄다.

| 항목 | 현재 환경 |
| --- | --- |
| Python | 3.14.7 |
| PyTorch | 2.14.0+cu130 |
| Torchvision | 0.29.0+cu130 |
| CUDA runtime | 13.0 |
| NumPy | 2.5.3 |
| SciPy | 1.18.1 |
| scikit-learn | 1.9.1 |
| PyWavelets | 1.10.0 |
| pandas | 3.0.6 |
| matplotlib | 3.11.2 |

선택한 측정 결과: `results/submission_20261003_151800_386572/`. 현재 설정의 시드는 37이다. 기존 결과 파일에는 실행 당시 시드·환경 기록이 없으므로 해당 실행이 모든 현재 설정을 사용했다는 사실을 CSV만으로 소급 검증할 수는 없다.

테스트 결과를 비교하여 이번 실행을 선택했으므로, 이 결과는 선택한 분할에서의 성능으로 보고한다. 새로운 독립 데이터에서의 성능 또는 검증 기반 최적 시드라고 주장하지 않는다.

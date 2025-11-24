# LumiNet 어댑터 학습 가이드

`train_adapter.py`는 ControlNet 분기의 어댑터만을 학습하도록 설계된 파이토치 라이트닝 스크립트입니다. 매니페스트(CSV) 또는 **이미지 폴더만**으로도 학습을 돌릴 수 있으니 아래 절차에 따라 데이터 준비, 체크포인트 설정, 실행 방법을 확인하세요.

## 1. 데이터 매니페스트 준비 (이미 힌트맵이 있을 때)
CSV 파일에 최소 `image`, `hint` 컬럼을 포함해야 합니다. `prompt`는 선택 사항입니다.

```csv
image,hint,prompt
/path/to/rgb1.png,/path/to/hint1.png,"a sunlit office"
/path/to/rgb2.png,/path/to/hint2.npy,"a night street"
```

- **image**: RGB 입력 이미지 경로.
- **hint**: 조명 위치/방향을 표현한 힌트 맵(이미지 또는 `.npy`). 채널 수는 `models/cldm_v21_LumiNet.yaml`의 `hint_channels`에 맞춰 자동 패딩/자르기가 되며, 해상도도 학습 이미지 크기(`--image-size`)로 자동 리사이즈됩니다. 값 범위가 `[0, 1]`이나 `[0, 255]`여도 학습 시 `[-1, 1]`로 정규화됩니다.
- **prompt**: (선택) 텍스트 프롬프트. 비워두면 빈 문자열이 사용됩니다.

### 이미지 폴더만 있을 때 (자동 힌트 생성)
`--train-images`에 RGB 이미지 폴더를 넘기면 힌트맵 없이도 학습됩니다. 내부적으로 **밝은 영역 마스크(가우시안 블러 + 퍼센타일 스레시홀드)**를 3채널로 반복해 RGB와 붙여 6채널 힌트를 매 스텝 자동 생성합니다.

```bash
python train_adapter.py \
  --config models/cldm_v21_LumiNet.yaml \
  --train-images /data/rgb_only \
  --image-size 512 \
  --auto-bright-percentile 95 \
  --auto-blur-ksize 11
```

- `--val-images`를 추가하면 검증도 동일하게 자동 힌트로 수행됩니다.
- `--auto-bright-percentile`을 낮추면 어두운 장면에서 더 넓은 영역을 밝은 영역으로 잡습니다.
- `--auto-blur-ksize`는 홀수 커널만 사용하며, 짝수를 넣으면 자동으로 +1 되어 적용됩니다.

## 2. 필수 자원
- 기본 LumiNet 설정 파일: `models/cldm_v21_LumiNet.yaml`
- 사전학습 체크포인트(선택): `--pretrained-ckpt /path/to/luminet.ckpt`
- 어댑터 초기값(선택): `--adapter-init /path/to/adapter_only.ckpt`

## 3. 실행 예시
```bash
python train_adapter.py \
  --config models/cldm_v21_LumiNet.yaml \
  --train-manifest /data/train.csv \
  --val-manifest /data/val.csv \
  --pretrained-ckpt /weights/luminet.ckpt \
  --logdir logs/adapter_run \
  --batch-size 2 \
  --image-size 512 \
  --learning-rate 1e-5 \
  --max-steps 20000
```

선택적으로 UNet 크로스어텐션을 함께 미세조정하려면 `--train-cross-attention` 플래그를 추가하세요.

## 4. 체크포인트 및 로그
- 체크포인트는 `--logdir` 하위에 `adapter-epoch-step.ckpt` 형식으로 저장됩니다.
- `--resume-from`에 저장된 체크포인트 경로를 지정하면 중단 지점부터 재개합니다.

## 5. 하드웨어/정밀도 팁
- GPU가 감지되면 자동으로 1개 GPU를 사용합니다. CPU 학습은 지원하지만 매우 느립니다.
- `--precision 16`(기본값)으로 자동 혼합 정밀도를 사용합니다. 문제가 발생하면 `--precision 32`로 변경하세요.

## 6. 데이터 검증
- `--val-manifest`를 제공하면 `val/loss_simple_ema`를 기준으로 상위 3개 체크포인트를 저장합니다.
- 제공하지 않으면 학습 손실(`train/loss`) 기준으로만 저장합니다.

## 7. 결과 활용
생성된 체크포인트는 `models/cldm_v21_LumiNet.yaml`의 `control_stage_config`를 유지한 채 `--adapter-init` 또는 모델 로드시 `control_model.load_state_dict`에 전달하여 인퍼런스에 사용할 수 있습니다.

## 8. 힌트맵(6채널) 빠르게 만드는 법
이미지밖에 없다면 `tools/generate_hint_maps.py`로 6채널 힌트맵(`RGB + 밝은 영역 마스크 3채널`)과 매니페스트를 자동으로 만들 수 있습니다.

```bash
# 이미지만 모아둔 디렉터리 기준
python tools/generate_hint_maps.py \
  --images /data/rgb_images \
  --output /data/hints \
  --manifest /data/train.csv \
  --resize 512 \
  --bright-percentile 95
```

- 결과로 각 이미지별 `*_hint.npy`가 생성되며, `train_adapter.py --train-manifest /data/train.csv`로 바로 학습할 수 있습니다.
- `--bright-percentile`을 낮추면 어두운 장면에서도 넓은 영역이 힌트로 잡힙니다.
- 힌트맵은 `.npy` 포맷이므로 `models/cldm_v21_LumiNet.yaml`의 `hint_channels: 6` 규격과 맞게 저장됩니다.

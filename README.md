# Chalna (찰나)

> 찰나(刹那) - 매우 짧은 순간. 정확한 타이밍을 잡는다.

SRT 자막 생성 서비스. VibeVoice ASR + Qwen Forced Alignment를 결합하여 정확한 타임스탬프와 화자 분리가 포함된 자막을 생성합니다.

## Features

- **VibeVoice ASR**: 60분 오디오 처리, 화자 분리 지원
- **Qwen Forced Alignment**: 단어 수준의 정확한 타임스탬프 보정
- **LLM Refinement**: Codex CLI를 통한 자막 교정 및 긴 문장 분리
- **다양한 출력 형식**: SRT, JSON
- **CLI & REST API**: 로컬 사용 및 서비스 배포 지원

## Installation

```bash
# Clone with submodules
git clone --recursive https://github.com/jonhpark/chalna.git
cd chalna

# Install dependencies
pip install -e .

# Install VibeVoice dependencies
pip install -e external/VibeVoice
```

### LLM Refinement 설정 (Optional)

LLM refinement 기능을 사용하려면 [Codex CLI](https://github.com/openai/codex)가 필요합니다.

```bash
# Codex CLI 설치
npm install -g @openai/codex

# venv 환경에서 실행 시 symlink 필요
# (venv의 PATH에 npm global bin이 포함되지 않음)
ln -sf $(which codex) /path/to/chalna/venv/bin/codex

# 예시: nvm 사용 시
ln -sf ~/.nvm/versions/node/$(node -v)/bin/codex ./venv/bin/codex
```

**참고**: venv 환경에서 Chalna 서버를 실행할 때, codex CLI가 PATH에 없으면 LLM refinement가 자동으로 스킵됩니다. 위 symlink를 설정하면 venv 내에서도 codex를 사용할 수 있습니다.

## Usage

### CLI

```bash
# Basic transcription
chalna transcribe audio.mp3 -o output.srt

# With context/hotwords
chalna transcribe meeting.mp3 -o meeting.srt -c "참석자: 철수, 영희"

# Without speaker labels
chalna transcribe lecture.wav -o lecture.srt --no-speaker

# Skip forced alignment (faster, less accurate timestamps)
chalna transcribe audio.mp3 -o output.srt --no-align

# JSON output
chalna transcribe audio.mp3 -o output.json --json
```

### REST API

```bash
# Start server
chalna serve --host 0.0.0.0 --port 8000

# Transcribe (curl)
curl -X POST http://localhost:8000/transcribe \
  -F "file=@audio.mp3" \
  -F "output_format=srt"
```

### Python

```python
from chalna import ChalnaPipeline

pipeline = ChalnaPipeline()
result = pipeline.transcribe("audio.mp3", context="참석자: 철수, 영희")

# Get SRT
srt_content = result.to_srt()

# Get segments
for seg in result.segments:
    print(f"[{seg.speaker_id}] {seg.start_time:.2f}s: {seg.text}")
```

## API Reference

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/transcribe` | 동기 자막 생성 |
| POST | `/transcribe/async` | 비동기 자막 생성 (긴 파일용) |
| GET | `/jobs/{job_id}` | 비동기 작업 상태 조회 |
| GET | `/health` | 서버 상태 확인 |

### Transcription Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | file | required | 오디오/비디오 파일 |
| `language` | string | auto | 언어 힌트 (ko, en, ja, zh) |
| `context` | string | null | 컨텍스트/핫워드 |
| `use_alignment` | bool | true | Qwen forced alignment 사용 |
| `use_llm_refinement` | bool | true | LLM 자막 교정 사용 |
| `output_format` | string | srt | 출력 형식 (srt, json) |

## Performance Reference

> **환경**: NVIDIA RTX 5090 (32GB VRAM), CUDA
> **측정 기준**: 한국어 음성, 기본 설정 (`use_alignment=true`, `use_llm_refinement=true`)

### 파이프라인 단계별 처리 시간

| 단계 | 설명 | GPU VRAM | 비고 |
|------|------|----------|------|
| 1. VibeVoice ASR | 음성 인식 + 화자 분리 | ~24GB | 오디오 길이에 선형 비례 |
| 2. Qwen Alignment | 타임스탬프 보정 | ~4GB | 세그먼트당 ffmpeg + 추론 |
| 3. LLM Refinement | 자막 교정 (Codex CLI) | 0 (외부 API) | 세그먼트 수에 비례, 네트워크 의존 |

### 오디오 길이별 실측 처리 시간

> 실측일: 2026-02-09, 한국어 팟캐스트 음성, `use_alignment=true`, `use_llm_refinement=true`
> LLM Refinement: 병렬 처리 (`max_workers=5`)

| 오디오 길이 | ASR | Alignment | LLM Refine | **합계** | 세그먼트 | 청크 |
|------------|-----|-----------|------------|---------|---------|------|
| 2분 | 18s | 1s | 13s | **40s** | 30 | 0 |
| 5분 | 44s | 3s | 42s | **1분 30초** | 109 | 0 |
| 10분 | 90s | 6s | 45s | **2분 25초** | 241 | 0 |
| 30분 | 261s | 18s | 131s | **6분 50초** | 719 | 3 |
| 114분 | 940s | 69s | 326s | **22분 16초** | 2622 | 12 |

**참고 사항**:
- **ASR**: 오디오 길이에 선형 비례 (~8초/분). 11분 이상은 10분 청크로 분할 처리
- **Alignment**: 전체 시간의 2~5%. 세그먼트당 ~0.03초
- **LLM Refinement**: 30개 세그먼트 단위로 Codex API 병렬 호출 (5 workers). Codex API 부하에 따라 변동 가능
- LLM Refinement 실패 시 자동 스킵되며, Alignment까지의 결과가 반환됨 (`metadata.refined: false`)
- **ETA 예측**: 작업 제출 시 `estimated_completion` 반환. 이전 작업 통계 기반 자동 보정

### Job Queue 동시 요청 테스트

> 2분 오디오 × 3건 동시 요청 (`/transcribe/async`), FIFO 단일 워커 큐

| 요청 | 총 시간 | queue_position 변화 | 결과 |
|------|--------|-------------------|------|
| Job 1 | 36s | `0` → 완료 | 즉시 처리 |
| Job 2 | 66s | `1` → `0` → 완료 | Job 1 완료 후 처리 |
| Job 3 | 93s | `2` → `1` → `0` → 완료 | Job 2 완료 후 처리 |

- 3건 직렬 처리 확인: wall time 94s ≈ 단일 작업 ~31s × 3
- `queue_position`으로 대기 순서 실시간 확인 가능 (0=처리중, 1+=대기중)
- GPU 충돌 없이 모든 요청 성공

### 제한 사항

| 항목 | 제한 |
|------|------|
| 최대 오디오 길이 | 10시간 (36,000초) |
| 최대 파일 크기 | 2GB |
| 지원 오디오 형식 | mp3, wav, flac, aac, ogg, opus, m4a, wma |
| 지원 영상 형식 | mp4, mov, webm, mkv, avi |
| 동시 요청 | FIFO 단일 워커 큐 (직렬 처리) |

> **영상 파일**: 오디오 스트림만 추출하여 처리합니다. 영상 디코딩은 수행하지 않으며, 오디오와 동일하게 처리됩니다. 단, 파일 크기가 크므로 2GB 제한에 유의하세요.

## Known Issues / TODO

- [x] **동시 요청 보호**: FIFO 단일 워커 큐로 직렬 처리 (GPU OOM / 파이프라인 상태 충돌 방지)
- [x] **LLM Refinement 상태 반환**: `metadata.refined: bool` 추가
- [x] **길이별 성능 벤치마크**: 2분/5분/10분/30분/114분 실측 완료 (위 Performance Reference 참조)
- [x] **LLM Refinement 병렬화**: `ThreadPoolExecutor(max_workers=5)`로 Codex API 병렬 호출 (114분 기준 28분→5분)
- [x] **ETA 예측**: 작업 제출/조회 시 예상 대기시간, 처리시간, 완료시각 반환. 이전 작업 통계 기반 자동 보정

## License

MIT License

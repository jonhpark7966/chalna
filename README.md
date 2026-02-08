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
| 1. VibeVoice ASR | 음성 인식 + 화자 분리 | ~24GB | 전체 시간의 ~85% |
| 2. Qwen Alignment | 타임스탬프 보정 | ~4GB | 세그먼트당 ffmpeg + 추론 |
| 3. LLM Refinement | 자막 교정 (Codex CLI) | 0 (외부 API) | 30 세그먼트/청크, 네트워크 의존 |

### 오디오 길이별 예상 처리 시간

<!-- TODO: 아래 표는 114분 오디오 벤치마크(ASR 16분 + Alignment 2분)에서 추정한 값입니다.
     실제 길이별 벤치마크 테스트를 실행하여 검증이 필요합니다.
     테스트 대상: 5분, 10분, 30분, 1시간, 2시간, 5시간, 10시간
     각 길이에서 단계별 소요 시간과 GPU 피크 메모리를 측정할 것.
     (sample/ 디렉토리에 다양한 길이의 테스트 오디오 준비 필요) -->

| 오디오 길이 | ASR (추정) | Alignment (추정) | LLM Refine (추정) | 합계 (추정) | 청크 수 |
|------------|-----------|-----------------|-------------------|-----------|--------|
| 5분 | ~1분 | ~10초 | ~10초 | **~1.5분** | 1 |
| 10분 | ~1.5분 | ~20초 | ~15초 | **~2.5분** | 1 |
| 30분 | ~4분 | ~1분 | ~40초 | **~6분** | 3 |
| 1시간 | ~8분 | ~2분 | ~1.5분 | **~12분** | 6 |
| 2시간 | ~16분 | ~4분 | ~3분 | **~23분** | 12 |
| 5시간 | ~40분 | ~10분 | ~7분 | **~57분** | 30 |
| 10시간 (최대) | ~80분 | ~20분 | ~15분 | **~2시간** | 60 |

**참고 사항**:
- ASR은 오디오 길이에 거의 선형 비례 (1분 오디오 ≈ 8~10초 처리)
- 11분 이상 오디오는 10분 청크로 분할 처리됨
- LLM Refinement은 네트워크 상태와 Codex API 부하에 따라 변동 가능
- LLM Refinement 실패 시 자동 스킵되며, Alignment까지의 결과가 반환됨

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

- [x] **동시 요청 보호**: FIFO 단일 워커 큐로 직렬 처리 (GPU OOM / 파이프라인 상태 충돌 방지). `GET /jobs/{id}`에서 `queue_position`으로 대기 순서 확인 가능
- [x] **LLM Refinement 상태 반환**: `metadata.refined: bool` 추가. Refinement 스킵 여부를 기본 응답에서 확인 가능
- [ ] **길이별 성능 벤치마크**: 5분/10분/30분/1시간/2시간/5시간/10시간 오디오로 단계별 소요 시간 실측 필요

## License

MIT License

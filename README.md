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

## License

MIT License

# Chalna (찰나)

> 찰나(刹那) - 매우 짧은 순간. 정확한 타이밍을 잡는다.

SRT 자막 생성 서비스. VibeVoice ASR + Qwen Forced Alignment를 결합하여 정확한 타임스탬프와 화자 분리가 포함된 자막을 생성합니다.

## Features

- **VibeVoice ASR**: 60분 오디오 처리, 화자 분리 지원
- **Qwen Forced Alignment**: 단어 수준의 정확한 타임스탬프 보정
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

## License

MIT License

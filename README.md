# Chalna (찰나)

> 찰나(刹那) - 매우 짧은 순간. 정확한 타이밍을 잡는다.

SRT 자막 생성 서비스. ElevenLabs `scribe_v2`로 음성 인식, 단어 timestamp, 화자 분리를 받아 Chalna의 SRT/JSON 출력과 optional LLM segmentation/refinement를 제공합니다.

## Features

- **ElevenLabs Scribe v2**: 음성 인식, word timestamp, 선택적 speaker diarization
- **LLM Segmentation**: Scribe word timestamp를 source of truth로 두고 GPT 기반 word range planning
- **LLM Refinement**: Codex CLI를 통한 자막 교정 및 긴 문장 분리
- **Response/segment cache**: 같은 음원/옵션 요청은 Scribe와 LLM segmentation 호출을 재사용
- **다양한 출력 형식**: SRT, JSON
- **Web UI**: FastAPI 내장 Web UI
- **CLI & REST API**: 로컬 사용 및 서비스 배포 지원

## Installation

```bash
git clone https://github.com/jonhpark/chalna.git
cd chalna
pip install -e .
```

### ElevenLabs 설정

```bash
export ELEVENLABS_API_KEY="..."
```

선택 설정:

| 환경 변수 | 기본값 | 설명 |
|-----------|--------|------|
| `ELEVENLABS_BASE_URL` | `https://api.elevenlabs.io` | ElevenLabs API base URL |
| `SCRIBE_MODEL_ID` | `scribe_v2` | Scribe 모델 ID |
| `CHALNA_SCRIBE_CACHE_DIR` | `chalna/results/scribe_cache` | Scribe 원문 응답 캐시 디렉터리 |
| `CHALNA_SCRIBE_TIMEOUT` | `600` | Scribe API timeout seconds |
| `CHALNA_SCRIBE_DELIVERY_MODE` | `webhook` | 기본은 비동기 수신, 비상 동기 경로는 `sync` |
| `ELEVENLABS_WEBHOOK_ID` | 없음 | Speech-to-Text 완료 webhook ID |
| `ELEVENLABS_WEBHOOK_SECRET` | 없음 | HMAC 검증용 webhook secret |
| `CHALNA_SCRIBE_WEBHOOK_TIMEOUT` | `7200` | 완료 webhook 대기 시간(초) |
| `CHALNA_SCRIBE_RECOVERY_TIMEOUT` | `90` | 기존 provider transcript 조회 timeout(초) |
| `CHALNA_LLM_SEGMENTATION_CACHE_DIR` | `chalna/results/segment_cache` | LLM segment plan 캐시 디렉터리 |
| `CHALNA_LLM_SEGMENTATION_MODEL` | `gpt-5.5` | word-to-segment planning 모델 |
| `CHALNA_LLM_SEGMENTATION_REASONING_EFFORT` | `xhigh` | planning reasoning effort |
| `CHALNA_LLM_SEGMENTATION_TIMEOUT` | `180` | planning timeout seconds |

`SCRIBE_CACHE_DIR`, `SCRIBE_TIMEOUT`도 이전 설정명 호환을 위해 계속 동작합니다.

FastAPI 서비스의 신규 Scribe 작업은 기본적으로 webhook delivery를 사용합니다. 독립 실행형
`chalna transcribe` 명령은 callback 서버와 durable runtime이 없으므로 운영자용 sync fallback을
명시적으로 사용합니다.

### LLM 설정 (Optional)

LLM segmentation/refinement 기능을 사용하려면 Codex CLI가 필요합니다. 사용하지 않으려면 Web UI에서 `LLM 자막 구간 나누기` 또는 `LLM 자막 교정`을 끄거나 CLI에서 `--no-llm-segmentation`, `--no-llm-refine`을 사용하세요.

```bash
npm install -g @openai/codex
```

## Usage

### CLI

```bash
chalna transcribe audio.mp3 -o output.srt
chalna transcribe meeting.mp3 -o meeting.srt -c "참석자: 철수, 영희"
chalna transcribe lecture.wav -o lecture.srt --no-speaker
chalna transcribe audio.mp3 -o output.srt --no-llm-segmentation
chalna transcribe audio.mp3 -o output.srt --no-llm-refine
chalna transcribe meeting.mp3 --num-speakers 2 --no-tag-audio-events
chalna transcribe audio.mp3 -o output.json --json
```

`--no-align`은 이전 버전 호환용 옵션이며 현재는 무시됩니다. Scribe v2 이후 Qwen forced aligner는 실행되지 않습니다.

### REST API / Web UI

```bash
chalna serve
# Web UI: http://localhost:7861/
# API Docs: http://localhost:7861/docs
```

```bash
curl -X POST http://localhost:7861/transcribe \
  -F "file=@audio.mp3" \
  -F "diarize=true" \
  -F "tag_audio_events=true" \
  -F "num_speakers=2" \
  -F "use_llm_segmentation=true" \
  -F "use_llm_refinement=true" \
  -F "output_format=srt"
```

비동기 처리:

```bash
curl -X POST http://localhost:7861/transcribe/async \
  -F "file=@long_audio.mp3" \
  -F "output_format=srt"

curl http://localhost:7861/jobs/{job_id}
```

### Python

```python
from chalna import ChalnaPipeline, LlmSegmentationOptions, ScribeOptions

pipeline = ChalnaPipeline(use_llm_segmentation=True, use_llm_refinement=True)
result = pipeline.transcribe(
    "audio.mp3",
    context="참석자: 철수, 영희",
    scribe_options=ScribeOptions(
        diarize=True,
        tag_audio_events=True,
        num_speakers=2,
    ),
    llm_segmentation_options=LlmSegmentationOptions(enabled=True),
)

print(result.to_srt())
```

## Transcription Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | file | required | 오디오/비디오 파일 |
| `language` | string | auto | Scribe language hint |
| `context` | string | null | LLM refinement 참고 컨텍스트 |
| `diarize` | bool | true | Scribe speaker diarization |
| `tag_audio_events` | bool | true | Scribe audio event 태깅 |
| `num_speakers` | int | null | 예상 화자 수, 1-32 |
| `use_llm_segmentation` | bool | true | Scribe words를 LLM word range plan으로 Segment화 |
| `bypass_llm_segmentation_cache` | bool | false | Segment plan cache read 없이 LLM segmentation 재계산 |
| `use_llm_refinement` | bool | true | Scribe 이후 LLM 자막 교정 |
| `use_alignment` | bool | ignored | Deprecated. Qwen forced alignment는 실행되지 않음 |
| `output_format` | string | srt | 출력 형식 (`srt`, `json`) |

## Pipeline

```text
validating
 -> scribe_v2 transcribe
 -> optional LLM segment plan
 -> optional LLM refine
 -> timestamp overlap cleanup only
 -> SRT/JSON 저장
```

Qwen forced aligner는 어떤 단계에서도 실행되지 않습니다. LLM segmentation은 `start_time/end_time`을 만들지 않고 word index range만 결정합니다. 실제 timestamp는 항상 Scribe word timestamp에서 계산됩니다. LLM segmentation이 실패하거나 invalid range를 반환하면 기존 heuristic segmenter로 fallback합니다. 이후 LLM refinement가 segment를 추가로 분리하면 가능한 경우 Scribe word timestamp로 경계를 추정하고, 실패하면 기존 segment 시간을 균등 분할합니다.

## Scribe Cache

Scribe API 호출 비용을 줄이기 위해 원문 응답은 `results/scribe_cache/{cache_key}.json`에 저장됩니다. 캐시 키는 현재 파일 내용 해시가 아니라 아래 메타데이터와 요청 옵션으로 계산합니다.

- file size, duration, format, codec, sample rate, channels
- model id, language code
- diarize, tag_audio_events, num_speakers
- timestamps granularity

같은 음원이라도 Scribe 옵션이 달라지면 다른 캐시 엔트리를 사용합니다.

## Segment Cache

LLM segmentation 결과는 `results/segment_cache/{cache_key}.json`에 저장됩니다. 캐시 키는 Scribe cache key, LLM model, reasoning effort, prompt version, language code, Scribe 옵션, segmentation 옵션으로 계산합니다.

## Limits

| 항목 | 제한 |
|------|------|
| 최대 오디오 길이 | 10시간 (36,000초) |
| 최대 파일 크기 | 2GB |
| 지원 오디오 형식 | mp3, wav, flac, aac, ogg, opus, m4a, wma |
| 지원 영상 형식 | mp4, mov, webm, mkv, avi |
| 동시 요청 | FIFO 단일 워커 큐 |

## Migration Note

이전 버전의 VibeVoice-ASR 및 Qwen Forced Aligner 런타임은 Scribe v2로 대체되었습니다. 기존 `use_alignment` 옵션은 API 호환성을 위해 남아 있지만 무시됩니다.

## License

MIT License

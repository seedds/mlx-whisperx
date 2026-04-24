# mlx-whisperx

`mlx-whisperx` is a WhisperX-style transcription pipeline for Apple Silicon. It uses a vendored `mlx-whisper` ASR backend, then optionally applies WhisperX forced alignment and pyannote diarization.

The project is intended to provide a practical local pipeline with WhisperX-like JSON, subtitle, and text outputs while keeping ASR execution on MLX.

## Why This Project Exists

This project adds WhisperX-like functionality to an `mlx-whisper` workflow. The goal is to keep ASR inference on MLX for Apple Silicon while providing the pipeline pieces people commonly use from WhisperX: VAD chunking, forced alignment, word timestamps, diarization hooks, and familiar JSON/subtitle outputs.

The implementation borrows ideas and code from both upstream projects:

- WhisperX, for the pipeline structure, alignment workflow, diarization integration, and output conventions.
- `mlx-whisper`, for the Apple Silicon ASR backend and model execution path.

This repository vendors and adapts code where needed so the pieces work together as a standalone `mlx-whisperx` package.

## Pipeline

```text
audio -> VAD -> mlx-whisper ASR -> forced alignment -> optional diarization -> writers
```

Default behavior:

- ASR model: `mlx-community/whisper-turbo`
- VAD backend: Silero
- Decoding: beam search with `beam_size=5` and `temperature=0`
- Alignment: enabled for transcription
- Diarization: disabled unless `--diarize` is passed

## Installation

Clone the repository and install it into a Python environment:

```bash
git clone https://github.com/seedds/mlx-whisperx.git
cd mlx-whisperx
python -m pip install -e .
```

`ffmpeg` must be available on `PATH` because audio loading is handled through the ffmpeg CLI.

On macOS with Homebrew:

```bash
brew install ffmpeg
```

Optional pyannote VAD and diarization use pyannote models and may require a Hugging Face token, depending on the selected model.

## Quick Start

Write every supported output format next to the input file:

```bash
mlx-whisperx audio.wav --output_dir . --output_format all
```

Write only JSON:

```bash
mlx-whisperx audio.wav --output_dir . --output_format json
```

Use a specific `mlx-whisper` model and language:

```bash
mlx-whisperx audio.wav \
  --model mlx-community/whisper-large-v3-turbo \
  --language en \
  --output_format json
```

Generate subtitles:

```bash
mlx-whisperx audio.wav \
  --output_format srt \
  --max_line_width 42 \
  --max_line_count 2
```

Enable word highlighting in SRT/VTT:

```bash
mlx-whisperx audio.wav --output_format vtt --highlight_words True
```

## Python API

```python
from mlx_whisperx import transcribe

result = transcribe(
    "audio.wav",
    model="mlx-community/whisper-large-v3-turbo",
    language="en",
)
```

Print one transcript segment per line:

```python
for segment in result["segments"]:
    print(segment["text"].strip())
```

Print segment timestamps:

```python
for segment in result["segments"]:
    print(f"[{segment['start']:.2f} -> {segment['end']:.2f}] {segment['text'].strip()}")
```

Print word-level timestamps:

```python
for word in result["word_segments"]:
    print(f"[{word['start']:.2f} -> {word['end']:.2f}] {word['word']}")
```

Common API options match the CLI names:

```python
result = transcribe(
    "audio.wav",
    model="mlx-community/whisper-turbo",
    language="en",
    beam_size=5,
    temperature=0.0,
    no_align=False,
    diarize=False,
    vad_method="silero",
)
```

## Output Schema

JSON output follows the WhisperX-style shape:

```json
{
  "segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "Example transcript text.",
      "words": [
        {"word": "Example", "start": 0.0, "end": 0.6, "score": 0.98}
      ]
    }
  ],
  "word_segments": [
    {"word": "Example", "start": 0.0, "end": 0.6, "score": 0.98}
  ],
  "language": "en"
}
```

When diarization is enabled, speaker labels are included where available:

```json
{"word": "Hello", "start": 0.0, "end": 0.4, "score": 0.99, "speaker": "SPEAKER_00"}
```

## CLI Reference

Basic options:

- `--model`: `mlx-whisper` model directory or Hugging Face repo.
- `--language`: language code. If omitted, language is auto-detected by ASR.
- `--task`: `transcribe` or `translate`.
- `--output_format`: `all`, `srt`, `vtt`, `txt`, `tsv`, `json`, or `aud`.
- `--output_dir`: directory for output files.
- `--output_name`: custom output basename.
- `--verbose`: print transcript and logs.

Decoding options:

- `--temperature`: sampling temperature. Default is `0.0`.
- `--beam_size`: beam size when `temperature=0`. Default is `5`.
- `--best_of`: number of candidates when sampling with `temperature > 0`.
- `--patience`: beam-search patience.
- `--length_penalty`: beam-search length penalty.
- `--suppress_tokens`: comma-separated token IDs to suppress.
- `--suppress_numerals`: suppress numeric and currency-symbol tokens.
- `--initial_prompt`: initial prompt for ASR.
- `--hotwords`: hint phrases appended to the prompt.
- `--condition_on_previous_text`: prompt backend windows with previous text inside each VAD chunk.

Precision and model-cache options:

- `--compute_type float16`: force MLX ASR fp16. This is the default.
- `--compute_type float32`: force MLX ASR fp32.
- `--model_dir`: cache directory for alignment, pyannote VAD, and diarization models.
- `--model_cache_only`: cached alignment models only. This does not affect ASR model downloads yet.

VAD options:

- `--vad_method silero`: default VAD backend.
- `--vad_method pyannote`: use pyannote VAD if your environment supports it.
- `--vad_onset`: VAD onset threshold.
- `--vad_offset`: VAD offset threshold.
- `--vad_model`: Hugging Face pyannote segmentation model used with `--vad_method pyannote`. Defaults to `pyannote/segmentation-3.0`.
- `--chunk_size`: merged VAD chunk size in seconds.
- `--no_vad`: transcribe the full file as one chunk.
- `--vad_dump_path`: write VAD chunks and settings to JSON.

Alignment options:

- `--no_align`: skip forced alignment.
- `--align_model`: override the alignment model.
- `--interpolate_method`: `nearest`, `linear`, or `ignore`.
- `--return_char_alignments`: include character alignments in JSON.

Diarization options:

- `--diarize`: assign speaker labels.
- `--diarize_model`: pyannote diarization model name.
- `--min_speakers`: minimum speaker count.
- `--max_speakers`: maximum speaker count.
- `--speaker_embeddings`: include speaker embeddings in JSON when available.
- `--hf_token`: Hugging Face token for gated pyannote models.

Subtitle options:

- `--max_line_width`: target subtitle line width.
- `--max_line_count`: maximum lines per subtitle cue. Requires `--max_line_width`.
- `--max_words_per_line`: maximum words per subtitle cue.
- `--highlight_words`: underline the active word in SRT/VTT output.

## Examples

Inspect VAD chunks before ASR:

```bash
mlx-whisperx audio.wav \
  --output_format json \
  --vad_dump_path audio.vad.json
```

Run deterministic beam search explicitly:

```bash
mlx-whisperx audio.wav \
  --language en \
  --temperature 0 \
  --beam_size 5 \
  --output_format json
```

Use temperature fallback:

```bash
mlx-whisperx audio.wav \
  --temperature 0 \
  --temperature_increment_on_fallback 0.2
```

Suppress numerals and currency symbols during decoding:

```bash
mlx-whisperx audio.wav --suppress_numerals --output_format json
```

Use pyannote VAD instead of the default Silero VAD:

```bash
mlx-whisperx audio.wav \
  --vad_method pyannote \
  --vad_model pyannote/segmentation-3.0 \
  --hf_token YOUR_HF_TOKEN \
  --output_format json
```

Skip forced alignment:

```bash
mlx-whisperx audio.wav --no_align --output_format json
```

Run diarization:

```bash
mlx-whisperx audio.wav \
  --diarize \
  --hf_token YOUR_HF_TOKEN \
  --output_format json
```

Process multiple files:

```bash
mlx-whisperx first.wav second.wav third.wav --output_dir transcripts --output_format all
```

## Current Behavior and Limitations

- ASR decodes merged VAD chunks serially.
- There is no `batch_size` CLI or API option.
- `translate` skips forced alignment because alignment models are transcription-language specific.
- `model_dir` applies to alignment, pyannote VAD, and diarization model loading, not ASR model downloads.
- `model_cache_only` currently applies to cached alignment models only.
- Pyannote VAD and diarization depend on a compatible PyTorch, torchaudio, pyannote installation, and Hugging Face model access when the selected model is gated.
- The vendored ASR backend lives under `mlx_whisperx.backend.mlx_whisper` so decoder behavior can be changed without modifying external reference repositories.

## Development Checks

Compile the package:

```bash
python -m py_compile mlx_whisperx/**/*.py
```

Check CLI help:

```bash
python -m mlx_whisperx --help
```

Build a wheel:

```bash
python -m build
```

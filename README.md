# mlx-whisperx

WhisperX-style transcription pipeline using an internal MLX Whisper ASR backend.

This project is intentionally separate from the checked-out source folders:

- `whisperX/`
- `mlx-examples/`

Those folders are treated as read-only references/dependencies.

## Pipeline

```text
audio -> VAD -> internal MLX Whisper ASR -> forced alignment -> optional diarization -> writers
```

## Usage

```bash
mlx-whisperx audio.wav --model mlx-community/whisper-turbo --output_format all
```

Default decoding uses `--beam_size 5` and `--temperature 0`.

To write `000.json` next to `000.m4a`:

```bash
mlx-whisperx 000.m4a --output_format json --output_dir .
```

To inspect the VAD chunks used before ASR:

```bash
mlx-whisperx 000.m4a --output_format json --output_dir . --vad_dump_path 000.vad.json
```

To generate wrapped SRT subtitles:

```bash
mlx-whisperx 000.m4a --output_format srt --output_dir . --max_line_width 42 --max_line_count 2 --highlight_words False
```

To suppress numeric and currency-symbol tokens during decoding, matching the WhisperX alignment workaround:

```bash
mlx-whisperx 000.m4a --output_format srt --output_dir . --suppress_numerals
```

To use beam-search decoding:

```bash
mlx-whisperx 000.m4a --language en --output_format json --beam_size 5 --temperature 0
```

The default VAD backend is Silero so the base pipeline does not require a working
pyannote install. Use `--vad_method pyannote` if your PyTorch/pyannote stack is
configured correctly.

With diarization:

```bash
mlx-whisperx audio.wav --diarize --hf_token YOUR_HF_TOKEN
```

## Notes

The ASR backend is vendored under `mlx_whisperx.backend.mlx_whisper` so this project can support decoder changes such as beam search without modifying `mlx-examples/` or requiring an external `mlx-whisper` install. ASR still decodes VAD chunks serially; batched MLX decoding can be added later.

## Parity checks

Compare generated JSON against a WhisperX reference JSON:

```bash
mlx-whisperx-parity sample_out_000.json 000.json
```

For machine-readable metrics:

```bash
mlx-whisperx-parity sample_out_000.json 000.json --json
```

The parity harness reports schema compatibility, segment/word counts, language match, rough WER, text similarity, matched-word timing drift, and positional word timing drift. Matched-word timing drift ignores inserted/deleted words when possible, which makes it more useful when one output includes extra intro or outro text.

# mlx-whisperx

WhisperX-style transcription pipeline using `mlx-whisper` as the ASR backend.

This project is intentionally separate from the checked-out source folders:

- `whisperX/`
- `mlx-examples/`

Those folders are treated as read-only references/dependencies.

## Pipeline

```text
audio -> VAD -> mlx-whisper ASR -> forced alignment -> optional diarization -> writers
```

## Usage

```bash
mlx-whisperx audio.wav --model mlx-community/whisper-turbo --output_format all
```

The default VAD backend is Silero so the base pipeline does not require a working
pyannote install. Use `--vad_method pyannote` if your PyTorch/pyannote stack is
configured correctly.

With diarization:

```bash
mlx-whisperx audio.wav --diarize --hf_token YOUR_HF_TOKEN
```

## Notes

The first implementation calls the original `mlx_whisper.transcribe()` for each VAD chunk. This favors correctness and isolation. Batched MLX decoding can be added later inside this project without modifying `mlx-examples/`.

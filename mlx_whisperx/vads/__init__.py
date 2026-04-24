def get_vad_class(vad_method: str):
    if vad_method == "silero":
        from .silero import Silero

        return Silero
    if vad_method == "pyannote":
        from .pyannote import Pyannote

        return Pyannote
    raise ValueError(f"Invalid VAD method: {vad_method}")

"""Voice activity detection backend selection."""

def get_vad_class(vad_method: str):
    """Return the VAD implementation class for a CLI/API backend name.

    Imports are intentionally local so optional dependencies for one backend do not
    prevent using the other backend.
    """
    if vad_method == "silero":
        from .silero import Silero

        return Silero
    if vad_method == "pyannote":
        from .pyannote import Pyannote

        return Pyannote
    raise ValueError(f"Invalid VAD method: {vad_method}")

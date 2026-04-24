from ._compat import import_whisperx_module


def load_align_model(*args, **kwargs):
    return import_whisperx_module("alignment").load_align_model(*args, **kwargs)


def align(*args, **kwargs):
    return import_whisperx_module("alignment").align(*args, **kwargs)

"""Forced-alignment helpers for converting ASR segments into word timestamps.

The MLX Whisper backend can produce text segments, but WhisperX-style output needs
word-level timings. This module loads a CTC model, scores transcript characters
against the audio, then backtracks through a trellis to assign timestamps to words.
"""

from dataclasses import dataclass
import re
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .audio import SAMPLE_RATE, load_audio
from .log_utils import get_logger


# Adapted from WhisperX's forced-alignment implementation for local use without
# importing the upstream whisperx package at runtime.
logger = get_logger(__name__)

LANGUAGES_WITHOUT_SPACES = {"ja", "zh"}
PUNKT_LANGUAGES = {
    "cs": "czech",
    "da": "danish",
    "de": "german",
    "el": "greek",
    "en": "english",
    "es": "spanish",
    "et": "estonian",
    "fi": "finnish",
    "fr": "french",
    "it": "italian",
    "nl": "dutch",
    "no": "norwegian",
    "pl": "polish",
    "pt": "portuguese",
    "sl": "slovene",
    "sv": "swedish",
    "tr": "turkish",
    "ml": "malayalam",
    "ru": "russian",
}

DEFAULT_ALIGN_MODELS_TORCH = {
    "en": "WAV2VEC2_ASR_BASE_960H",
    "fr": "VOXPOPULI_ASR_BASE_10K_FR",
    "de": "VOXPOPULI_ASR_BASE_10K_DE",
    "es": "VOXPOPULI_ASR_BASE_10K_ES",
    "it": "VOXPOPULI_ASR_BASE_10K_IT",
}

DEFAULT_ALIGN_MODELS_HF = {
    "ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
    "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    "nl": "jonatasgrosman/wav2vec2-large-xlsr-53-dutch",
    "uk": "Yehor/wav2vec2-xls-r-300m-uk-with-small-lm",
    "pt": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese",
    "ar": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
    "cs": "comodoro/wav2vec2-xls-r-300m-cs-250",
    "ru": "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
    "pl": "jonatasgrosman/wav2vec2-large-xlsr-53-polish",
    "hu": "jonatasgrosman/wav2vec2-large-xlsr-53-hungarian",
    "fi": "jonatasgrosman/wav2vec2-large-xlsr-53-finnish",
    "fa": "jonatasgrosman/wav2vec2-large-xlsr-53-persian",
    "el": "jonatasgrosman/wav2vec2-large-xlsr-53-greek",
    "tr": "mpoyraz/wav2vec2-xls-r-300m-cv7-turkish",
    "da": "saattrupdan/wav2vec2-xls-r-300m-ftspeech",
    "he": "imvladikon/wav2vec2-xls-r-300m-hebrew",
    "vi": "nguyenvulebinh/wav2vec2-base-vi-vlsp2020",
    "ko": "kresnik/wav2vec2-large-xlsr-korean",
    "ur": "kingabzpro/wav2vec2-large-xls-r-300m-Urdu",
    "te": "anuragshas/wav2vec2-large-xlsr-53-telugu",
    "hi": "theainerd/Wav2Vec2-large-xlsr-hindi",
    "ca": "softcatala/wav2vec2-large-xlsr-catala",
    "ml": "gvs/wav2vec2-large-xlsr-malayalam",
    "no": "NbAiLab/nb-wav2vec2-1b-bokmaal-v2",
    "nn": "NbAiLab/nb-wav2vec2-1b-nynorsk",
    "sk": "comodoro/wav2vec2-xls-r-300m-sk-cv8",
    "sl": "anton-l/wav2vec2-large-xlsr-53-slovenian",
    "hr": "classla/wav2vec2-xls-r-parlaspeech-hr",
    "ro": "gigant/romanian-wav2vec2",
    "eu": "stefan-it/wav2vec2-large-xlsr-53-basque",
    "gl": "ifrz/wav2vec2-large-xlsr-galician",
    "ka": "xsway/wav2vec2-large-xlsr-georgian",
    "lv": "jimregan/wav2vec2-large-xlsr-latvian-cv",
    "tl": "Khalsuu/filipino-wav2vec2-l-xls-r-300m-official",
    "sv": "KBLab/wav2vec2-large-voxrex-swedish",
    "id": "cahya/wav2vec2-large-xlsr-indonesian",
}


def interpolate_nans(values: pd.Series, method: str = "nearest") -> pd.Series:
    """Fill missing timestamp values while preserving known alignment points."""
    if values.notnull().sum() > 1:
        return values.interpolate(method=method).ffill().bfill()
    return values.ffill().bfill()


def _sentence_spans(text: str, language: str) -> list[tuple[int, int]]:
    """Return character spans for sentence-like chunks inside an ASR segment."""
    try:
        from nltk.data import load as nltk_load

        punkt_lang = PUNKT_LANGUAGES.get(language, "english")
        try:
            splitter = nltk_load(f"tokenizers/punkt_tab/{punkt_lang}.pickle")
        except LookupError:
            import nltk

            nltk.download("punkt_tab", quiet=True)
            splitter = nltk_load(f"tokenizers/punkt_tab/{punkt_lang}.pickle")
        spans = list(splitter.span_tokenize(text))
        return spans or [(0, len(text))]
    except Exception:
        # Fall back to a punctuation regex if NLTK data is unavailable or unsupported
        # for the requested language.
        spans: list[tuple[int, int]] = []
        start = 0
        for match in re.finditer(r"[^.!?。！？]+[.!?。！？]?", text):
            end = match.end()
            if end > start:
                spans.append((start, end))
            start = end
        return spans or [(0, len(text))]


def load_align_model(
    language_code: str,
    device: str,
    model_name: Optional[str] = None,
    model_dir: Optional[str] = None,
    model_cache_only: bool = False,
):
    """Load the default or requested CTC model used for forced alignment.

    Torchaudio bundles are preferred when the language has a known built-in model;
    otherwise Hugging Face Wav2Vec2 CTC models are loaded. The returned metadata tells
    `align` how to map transcript characters into CTC token IDs.
    """
    try:
        import torchaudio
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    except Exception as exc:
        raise RuntimeError(
            "Alignment dependencies could not be imported. Install compatible torch, "
            "torchaudio, and transformers packages before using alignment, or pass no_align=True."
        ) from exc

    if model_name is None:
        # Match WhisperX defaults: use compact torchaudio bundles where available and
        # language-specific Hugging Face checkpoints for broader language coverage.
        if language_code in DEFAULT_ALIGN_MODELS_TORCH:
            model_name = DEFAULT_ALIGN_MODELS_TORCH[language_code]
        elif language_code in DEFAULT_ALIGN_MODELS_HF:
            model_name = DEFAULT_ALIGN_MODELS_HF[language_code]
        else:
            raise ValueError(
                f"No default alignment model for language {language_code!r}. "
                "Pass an explicit model with align_model."
            )

    if model_name in torchaudio.pipelines.__all__:
        pipeline_type = "torchaudio"
        bundle = torchaudio.pipelines.__dict__[model_name]
        align_model = bundle.get_model(dl_kwargs={"model_dir": model_dir}).to(device)
        labels = bundle.get_labels()
        align_dictionary = {char.lower(): idx for idx, char in enumerate(labels)}
    else:
        try:
            processor = Wav2Vec2Processor.from_pretrained(
                model_name,
                cache_dir=model_dir,
                local_files_only=model_cache_only,
            )
            align_model = Wav2Vec2ForCTC.from_pretrained(
                model_name,
                cache_dir=model_dir,
                local_files_only=model_cache_only,
            ).to(device)
        except Exception as exc:
            raise ValueError(
                f"The alignment model {model_name!r} could not be loaded from Hugging Face "
                "or torchaudio. Check the model name or cache settings."
            ) from exc
        pipeline_type = "huggingface"
        # Hugging Face vocabularies map characters/subwords to integer CTC classes.
        align_dictionary = {
            char.lower(): code for char, code in processor.tokenizer.get_vocab().items()
        }

    return align_model, {
        "language": language_code,
        "dictionary": align_dictionary,
        "type": pipeline_type,
    }


def align(
    transcript: Iterable[dict],
    model,
    align_model_metadata: dict,
    audio: str | np.ndarray,
    device: str,
    interpolate_method: str = "nearest",
    return_char_alignments: bool = False,
    print_progress: bool = False,
    combined_progress: bool = False,
    progress_callback=None,
) -> dict:
    """Align transcript segments to audio and return WhisperX-style segments.

    The algorithm normalizes each transcript into alignable characters, runs the CTC
    model over the corresponding audio span, computes a trellis of blank/token scores,
    backtracks the best path, and aggregates character timings into words and sentence
    subsegments.
    """
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required for alignment, or pass no_align=True.") from exc

    if not torch.is_tensor(audio):
        # Accept either a path, NumPy waveform, or Torch tensor to match the public API.
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)
    if len(audio.shape) == 1:
        audio = audio.unsqueeze(0)

    max_duration = audio.shape[1] / SAMPLE_RATE
    model_dictionary = align_model_metadata["dictionary"]
    model_lang = align_model_metadata["language"]
    model_type = align_model_metadata["type"]
    transcript = list(transcript)
    total_segments = len(transcript)

    segment_data: dict[int, dict] = {}
    for sdx, segment in enumerate(transcript):
        if print_progress:
            base_progress = ((sdx + 1) / total_segments) * 100
            percent_complete = 50 + base_progress / 2 if combined_progress else base_progress
            print(f"Progress: {percent_complete:.2f}%...")

        text = segment["text"]
        num_leading = len(text) - len(text.lstrip())
        num_trailing = len(text) - len(text.rstrip())
        per_word = list(text) if model_lang in LANGUAGES_WITHOUT_SPACES else text.split(" ")

        clean_char: list[str] = []
        clean_cdx: list[int] = []
        for cdx, char in enumerate(text):
            char_ = char.lower()
            if model_lang not in LANGUAGES_WITHOUT_SPACES:
                # Most CTC vocabularies use `|` as the word separator instead of a
                # literal space. Languages without spaces are aligned character-wise.
                char_ = char_.replace(" ", "|")

            if cdx < num_leading or cdx > len(text) - num_trailing - 1:
                continue
            if char_ in model_dictionary:
                clean_char.append(char_)
                clean_cdx.append(cdx)
            elif char_ not in {" ", "|"}:
                # Unknown non-space characters are retained and later mapped to a
                # wildcard CTC column so punctuation does not collapse nearby timing.
                clean_char.append(char_)
                clean_cdx.append(cdx)

        segment_data[sdx] = {
            "clean_char": clean_char,
            "clean_cdx": clean_cdx,
            "clean_wdx": list(range(len(per_word))),
            "sentence_spans": _sentence_spans(text, model_lang),
        }

    aligned_segments: list[dict] = []

    for sdx, segment in enumerate(transcript):
        t1 = segment["start"]
        t2 = segment["end"]
        text = segment["text"]
        avg_logprob = segment.get("avg_logprob")
        aligned_seg = {"start": t1, "end": t2, "text": text, "words": [], "chars": None}
        if avg_logprob is not None:
            aligned_seg["avg_logprob"] = avg_logprob
        if return_char_alignments:
            aligned_seg["chars"] = []

        if len(segment_data[sdx]["clean_char"]) == 0:
            logger.warning('Failed to align segment "%s": no alignable characters', text)
            aligned_segments.append(aligned_seg)
            continue
        if t1 >= max_duration:
            logger.warning('Failed to align segment "%s": start is past audio duration', text)
            aligned_segments.append(aligned_seg)
            continue

        text_clean = "".join(segment_data[sdx]["clean_char"])
        f1 = int(t1 * SAMPLE_RATE)
        f2 = int(t2 * SAMPLE_RATE)
        waveform_segment = audio[:, f1:f2]
        if waveform_segment.shape[-1] < 400:
            # Very short spans can fail inside convolutional frontends; pad to a safe
            # minimum while keeping `lengths` so torchaudio still knows true length.
            lengths = torch.as_tensor([waveform_segment.shape[-1]]).to(device)
            waveform_segment = torch.nn.functional.pad(
                waveform_segment,
                (0, 400 - waveform_segment.shape[-1]),
            )
        else:
            lengths = None

        with torch.inference_mode():
            if model_type == "torchaudio":
                emissions, _ = model(waveform_segment.to(device), lengths=lengths)
            elif model_type == "huggingface":
                emissions = model(waveform_segment.to(device)).logits
            else:
                raise NotImplementedError(f"Align model type {model_type!r} is not supported")
            emissions = torch.log_softmax(emissions, dim=-1)

        emission = emissions[0].cpu().detach()
        blank_id = 0
        for char, code in model_dictionary.items():
            if char in {"[pad]", "<pad>"}:
                blank_id = code

        has_wildcard = any(char not in model_dictionary for char in text_clean)
        if has_wildcard:
            # For characters missing from the CTC dictionary, add a synthetic wildcard
            # class scored as the best non-blank class at each frame.
            non_blank_mask = torch.ones(emission.size(1), dtype=torch.bool)
            non_blank_mask[blank_id] = False
            wildcard_col = emission[:, non_blank_mask].max(dim=1).values
            emission = torch.cat([emission, wildcard_col.unsqueeze(1)], dim=1)
            wildcard_id = emission.size(1) - 1
            tokens = [model_dictionary.get(char, wildcard_id) for char in text_clean]
        else:
            tokens = [model_dictionary[char] for char in text_clean]

        trellis = get_trellis(emission, tokens, blank_id)
        path = backtrack(trellis, emission, tokens, blank_id)
        if path is None:
            logger.warning('Failed to align segment "%s": backtrack failed', text)
            aligned_segments.append(aligned_seg)
            continue

        char_segments = merge_repeats(path, text_clean)
        duration = t2 - t1
        # Convert trellis frame indices back to seconds in the original audio span.
        ratio = duration * waveform_segment.size(0) / (trellis.size(0) - 1)

        char_segments_arr = []
        word_idx = 0
        for cdx, char in enumerate(text):
            start, end, score = None, None, None
            if cdx in segment_data[sdx]["clean_cdx"]:
                clean_idx = segment_data[sdx]["clean_cdx"].index(cdx)
                char_seg = char_segments[clean_idx]
                start = round(char_seg.start * ratio + t1, 3)
                end = round(char_seg.end * ratio + t1, 3)
                score = round(char_seg.score, 3)

            char_segments_arr.append(
                {
                    "char": char,
                    "start": start,
                    "end": end,
                    "score": score,
                    "word-idx": word_idx,
                }
            )
            if model_lang in LANGUAGES_WITHOUT_SPACES:
                word_idx += 1
            elif cdx == len(text) - 1 or text[cdx + 1] == " ":
                word_idx += 1

        char_segments_df = pd.DataFrame(char_segments_arr)
        aligned_subsegments: list[dict] = []
        char_segments_df["sentence-idx"] = None

        for sdx2, (sstart, send) in enumerate(segment_data[sdx]["sentence_spans"]):
            # Build smaller subtitle-friendly subsegments while preserving the original
            # text ordering and word timings.
            sentence_mask = (char_segments_df.index >= sstart) & (char_segments_df.index <= send)
            curr_chars = char_segments_df.loc[sentence_mask]
            char_segments_df.loc[sentence_mask, "sentence-idx"] = sdx2
            sentence_text = text[sstart:send]
            sentence_start = curr_chars["start"].min()
            end_chars = curr_chars[curr_chars["char"] != " "]
            sentence_end = end_chars["end"].max()
            sentence_words = []

            for word_idx in curr_chars["word-idx"].unique():
                word_chars = curr_chars.loc[curr_chars["word-idx"] == word_idx]
                word_text = "".join(word_chars["char"].tolist()).strip()
                if len(word_text) == 0:
                    continue
                word_chars = word_chars[word_chars["char"] != " "]
                word_start = word_chars["start"].min()
                word_end = word_chars["end"].max()
                word_score = round(word_chars["score"].mean(), 3)
                word_segment = {"word": word_text}
                if not np.isnan(word_start):
                    word_segment["start"] = word_start
                if not np.isnan(word_end):
                    word_segment["end"] = word_end
                if not np.isnan(word_score):
                    word_segment["score"] = word_score
                sentence_words.append(word_segment)

            if sentence_words:
                starts = pd.Series([word.get("start", np.nan) for word in sentence_words])
                ends = pd.Series([word.get("end", np.nan) for word in sentence_words])
                if starts.isna().any() and starts.notna().any():
                    # Missing timings usually come from punctuation or unsupported
                    # characters. Interpolation keeps the word list complete.
                    starts = interpolate_nans(starts, method=interpolate_method)
                    ends = interpolate_nans(ends, method=interpolate_method)
                    for idx, word in enumerate(sentence_words):
                        if "start" not in word and pd.notna(starts.iloc[idx]):
                            word["start"] = starts.iloc[idx]
                        if "end" not in word and pd.notna(ends.iloc[idx]):
                            word["end"] = ends.iloc[idx]

            subsegment = {
                "text": sentence_text,
                "start": sentence_start,
                "end": sentence_end,
                "words": sentence_words,
            }
            if avg_logprob is not None:
                subsegment["avg_logprob"] = avg_logprob
            if return_char_alignments:
                chars = curr_chars[["char", "start", "end", "score"]].copy()
                chars.fillna(-1, inplace=True)
                subsegment["chars"] = [
                    {key: value for key, value in char.items() if value != -1}
                    for char in chars.to_dict("records")
                ]
            aligned_subsegments.append(subsegment)

        aligned_subsegments_df = pd.DataFrame(aligned_subsegments)
        # Group adjacent sentence chunks that interpolate to identical boundaries.
        aligned_subsegments_df["start"] = interpolate_nans(
            aligned_subsegments_df["start"],
            method=interpolate_method,
        )
        aligned_subsegments_df["end"] = interpolate_nans(
            aligned_subsegments_df["end"],
            method=interpolate_method,
        )
        agg_dict = {"text": " ".join, "words": "sum"}
        if model_lang in LANGUAGES_WITHOUT_SPACES:
            agg_dict["text"] = "".join
        if return_char_alignments:
            agg_dict["chars"] = "sum"
        if avg_logprob is not None:
            agg_dict["avg_logprob"] = "first"
        aligned_subsegments = aligned_subsegments_df.groupby(
            ["start", "end"],
            as_index=False,
        ).agg(agg_dict).to_dict("records")
        if progress_callback is not None:
            progress_callback(((sdx + 1) / total_segments) * 100)
        aligned_segments += aligned_subsegments

    word_segments = [word for segment in aligned_segments for word in segment["words"]]
    return {"segments": aligned_segments, "word_segments": word_segments}


def get_trellis(emission, tokens, blank_id=0):
    """Build the CTC dynamic-programming table for transcript token alignment."""
    import torch

    num_frame = emission.size(0)
    num_tokens = len(tokens)
    trellis = torch.empty((num_frame + 1, num_tokens + 1))
    trellis[0, 0] = 0
    trellis[1:, 0] = torch.cumsum(emission[:, blank_id], 0)
    trellis[0, -num_tokens:] = -float("inf")
    trellis[-num_tokens:, 0] = float("inf")

    for t in range(num_frame):
        # At each frame, either stay on the current token by emitting blank, or advance
        # to the next transcript token by emitting that token.
        trellis[t + 1, 1:] = torch.maximum(
            trellis[t, 1:] + emission[t, blank_id],
            trellis[t, :-1] + emission[t, tokens],
        )
    return trellis


@dataclass
class Point:
    """Single CTC path point linking a transcript token to an emission frame."""

    token_index: int
    time_index: int
    score: float


def backtrack(trellis, emission, tokens, blank_id=0):
    """Recover the best CTC path from the completed trellis."""
    import torch

    j = trellis.size(1) - 1
    t_start = torch.argmax(trellis[:, j]).item()
    path = []
    for t in range(t_start, 0, -1):
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]
        # Record the frame score used by the winning transition for later averaging.
        prob = emission[t - 1, tokens[j - 1] if changed > stayed else blank_id].exp().item()
        path.append(Point(j - 1, t - 1, prob))
        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        return None
    return path[::-1]


@dataclass
class Segment:
    """Merged repeated CTC token span with an average confidence score."""

    label: str
    start: int
    end: int
    score: float

    @property
    def length(self):
        return self.end - self.start


def merge_repeats(path, transcript):
    """Collapse repeated CTC path points into character-level spans."""
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[idx].score for idx in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments

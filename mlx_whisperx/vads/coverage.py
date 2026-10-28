"""Energy-based coverage: decide which audio must reach the decoder.

VAD answers "is this speech?", which is the wrong question for deciding what to
transcribe. A neural VAD's speech probability collapses on stylised delivery --
counting, chanting, cheering, singing -- that is plainly audible and plainly
dialogue. Audio it rejects is never chunked and never decoded, so a missed
detection is unrecoverable and leaves no trace in the output.

This module answers the safer question, "is anything audible here?", using frame
energy. Energy is a far blunter signal than a neural VAD, but it fails in the
safe direction: it lets non-speech through rather than deleting speech. VAD is
retained for what it is good at -- locating the pauses between utterances, which
is where chunk seams belong.
"""

import re
from collections import Counter

import numpy as np

SAMPLE_RATE = 16000


def frame_rms(audio: np.ndarray, win: float = 0.25) -> np.ndarray:
    """Root-mean-square amplitude of each fixed-length frame."""
    n = int(win * SAMPLE_RATE)
    count = len(audio) // n
    if count == 0:
        return np.empty(0, dtype=np.float32)
    frames = np.asarray(audio[: count * n], dtype=np.float32).reshape(count, n)
    return np.sqrt((frames**2).mean(axis=1))


def speech_level(audio: np.ndarray, regions, win: float = 0.25) -> float:
    """Median frame energy inside known speech, used as the reference scale.

    Absolute amplitude varies with source, mastering, and channel selection, so
    a fixed energy floor cannot transfer between files. Measuring the level of
    this file's own speech makes the floor relative and therefore portable.
    """
    values = []
    for region in regions:
        start, end = (region.start, region.end) if hasattr(region, "start") else region
        segment = audio[int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)]
        values.append(frame_rms(segment, win))
    values = [v for v in values if v.size]
    if not values:
        return 0.0
    return float(np.median(np.concatenate(values)))


def audible_regions(
    audio: np.ndarray,
    floor: float,
    min_silent_run: float = 1.0,
    win: float = 0.25,
) -> list[tuple[float, float]]:
    """Intervals carrying audible signal, merged across short quiet gaps.

    This is the only mechanism permitted to remove audio from coverage. A brief
    dip between words must not break a region, so only a *run* of quiet frames
    long enough to be a real pause ends one.
    """
    duration = len(audio) / SAMPLE_RATE
    energy = frame_rms(audio, win)
    if energy.size == 0:
        return [(0.0, duration)] if duration > 0 else []

    loud = energy >= floor
    gap_frames = max(1, int(round(min_silent_run / win)))

    out: list[list[float]] = []
    start: int | None = None
    quiet = 0
    for index, is_loud in enumerate(loud):
        if is_loud:
            if start is None:
                # Begin one frame early: the onset of a word usually sits inside
                # the frame before the one that first crosses the floor.
                start = max(0, index - 1)
            quiet = 0
        elif start is not None:
            quiet += 1
            if quiet >= gap_frames:
                out.append([start * win, (index - quiet + 1) * win])
                start, quiet = None, 0
    if start is not None:
        out.append([start * win, duration])

    return [
        (max(0.0, a), min(duration, b))
        for a, b in out
        if min(duration, b) - max(0.0, a) > 0.2
    ]


def covered_seconds(regions) -> float:
    """Total duration spanned by a list of (start, end) regions."""
    return float(sum(max(0.0, end - start) for start, end in regions))


def energy_pauses(
    audio: np.ndarray,
    floor: float,
    min_pause: float = 0.3,
    win: float = 0.05,
) -> list[float]:
    """Midpoints of the quiet dips between utterances.

    Chunk seams belong in these dips: cutting inside a word corrupts the text on
    both sides of the boundary. VAD turn edges are the obvious source of
    candidates, but a VAD that missed a passage cannot nominate pauses inside
    it -- precisely where seams then land blind. Energy dips are available
    everywhere, including in the speech the VAD rejected.

    The window is deliberately shorter than the one used for coverage: a gap
    between sentences is a few hundred milliseconds, which a 250ms frame cannot
    resolve.
    """
    energy = frame_rms(audio, win)
    if energy.size == 0:
        return []

    quiet = energy < floor
    gap_frames = max(1, int(round(min_pause / win)))

    pauses: list[float] = []
    run_start: int | None = None
    for index, is_quiet in enumerate(quiet):
        if is_quiet:
            if run_start is None:
                run_start = index
        else:
            if run_start is not None and index - run_start >= gap_frames:
                pauses.append(round((run_start + index) / 2 * win, 3))
            run_start = None
    if run_start is not None and len(quiet) - run_start >= gap_frames:
        pauses.append(round((run_start + len(quiet)) / 2 * win, 3))
    return pauses


# A unit of up to four characters repeated ten or more times in a row.
_CHARACTER_RUN = re.compile(r"(.{1,4}?)\1{9,}", re.DOTALL)


def is_repetition_loop(
    text: str,
    min_words: int = 8,
    dominance: float = 0.5,
    min_chars: int = 40,
) -> bool:
    """True if one token dominates the text, the signature of a decoder loop.

    Fed music or noise, the decoder can emit a single word hundreds of times.
    Whisper's own compression-ratio check would catch this, but it only decides
    whether to retry at a hotter temperature -- and it is skipped entirely when
    `no_speech_prob` is high, which is precisely when loops occur.

    A loop can also run inside one word: a stretched shout decodes as "Doctooo"
    followed by hundreds of "o", and scripts without spaces loop as one long run
    of a single character. Neither has words to count, so runs of a short
    repeated unit are measured by the characters they cover instead.

    Real dialogue does repeat ("no, no, no", "Noooooo!"), so the bar is
    deliberately high: a majority of a long utterance being one token, or one
    repeated run, is not something speech does.
    """
    lowered = text.lower()
    words = re.findall(r"[^\W\d_]+", lowered)
    if len(words) >= min_words:
        _, count = Counter(words).most_common(1)[0]
        if count / len(words) > dominance:
            return True

    compact = re.sub(r"\s+", "", lowered)
    if len(compact) < min_chars:
        return False
    looped = sum(len(match.group(0)) for match in _CHARACTER_RUN.finditer(compact))
    return looped / len(compact) > dominance

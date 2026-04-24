import json
import pathlib
import re
from typing import Callable, Optional, TextIO


LANGUAGES_WITHOUT_SPACES = {"ja", "zh"}


def format_timestamp(seconds: float, always_include_hours: bool = False, decimal_marker: str = ".") -> str:
    seconds = max(0.0, seconds)
    milliseconds = round(seconds * 1000.0)
    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000
    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000
    seconds_int = milliseconds // 1_000
    milliseconds -= seconds_int * 1_000
    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds_int:02d}{decimal_marker}{milliseconds:03d}"


class ResultWriter:
    extension: str

    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def __call__(self, result: dict, output_name: str, options: Optional[dict] = None):
        output_path = (pathlib.Path(self.output_dir) / output_name).with_suffix(f".{self.extension}")
        with output_path.open("w", encoding="utf-8") as file:
            self.write_result(result, file=file, options=options or {})

    def write_result(self, result: dict, file: TextIO, options: dict):
        raise NotImplementedError


class WriteTXT(ResultWriter):
    extension = "txt"

    def write_result(self, result: dict, file: TextIO, options: dict):
        for segment in result.get("segments", []):
            text = segment.get("text", "").strip()
            speaker = segment.get("speaker")
            if speaker:
                text = f"[{speaker}]: {text}"
            print(text, file=file, flush=True)


class WriteJSON(ResultWriter):
    extension = "json"

    def write_result(self, result: dict, file: TextIO, options: dict):
        json.dump(result, file, ensure_ascii=False)


class WriteAUD(WriteJSON):
    extension = "aud"


class WriteTSV(ResultWriter):
    extension = "tsv"

    def write_result(self, result: dict, file: TextIO, options: dict):
        print("start", "end", "speaker", "text", sep="\t", file=file)
        for segment in result.get("segments", []):
            print(round(1000 * segment.get("start", 0.0)), file=file, end="\t")
            print(round(1000 * segment.get("end", 0.0)), file=file, end="\t")
            print(segment.get("speaker", ""), file=file, end="\t")
            print(segment.get("text", "").strip().replace("\t", " "), file=file, flush=True)


class SubtitlesWriter(ResultWriter):
    always_include_hours: bool
    decimal_marker: str

    def format_timestamp(self, seconds: float) -> str:
        return format_timestamp(seconds, self.always_include_hours, self.decimal_marker)

    def _segment_text(self, segment: dict) -> str:
        text = segment.get("text", "").strip().replace("-->", "->")
        speaker = segment.get("speaker")
        return f"[{speaker}]: {text}" if speaker else text

    @staticmethod
    def _join_words(words: list[str], language: Optional[str]) -> str:
        if language in LANGUAGES_WITHOUT_SPACES:
            return "".join(words)
        return " ".join(words).replace(" \n", "\n").replace("\n ", "\n")

    def iterate_result(self, result: dict, options: dict):
        segments = result.get("segments", [])
        if not segments:
            return

        raw_max_line_width: Optional[int] = options.get("max_line_width")
        max_line_count: Optional[int] = options.get("max_line_count")
        max_words_per_line: Optional[int] = options.get("max_words_per_line")
        highlight_words = options.get("highlight_words", False)
        max_line_width = 1000 if raw_max_line_width is None else raw_max_line_width
        preserve_segments = max_line_count is None or raw_max_line_width is None
        language = result.get("language")

        def iterate_subtitles():
            line_len = 0
            line_count = 1
            word_count = 0
            subtitle: list[dict] = []
            times: list[tuple[float, float, Optional[str]]] = []
            last = float(segments[0].get("start", 0.0))

            for segment in segments:
                segment_words = [word for word in segment.get("words", []) if isinstance(word, dict)]
                for idx, original_timing in enumerate(segment_words):
                    timing = original_timing.copy()
                    word_text = str(timing.get("word", ""))
                    if not word_text.strip():
                        continue

                    long_pause = not preserve_segments
                    if "start" in timing:
                        long_pause = long_pause and float(timing["start"]) - last > 3.0
                    else:
                        long_pause = False

                    has_room = line_len + len(word_text) <= max_line_width
                    seg_break = idx == 0 and len(subtitle) > 0 and preserve_segments
                    word_break = max_words_per_line is not None and word_count >= max_words_per_line
                    if line_len > 0 and has_room and not long_pause and not seg_break and not word_break:
                        line_len += len(word_text)
                    else:
                        timing["word"] = word_text.strip()
                        if (
                            (len(subtitle) > 0 and word_break)
                            or (
                                len(subtitle) > 0
                                and max_line_count is not None
                                and (long_pause or line_count >= max_line_count)
                            )
                            or seg_break
                        ):
                            yield subtitle, times
                            subtitle = []
                            times = []
                            line_count = 1
                            word_count = 0
                        elif line_len > 0:
                            line_count += 1
                            timing["word"] = "\n" + timing["word"]
                        line_len = len(timing["word"].strip())

                    subtitle.append(timing)
                    times.append(
                        (
                            float(segment.get("start", 0.0)),
                            float(segment.get("end", 0.0)),
                            timing.get("speaker") or segment.get("speaker"),
                        )
                    )
                    word_count += 1
                    if "start" in timing:
                        last = float(timing["start"])

            if subtitle:
                yield subtitle, times

        if any(segment.get("words") for segment in segments):
            for subtitle, times in iterate_subtitles():
                speaker = times[0][2]
                prefix = f"[{speaker}]: " if speaker is not None else ""

                word_starts = [float(word["start"]) for word in subtitle if "start" in word]
                word_ends = [float(word["end"]) for word in subtitle if "end" in word]
                if word_starts and word_ends:
                    subtitle_start = self.format_timestamp(min(word_starts))
                    subtitle_end = self.format_timestamp(max(word_ends))
                else:
                    subtitle_start = self.format_timestamp(times[0][0])
                    subtitle_end = self.format_timestamp(times[-1][1])

                raw_words = [str(word.get("word", "")) for word in subtitle]
                subtitle_text = self._join_words(raw_words, language)
                has_timing = bool(word_starts and word_ends)

                if highlight_words and has_timing:
                    last = subtitle_start
                    for word_idx, word in enumerate(subtitle):
                        if "start" not in word or "end" not in word:
                            continue
                        word_start = self.format_timestamp(float(word["start"]))
                        word_end = self.format_timestamp(float(word["end"]))
                        if last != word_start:
                            yield last, word_start, prefix + subtitle_text
                        highlighted_words = [
                            re.sub(r"^(\s*)(.*)$", r"\1<u>\2</u>", token) if idx == word_idx else token
                            for idx, token in enumerate(raw_words)
                        ]
                        yield word_start, word_end, prefix + self._join_words(highlighted_words, language)
                        last = word_end
                else:
                    yield subtitle_start, subtitle_end, prefix + subtitle_text
            return

        for segment in segments:
            yield (
                self.format_timestamp(segment.get("start", 0.0)),
                self.format_timestamp(segment.get("end", 0.0)),
                self._segment_text(segment),
            )


class WriteVTT(SubtitlesWriter):
    extension = "vtt"
    always_include_hours = False
    decimal_marker = "."

    def write_result(self, result: dict, file: TextIO, options: dict):
        print("WEBVTT\n", file=file)
        for start, end, text in self.iterate_result(result, options):
            print(f"{start} --> {end}\n{text}\n", file=file, flush=True)


class WriteSRT(SubtitlesWriter):
    extension = "srt"
    always_include_hours = True
    decimal_marker = ","

    def write_result(self, result: dict, file: TextIO, options: dict):
        for idx, (start, end, text) in enumerate(self.iterate_result(result, options), start=1):
            print(f"{idx}\n{start} --> {end}\n{text}\n", file=file, flush=True)


def get_writer(output_format: str, output_dir: str) -> Callable[[dict, str, Optional[dict]], None]:
    writers = {
        "txt": WriteTXT,
        "vtt": WriteVTT,
        "srt": WriteSRT,
        "tsv": WriteTSV,
        "json": WriteJSON,
        "aud": WriteAUD,
    }
    if output_format == "all":
        selected = [writer(output_dir) for writer in writers.values()]

        def write_all(result: dict, output_name: str, options: Optional[dict] = None):
            seen: set[str] = set()
            for writer in selected:
                if writer.extension in seen:
                    continue
                seen.add(writer.extension)
                writer(result, output_name, options)

        return write_all
    if output_format not in writers:
        raise ValueError(f"Unsupported output format: {output_format}")
    return writers[output_format](output_dir)

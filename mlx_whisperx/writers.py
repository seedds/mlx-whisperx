import json
import pathlib
import re
from typing import Callable, Optional, TextIO


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

    def iterate_result(self, result: dict, options: dict):
        highlight_words = options.get("highlight_words", False)
        max_words_per_line = options.get("max_words_per_line") or 1000

        for segment in result.get("segments", []):
            words = [w for w in segment.get("words", []) if "start" in w and "end" in w]
            if not words:
                yield (
                    self.format_timestamp(segment.get("start", 0.0)),
                    self.format_timestamp(segment.get("end", 0.0)),
                    self._segment_text(segment),
                )
                continue

            for idx in range(0, len(words), max_words_per_line):
                subtitle_words = words[idx : idx + max_words_per_line]
                speaker = subtitle_words[0].get("speaker") or segment.get("speaker")
                prefix = f"[{speaker}]: " if speaker else ""
                subtitle_text = prefix + "".join(w.get("word", "") for w in subtitle_words).strip()
                start = self.format_timestamp(subtitle_words[0]["start"])
                end = self.format_timestamp(subtitle_words[-1]["end"])

                if not highlight_words:
                    yield start, end, subtitle_text
                    continue

                last = start
                raw_words = [w.get("word", "") for w in subtitle_words]
                for word_idx, word in enumerate(subtitle_words):
                    word_start = self.format_timestamp(word["start"])
                    word_end = self.format_timestamp(word["end"])
                    if last != word_start:
                        yield last, word_start, subtitle_text
                    highlighted = prefix + "".join(
                        re.sub(r"^(\s*)(.*)$", r"\1<u>\2</u>", token) if i == word_idx else token
                        for i, token in enumerate(raw_words)
                    ).strip()
                    yield word_start, word_end, highlighted
                    last = word_end


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

import argparse
import json
import re
import statistics
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


EXPECTED_TOP_LEVEL_KEYS = ["segments", "word_segments", "language"]
EXPECTED_SEGMENT_KEYS = ["start", "end", "text", "words"]
EXPECTED_WORD_KEYS = ["word", "start", "end", "score"]


def _load_json(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return data


def _tokens(result: dict[str, Any]) -> list[str]:
    text = " ".join(str(segment.get("text", "")) for segment in result.get("segments", []))
    return re.findall(r"[\w']+", text.lower())


def _word_error_rate(reference: list[str], hypothesis: list[str]) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0

    previous = list(range(len(hypothesis) + 1))
    for ref_index, ref_token in enumerate(reference, start=1):
        current = [ref_index]
        for hyp_index, hyp_token in enumerate(hypothesis, start=1):
            substitution_cost = 0 if ref_token == hyp_token else 1
            current.append(
                min(
                    previous[hyp_index] + 1,
                    current[hyp_index - 1] + 1,
                    previous[hyp_index - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1] / len(reference)


def _key_variants(items: list[dict[str, Any]]) -> list[list[str]]:
    variants: list[list[str]] = []
    for item in items:
        keys = list(item.keys())
        if keys not in variants:
            variants.append(keys)
    return variants


def _word_segments(result: dict[str, Any]) -> list[dict[str, Any]]:
    words = result.get("word_segments")
    if isinstance(words, list):
        return [word for word in words if isinstance(word, dict)]

    nested_words: list[dict[str, Any]] = []
    for segment in result.get("segments", []):
        if not isinstance(segment, dict):
            continue
        nested_words.extend(word for word in segment.get("words", []) if isinstance(word, dict))
    return nested_words


def _normalize_word(word: str) -> str:
    tokens = re.findall(r"[\w']+", word.lower())
    return "".join(tokens)


def _timing_summary(pairs: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    start_deltas: list[float] = []
    end_deltas: list[float] = []

    for reference_word, generated_word in pairs:
        if "start" in reference_word and "start" in generated_word:
            start_deltas.append(abs(float(reference_word["start"]) - float(generated_word["start"])))
        if "end" in reference_word and "end" in generated_word:
            end_deltas.append(abs(float(reference_word["end"]) - float(generated_word["end"])))

    return {
        "compared_words": len(pairs),
        "start_mean_abs_delta": round(statistics.fmean(start_deltas), 4) if start_deltas else None,
        "start_max_abs_delta": round(max(start_deltas), 4) if start_deltas else None,
        "end_mean_abs_delta": round(statistics.fmean(end_deltas), 4) if end_deltas else None,
        "end_max_abs_delta": round(max(end_deltas), 4) if end_deltas else None,
    }


def _positional_timing_drift(reference_words: list[dict[str, Any]], generated_words: list[dict[str, Any]]) -> dict[str, Any]:
    return _timing_summary(list(zip(reference_words, generated_words)))


def _aligned_timing_drift(reference_words: list[dict[str, Any]], generated_words: list[dict[str, Any]]) -> dict[str, Any]:
    reference_tokens = [_normalize_word(str(word.get("word", ""))) for word in reference_words]
    generated_tokens = [_normalize_word(str(word.get("word", ""))) for word in generated_words]
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []

    for tag, reference_start, reference_end, generated_start, generated_end in SequenceMatcher(
        None, reference_tokens, generated_tokens, autojunk=False
    ).get_opcodes():
        if tag != "equal":
            continue
        for offset in range(reference_end - reference_start):
            pairs.append((reference_words[reference_start + offset], generated_words[generated_start + offset]))

    summary = _timing_summary(pairs)
    summary["match_ratio"] = round(len(pairs) / len(reference_words), 4) if reference_words else 1.0
    return summary


def _schema_metrics(reference: dict[str, Any], generated: dict[str, Any]) -> dict[str, Any]:
    reference_segments = [segment for segment in reference.get("segments", []) if isinstance(segment, dict)]
    generated_segments = [segment for segment in generated.get("segments", []) if isinstance(segment, dict)]
    reference_words = _word_segments(reference)
    generated_words = _word_segments(generated)

    generated_top_keys = list(generated.keys())
    generated_segment_variants = _key_variants(generated_segments)
    generated_word_variants = _key_variants(generated_words)

    return {
        "top_level_keys": generated_top_keys,
        "top_level_key_order_match": generated_top_keys[:3] == EXPECTED_TOP_LEVEL_KEYS,
        "required_top_level_keys_present": all(key in generated for key in EXPECTED_TOP_LEVEL_KEYS),
        "segment_key_variants": generated_segment_variants,
        "segment_keys_match_reference": generated_segment_variants == _key_variants(reference_segments),
        "segment_required_keys_present": all(
            all(key in segment for key in EXPECTED_SEGMENT_KEYS) for segment in generated_segments
        ),
        "word_key_variants": generated_word_variants,
        "word_keys_match_reference": generated_word_variants == _key_variants(reference_words),
        "word_required_keys_present": all(
            all(key in word for key in EXPECTED_WORD_KEYS) for word in generated_words
        ),
    }


def compare(reference: dict[str, Any], generated: dict[str, Any]) -> dict[str, Any]:
    reference_tokens = _tokens(reference)
    generated_tokens = _tokens(generated)
    reference_words = _word_segments(reference)
    generated_words = _word_segments(generated)
    schema = _schema_metrics(reference, generated)

    return {
        "schema_ok": all(
            [
                schema["top_level_key_order_match"],
                schema["required_top_level_keys_present"],
                schema["segment_keys_match_reference"],
                schema["segment_required_keys_present"],
                schema["word_keys_match_reference"],
                schema["word_required_keys_present"],
            ]
        ),
        "schema": schema,
        "counts": {
            "reference_segments": len(reference.get("segments", [])),
            "generated_segments": len(generated.get("segments", [])),
            "reference_word_segments": len(reference_words),
            "generated_word_segments": len(generated_words),
            "reference_tokens": len(reference_tokens),
            "generated_tokens": len(generated_tokens),
        },
        "language": {
            "reference": reference.get("language"),
            "generated": generated.get("language"),
            "match": reference.get("language") == generated.get("language"),
        },
        "text": {
            "wer": round(_word_error_rate(reference_tokens, generated_tokens), 4),
            "similarity": round(SequenceMatcher(None, reference_tokens, generated_tokens).ratio(), 4),
        },
        "timing": _aligned_timing_drift(reference_words, generated_words),
        "positional_timing": _positional_timing_drift(reference_words, generated_words),
    }


def _print_report(metrics: dict[str, Any]) -> None:
    print(f"Schema OK: {metrics['schema_ok']}")
    print(f"Language: reference={metrics['language']['reference']} generated={metrics['language']['generated']} match={metrics['language']['match']}")
    print(
        "Segments: "
        f"reference={metrics['counts']['reference_segments']} "
        f"generated={metrics['counts']['generated_segments']}"
    )
    print(
        "Word segments: "
        f"reference={metrics['counts']['reference_word_segments']} "
        f"generated={metrics['counts']['generated_word_segments']}"
    )
    print(f"Text WER: {metrics['text']['wer']}")
    print(f"Text similarity: {metrics['text']['similarity']}")
    print(
        "Timing drift (matched words): "
        f"compared_words={metrics['timing']['compared_words']} "
        f"match_ratio={metrics['timing']['match_ratio']} "
        f"start_mean={metrics['timing']['start_mean_abs_delta']}s "
        f"start_max={metrics['timing']['start_max_abs_delta']}s "
        f"end_mean={metrics['timing']['end_mean_abs_delta']}s "
        f"end_max={metrics['timing']['end_max_abs_delta']}s"
    )
    print(
        "Timing drift (positional): "
        f"compared_words={metrics['positional_timing']['compared_words']} "
        f"start_mean={metrics['positional_timing']['start_mean_abs_delta']}s "
        f"start_max={metrics['positional_timing']['start_max_abs_delta']}s "
        f"end_mean={metrics['positional_timing']['end_mean_abs_delta']}s "
        f"end_max={metrics['positional_timing']['end_max_abs_delta']}s"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare mlx-whisperx JSON output against a reference WhisperX JSON file.")
    parser.add_argument("reference", help="Reference WhisperX JSON path")
    parser.add_argument("generated", help="Generated mlx-whisperx JSON path")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON metrics")
    parser.add_argument("--fail-on-schema", action="store_true", help="Exit with status 1 if schema parity fails")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics = compare(_load_json(args.reference), _load_json(args.generated))
    if args.json:
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
    else:
        _print_report(metrics)
    if args.fail_on_schema and not metrics["schema_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

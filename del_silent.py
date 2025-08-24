from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import os


from typing import List, Optional, Sequence, Tuple
import math


def _compute_silence_threshold(audio: AudioSegment, silence_thresh: Optional[float]) -> float:
    """Return an absolute dBFS silence threshold.

    If ``silence_thresh`` is None, estimate a robust threshold that preserves
    audible tails (e.g., reverb/echo) by combining:
      - a noise-floor percentile + margin
      - a cap relative to peak loudness
      - a hard floor in dBFS
    This avoids cutting obvious decays common in game SFX.
    """
    if silence_thresh is not None:
        return silence_thresh

    # Guard: if the whole clip is silent, use a sane default
    if audio.dBFS == float("-inf"):
        return -55.0

    # Windowed RMS dBFS to estimate noise floor and peak more robustly than
    # using the global average. Keep dependencies minimal (no numpy).
    window_ms = 50  # analysis window
    hop_ms = 25     # hop size

    samples = audio.get_array_of_samples()
    if not samples:
        return max(-60.0, audio.dBFS - 24.0)

    # Convert to mono by averaging interleaved channels for RMS
    channels = audio.channels
    frame_rate = audio.frame_rate
    max_possible = float(audio.max_possible_amplitude)

    samples_per_ms = (frame_rate * channels) / 1000.0
    win = max(1, int(samples_per_ms * window_ms))
    hop = max(1, int(samples_per_ms * hop_ms))

    # Compute RMS in sliding windows
    rms_db_values: List[float] = []
    total_samples = len(samples)
    idx = 0
    while idx + win <= total_samples:
        # Average across channels per frame
        # Note: samples are interleaved by channel already
        # Compute RMS directly on interleaved data which approximates mono RMS
        acc = 0.0
        for s in samples[idx:idx + win]:
            acc += float(s) * float(s)
        mean_sq = acc / win
        if mean_sq <= 0.0 or max_possible <= 0.0:
            db = float("-inf")
        else:
            rms = math.sqrt(mean_sq)
            ratio = rms / max_possible
            # Avoid log of zero
            if ratio <= 0.0:
                db = float("-inf")
            else:
                db = 20.0 * math.log10(ratio)
        rms_db_values.append(db)
        idx += hop

    # Fallback if something went wrong
    finite_vals = [v for v in rms_db_values if math.isfinite(v)]
    if not finite_vals:
        # Use a conservative threshold relative to average loudness
        return max(-60.0, audio.dBFS - 24.0)

    finite_vals.sort()

    def percentile(sorted_vals: List[float], q: float) -> float:
        # q in [0, 1]
        if not sorted_vals:
            return float("-inf")
        if q <= 0:
            return sorted_vals[0]
        if q >= 1:
            return sorted_vals[-1]
        pos = (len(sorted_vals) - 1) * q
        low = int(math.floor(pos))
        high = int(math.ceil(pos))
        if low == high:
            return sorted_vals[low]
        frac = pos - low
        return sorted_vals[low] * (1 - frac) + sorted_vals[high] * frac

    noise_floor_db = percentile(finite_vals, 0.10)  # 10th percentile
    peak_db = finite_vals[-1]

    # Combine heuristics:
    # - noise floor + margin keeps audible tails
    # - cap relative to peak avoids overly high thresholds on quiet audio
    # - hard floor ensures we don't go unrealistically low
    margin_db = 6.0
    peak_drop_db = 50.0
    hard_floor_db = -60.0

    candidate_noise = noise_floor_db + margin_db
    candidate_peak_rel = peak_db - peak_drop_db
    threshold = max(candidate_noise, candidate_peak_rel, hard_floor_db)

    return float(threshold)


def _merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Merge overlapping or contiguous intervals given as (start_ms, end_ms)."""
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged: List[Tuple[int, int]] = []
    cur_start, cur_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= cur_end:
            if end > cur_end:
                cur_end = end
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    merged.append((cur_start, cur_end))
    return merged


def trim_silence_file(
    input_path: str,
    output_path: Optional[str] = None,
    *,
    min_silence_len: int = 150,
    silence_thresh: Optional[float] = None,
    keep_silence: int = 0,
    remove_interior: bool = False,
    trim_leading: bool = True,
    trim_trailing: bool = True,
) -> str:
    """Trim silence from an audio file and write the result.

    By default, trims only leading/trailing silence which is ideal for short clips
    with obvious trailing ends. If ``remove_interior`` is True, all silent parts
    are removed, concatenating non-silent chunks.

    Args:
        input_path: Path to the input audio file.
        output_path: Where to save the trimmed file. If None, replaces input in-place.
        min_silence_len: Minimum silence length in ms to consider as silence.
        silence_thresh: Silence threshold in dBFS. If None, computed relative to clip loudness.
        keep_silence: Milliseconds of context to keep around non-silent audio.
        remove_interior: If True, delete silence anywhere (not just edges).
        trim_leading: If False, keep leading portion even if silent.
        trim_trailing: If False, keep trailing portion even if silent.

    Returns:
        Path to the written file (same as input if in-place).
    """
    audio = AudioSegment.from_file(input_path)
    threshold = _compute_silence_threshold(audio, silence_thresh)

    # Find all non-silent regions
    non_silent: List[Tuple[int, int]] = detect_nonsilent(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=threshold,
        seek_step=1,
    )

    if not non_silent:
        # Nothing detected as non-silent; keep original to avoid empty outputs
        trimmed = audio
    else:
        if remove_interior:
            # Expand each non-silent chunk by keep_silence on both sides then merge
            expanded: List[Tuple[int, int]] = []
            for start, end in non_silent:
                start_expanded = max(0, start - keep_silence)
                end_expanded = min(len(audio), end + keep_silence)
                expanded.append((start_expanded, end_expanded))
            chunks = _merge_intervals(expanded)
            trimmed = AudioSegment.empty()
            for start, end in chunks:
                trimmed += audio[start:end]
        else:
            # Trim only the edges
            start, end = non_silent[0][0], non_silent[-1][1]
            if trim_leading:
                start = max(0, start - keep_silence)
            else:
                start = 0
            if trim_trailing:
                end = min(len(audio), end + keep_silence)
            else:
                end = len(audio)
            trimmed = audio[start:end]

    # Determine export format from output path or input extension
    def _export(audio_seg: AudioSegment, dest_path: str) -> None:
        ext = os.path.splitext(dest_path)[1].lower().lstrip(".")
        if not ext:
            # Default to wav if no extension provided
            dest_path = dest_path + ".wav"
            ext = "wav"
        audio_seg.export(dest_path, format=ext)

    if output_path is None:
        # In-place: export to a temp file next to input then atomically replace
        directory, filename = os.path.split(input_path)
        name, ext = os.path.splitext(filename)
        temp_path = os.path.join(directory or ".", f"{name}.__tmp__{ext}")
        _export(trimmed, temp_path)
        os.replace(temp_path, input_path)
        return input_path
    else:
        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        _export(trimmed, output_path)
        return output_path


def trim_directory_inplace(
    root_dir: str,
    *,
    extensions: Sequence[str] = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"),
    min_silence_len: int = 150,
    silence_thresh: Optional[float] = None,
    keep_silence: int = 0,
    remove_interior: bool = False,
    trim_leading: bool = True,
    trim_trailing: bool = True,
    show_progress: bool = True,
) -> None:
    """Recursively trim silence from all audio files under ``root_dir`` in place.

    Args:
        root_dir: Directory to process.
        extensions: File extensions to include (lowercased, with dots).
        min_silence_len: Minimum silence length in ms to consider as silence.
        silence_thresh: Silence threshold in dBFS. If None, computed relative to clip loudness.
        keep_silence: Milliseconds of context to keep around non-silent audio.
        remove_interior: If True, delete silence anywhere (not just edges).
        trim_leading: If False, keep leading portion even if silent.
        trim_trailing: If False, keep trailing portion even if silent.
        show_progress: If True, prints progress to stdout.
    """
    exts = tuple(e.lower() for e in extensions)
    processed = 0
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            if os.path.splitext(fname)[1].lower() in exts:
                fpath = os.path.join(dirpath, fname)
                try:
                    trim_silence_file(
                        fpath,
                        output_path=None,
                        min_silence_len=min_silence_len,
                        silence_thresh=silence_thresh,
                        keep_silence=keep_silence,
                        remove_interior=remove_interior,
                        trim_leading=trim_leading,
                        trim_trailing=trim_trailing,
                    )
                    processed += 1
                    if show_progress:
                        print(f"Trimmed: {fpath}")
                except Exception as exc:
                    if show_progress:
                        print(f"Failed: {fpath} -> {exc}")
    if show_progress:
        print(f"Done. Processed {processed} files under '{root_dir}'.")




if __name__ == "__main__":
    # trim_silence_file(input_path="cs2 sounds/grenade/flashbang/flashbang_explode1.wav" , min_silence_len=1, trim_leading=False , trim_trailing=True , keep_silence=0 , silence_thresh=-25 , output_path="outputs")    
    trim_directory_inplace(root_dir="cs2 sounds" , extensions=(".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac") , min_silence_len=1 , silence_thresh=-25 , keep_silence=0 , remove_interior=False , trim_leading=False , trim_trailing=True , show_progress=True)
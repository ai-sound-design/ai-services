"""Replace a generated sound with pieces of library recordings.

The generated sound is the template. First it is searched as a whole in the
search service's /search/by_audio, which answers with library windows that
sound alike (file, second, score). Then it is split, tentatively, where its
spectrum changes most (spectral flux), and both halves are searched; the split
stays only when the length-weighted similarity improves by `split_gain`. That
repeats on the piece that gains most until `pieces_per_10s` (an upper bound)
is reached or splitting stops paying off, and no piece is shorter than
`min_piece_seconds`. A recording that fits the whole sound is therefore kept
whole; only the parts that do not fit are cut finer and searched again. A small
dynamic programme then picks one window per piece so that the sequence prefers
to stay in the same recording (a switch costs `switch_penalty`). Pieces under
`min_similarity` are dropped. Further `layers` add, per piece, the next-best
different recording, but only where it still reaches `min_similarity`, to be
stacked underneath.

Nothing here touches the model; the search service holds CLAP and the index.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any, Optional

import httpx
import numpy as np

log = logging.getLogger("hybrid.matching")

SAMPLE_RATE = 48000


# ── the generated sound ──────────────────────────────────────────────────────

def decode(path: Path) -> np.ndarray:
    done = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-f", "f32le", "-ac", "1",
                           "-ar", str(SAMPLE_RATE), "-"], check=True, capture_output=True)
    return np.frombuffer(done.stdout, dtype=np.float32)


def spectral_flux(samples: np.ndarray, frame: int = 2048, hop: int = 512) -> tuple[np.ndarray, float]:
    """Onset-like novelty per frame (half-wave rectified log-spectral difference), and the
    frame rate in frames per second."""
    if len(samples) < frame * 2:
        return np.zeros(1), SAMPLE_RATE / hop
    window = np.hanning(frame).astype(np.float32)
    count = 1 + (len(samples) - frame) // hop
    frames = np.lib.stride_tricks.as_strided(
        samples, shape=(count, frame), strides=(samples.strides[0] * hop, samples.strides[0]))
    spectra = np.abs(np.fft.rfft(frames * window, axis=1))
    logspec = np.log1p(spectra * 10.0)
    diff = np.diff(logspec, axis=0)
    flux = np.maximum(diff, 0.0).sum(axis=1)
    flux = np.concatenate([[0.0], flux])
    # Smooth over ~50 ms so a single noisy frame does not become a cut.
    kernel = np.ones(5) / 5.0
    return np.convolve(flux, kernel, mode="same"), SAMPLE_RATE / hop


def boundary_candidates(samples: np.ndarray, min_piece: float) -> list[float]:
    """Every plausible cut, strongest novelty first, at least `min_piece` from the ends
    and from each other."""
    duration = len(samples) / SAMPLE_RATE
    flux, rate = spectral_flux(samples)
    chosen: list[float] = []
    for index in np.argsort(flux)[::-1]:
        t = index / rate
        if flux[index] <= 0:
            break
        if t < min_piece or duration - t < min_piece or any(abs(t - c) < min_piece for c in chosen):
            continue
        chosen.append(round(float(t), 3))
    return chosen


def cut_points(samples: np.ndarray, pieces_per_10s: int, min_piece: float) -> list[float]:
    """Piece boundaries in seconds (excluding 0 and the end)."""
    duration = len(samples) / SAMPLE_RATE
    max_pieces = max(1, int(duration / 10.0 * pieces_per_10s + 0.5))
    max_pieces = min(max_pieces, int(duration // min_piece) if min_piece > 0 else max_pieces)
    if max_pieces <= 1 or duration < 2 * min_piece:
        return []
    flux, rate = spectral_flux(samples)
    # Candidate boundaries: local maxima of the novelty, strongest first.
    order = np.argsort(flux)[::-1]
    chosen: list[float] = []
    for index in order:
        t = index / rate
        if t < min_piece or duration - t < min_piece:
            continue
        if any(abs(t - c) < min_piece for c in chosen):
            continue
        if flux[index] <= 0:
            break
        chosen.append(round(float(t), 3))
        if len(chosen) >= max_pieces - 1:
            break
    return sorted(chosen)


def pieces_of(samples: np.ndarray, pieces_per_10s: int, min_piece: float) -> list[tuple[float, float]]:
    """[(start, length)] covering the whole sound."""
    duration = len(samples) / SAMPLE_RATE
    bounds = [0.0] + cut_points(samples, pieces_per_10s, min_piece) + [duration]
    return [(bounds[i], round(bounds[i + 1] - bounds[i], 3)) for i in range(len(bounds) - 1)]


def write_piece(samples: np.ndarray, start: float, length: float, dst: Path) -> None:
    piece = samples[int(start * SAMPLE_RATE): int((start + length) * SAMPLE_RATE)]
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-nostdin", "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE),
                    "-i", "-", "-c:a", "pcm_s16le", str(dst)], input=piece.tobytes(), check=True, capture_output=True)


# ── the search ───────────────────────────────────────────────────────────────

async def search_by_audio(client: httpx.AsyncClient, search_url: str, piece: Path, *, text: str,
                          text_weight: float, library: Optional[str], category: Optional[str],
                          limit: int) -> list[dict]:
    data = {"text": text or "", "text_weight": str(text_weight), "limit": str(limit), "refine": "true"}
    if library:
        data["library"] = library
    if category:
        data["category"] = category
    with piece.open("rb") as handle:
        response = await client.post(f"{search_url}/search/by_audio",
                                     files={"audio": (piece.name, handle, "audio/wav")}, data=data)
    if response.status_code != 200:
        raise RuntimeError(f"search service: {response.status_code} {response.text[:300]}")
    return response.json().get("results", [])


def choose_sequence(candidates: list[list[dict]], switch_penalty: float) -> list[dict]:
    """One candidate per piece, minimising (1 - similarity) plus a penalty for every
    change of recording. Plain Viterbi over the candidate lists."""
    if not candidates:
        return []
    cost = [[1.0 - float(c["similarity"]) for c in options] for options in candidates]
    back: list[list[int]] = []
    best = cost[0]
    for i in range(1, len(candidates)):
        current, pointers = [], []
        for j, option in enumerate(candidates[i]):
            choices = [best[k] + cost[i][j] + (0.0 if candidates[i - 1][k]["id"] == option["id"] else switch_penalty)
                       for k in range(len(candidates[i - 1]))]
            k = int(np.argmin(choices)) if choices else 0
            current.append(choices[k] if choices else cost[i][j])
            pointers.append(k)
        back.append(pointers)
        best = current
    path = [int(np.argmin(best))] if best else [0]
    for pointers in reversed(back):
        path.append(pointers[path[-1]])
    path.reverse()
    return [candidates[i][j] if candidates[i] else None for i, j in enumerate(path)]


def choose_layers(candidates: list[list[dict]], chosen: list[dict | None], layers: int,
                  min_similarity: float = 0.0) -> list[list[dict | None]]:
    """For layers 2..n: per piece the next-best candidate from a recording not used above
    it, and only where it still reaches `min_similarity`; elsewhere the layer stays empty."""
    stack: list[list[dict | None]] = []
    for layer in range(1, layers):
        row: list[dict | None] = []
        for i, options in enumerate(candidates):
            used = {chosen[i]["id"]} if chosen[i] else set()
            used |= {prev[i]["id"] for prev in stack if prev[i]}
            pick = next((c for c in options if c["id"] not in used and float(c["similarity"]) >= min_similarity), None)
            row.append(pick)
        stack.append(row)
    return stack


async def fetch_snippet(client: httpx.AsyncClient, search_url: str, sound_id: int, start: float,
                        length: float, dst: Path, channels: int = 2) -> None:
    # One channel count for every piece: a mono and a stereo clip cannot share a track.
    response = await client.get(f"{search_url}/sounds/{sound_id}/snippet",
                                params={"start": f"{start:.3f}", "length": f"{length:.3f}", "channels": str(channels)})
    if response.status_code != 200:
        raise RuntimeError(f"search service snippet: {response.status_code} {response.text[:200]}")
    dst.write_bytes(response.content)


async def match_generated(generated: Path, work: Path, *, search_url: str, text: str, settings: dict[str, Any],
                          category: Optional[str], on_progress=None) -> list[dict[str, Any]]:
    """Pieces of library recordings that replace `generated`. Each piece:
    {layer, start_seconds, length_seconds, handle_before_seconds, handle_after_seconds,
     sound_id, library, external_id, description, offset_seconds, similarity, path}.
    `start_seconds` is relative to the generated sound; the file holds the handles too,
    so it starts `handle_before_seconds` before that."""
    samples = decode(generated)
    duration = len(samples) / SAMPLE_RATE
    min_piece = float(settings.get("min_piece_seconds", 2.0))
    max_pieces = max(1, int(duration / 10.0 * int(settings.get("pieces_per_10s", 3)) + 0.5))
    split_gain = float(settings.get("split_gain", 0.03))
    min_similarity = float(settings.get("min_similarity", 0.0))
    layers = max(1, int(settings.get("layers", 1)))
    text_weight = float(settings.get("text_weight", 0.0))
    library = settings.get("library") or None
    limit = max(layers + 2, int(settings.get("candidates", 8)))
    searched: dict[tuple[float, float], list[dict]] = {}
    calls = 0

    async with httpx.AsyncClient(timeout=float(settings.get("timeout_seconds", 120))) as client:

        async def search(start: float, length: float) -> list[dict]:
            nonlocal calls
            key = (round(start, 3), round(length, 3))
            if key not in searched:
                calls += 1
                if on_progress:
                    on_progress(calls, max_pieces * 2)
                piece = work / f"{generated.stem}_piece_{calls}.wav"
                write_piece(samples, start, length, piece)
                try:
                    searched[key] = await search_by_audio(client, search_url, piece, text=text,
                                                          text_weight=text_weight, library=library,
                                                          category=category, limit=limit)
                finally:
                    piece.unlink(missing_ok=True)
            return searched[key]

        def score(found: list[dict]) -> float:
            return float(found[0]["similarity"]) if found else 0.0

        # The whole sound first; then split where it pays, the most rewarding piece
        # first, until the bound is reached or no split improves the match any more.
        pieces: list[tuple[float, float]] = [(0.0, round(duration, 3))]
        await search(0.0, duration)
        cuts = boundary_candidates(samples, min_piece)
        while len(pieces) < max_pieces and cuts:
            best = None                                          # (gain, piece index, cut)
            for i, (start, length) in enumerate(pieces):
                inside = [c for c in cuts if start + min_piece <= c <= start + length - min_piece]
                if not inside:
                    continue
                cut = inside[0]                                  # the strongest change inside this piece
                whole = score(await search(start, length))
                left = score(await search(start, cut - start))
                right = score(await search(cut, start + length - cut))
                gain = (left * (cut - start) + right * (start + length - cut)) / length - whole
                if best is None or gain > best[0]:
                    best = (gain, i, cut)
            if best is None or best[0] < split_gain:
                break
            _, i, cut = best
            start, length = pieces[i]
            pieces[i:i + 1] = [(start, round(cut - start, 3)), (round(cut, 3), round(start + length - cut, 3))]
            cuts.remove(cut)
        log.info("%s: %d piece(s) after %d search(es), bound %d", generated.name, len(pieces), calls, max_pieces)

        candidates = [await search(start, length) for start, length in pieces]
        chosen = choose_sequence(candidates, float(settings.get("switch_penalty", 0.15)))
        chosen = [c if c and float(c["similarity"]) >= min_similarity else None for c in chosen]
        rows = [chosen] + choose_layers(candidates, chosen, layers, min_similarity)

        # Handles: extra seconds of the recording before and after the matched stretch
        # (an ambience needs them for a fade), as far as the recording reaches.
        handle = max(0.0, float(settings.get("handle_seconds", 0.0)))
        # An event longer than the generated sound (the generation model has a maximum
        # length): a recording that matched the sound as a whole is cut in the length of
        # the event instead, as far as it reaches, so a two-minute room tone gets two
        # minutes of the recording and not twelve seconds of it.
        extend = max(0.0, float(settings.get("extend_to_seconds", 0.0)))

        out: list[dict[str, Any]] = []
        for layer, row in enumerate(rows, start=1):
            for (start, length), pick in zip(pieces, row):
                if pick is None:
                    continue
                offset = float(pick["offset_seconds"])
                total = float(pick.get("duration_seconds") or 0.0)
                if extend > duration and len(pieces) == 1:
                    length = round(min(extend, total - offset) if total > 0 else extend, 3)
                before = min(handle, offset)
                after = min(handle, max(0.0, total - offset - length)) if total > 0 else handle
                dst = work / f"{generated.stem}_L{layer}_{start:.2f}.wav"
                await fetch_snippet(client, search_url, int(pick["id"]), offset - before, length + before + after, dst,
                                    channels=int(settings.get("channels", 2)))
                out.append({"layer": layer, "start_seconds": round(start, 3), "length_seconds": length,
                            "handle_before_seconds": round(before, 3), "handle_after_seconds": round(after, 3),
                            "sound_id": int(pick["id"]), "library": pick.get("library"),
                            "external_id": pick.get("external_id"), "description": pick.get("description"),
                            "category": pick.get("category"), "offset_seconds": offset,
                            "similarity": float(pick["similarity"]), "path": dst})
    return out

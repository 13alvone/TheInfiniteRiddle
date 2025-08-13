#!/usr/bin/env python3
"""Core utilities for The Infinite Riddle."""
import hashlib
import hmac
import logging
import math
import secrets
import time
from typing import Dict, List, Tuple, Optional, Union


# ----------------------------- Logging Setup -----------------------------
def setup_logging(verbosity: int) -> None:
    level = logging.INFO
    if verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.debug("[DEBUG] Logging initialized with verbosity=%d", verbosity)


# ----------------------------- Entropy & PRNG -----------------------------
def blake2b_digest(data: bytes) -> bytes:
    h = hashlib.blake2b(data, digest_size=32)
    return h.digest()


def hkdf_sha256(ikm: bytes, salt: bytes, info: bytes, length: int) -> bytes:
    prk = hmac.new(salt, ikm, hashlib.sha256).digest()
    okm = b""
    prev = b""
    i = 1
    while len(okm) < length:
        prev = hmac.new(prk, prev + info + bytes([i]), hashlib.sha256).digest()
        okm += prev
        i += 1
    return okm[:length]


class Xoshiro256StarStar:
    def __init__(self, seed32: bytes):
        self.s = [int.from_bytes(seed32[i:i+8], "little") for i in range(0, 32, 8)]

    @staticmethod
    def _rotl(x: int, k: int) -> int:
        return ((x << k) & 0xFFFFFFFFFFFFFFFF) | (x >> (64 - k))

    def next64(self) -> int:
        s0, s1, s2, s3 = self.s
        result = self._rotl(s1 * 5, 7) * 9
        t = s1 << 17
        s2 ^= s0
        s3 ^= s1
        s1 ^= s2
        s0 ^= s3
        s2 ^= t
        s3 = self._rotl(s3, 45)
        self.s = [s0 & 0xFFFFFFFFFFFFFFFF, s1 & 0xFFFFFFFFFFFFFFFF,
                  s2 & 0xFFFFFFFFFFFFFFFF, s3 & 0xFFFFFFFFFFFFFFFF]
        return result & 0xFFFFFFFFFFFFFFFF

    def randbits(self, k: int) -> int:
        return self.next64() >> (64 - k)

    def uniform(self) -> float:
        return self.next64() / 2**64

    def choice(self, seq):
        return seq[int(self.uniform() * len(seq)) % len(seq)]

    def weighted_choice(self, items_weights: List[Tuple[object, float]]):
        total = sum(w for _, w in items_weights)
        r = self.uniform() * total
        upto = 0.0
        for item, weight in items_weights:
            if upto + weight >= r:
                return item
            upto += weight
        return items_weights[-1][0]

    def normal(self, mu: float, sigma: float) -> float:
        u1 = self.uniform()
        u2 = self.uniform()
        z0 = (-2.0 * math.log(max(u1, 1e-12)))**0.5 * math.cos(2*math.pi*u2)
        return z0 * sigma + mu

    def randint(self, a: int, b: int) -> int:
        return a + int(self.uniform() * ((b - a) + 1))


def gather_entropy() -> bytes:
    return secrets.token_bytes(32) + int(time.time_ns()).to_bytes(8, "little")


def domain_prngs(root_seed: bytes) -> Dict[str, Xoshiro256StarStar]:
    labels = [
        b"names",
        b"form",
        b"rhythm",
        b"harmony",
        b"melody",
        b"synth",
        b"mythic",
        b"ctrl",
        b"video",
    ]
    out = {}
    for label in labels:
        seed = hkdf_sha256(root_seed, b"riddlev0", label, 32)
        out[label.decode()] = Xoshiro256StarStar(seed)
    return out


# ----------------------------- Music Theory Helpers -----------------------------
MODES = {
    "lydian":       [0, 2, 4, 6, 7, 9, 11],
    "ionian_sharp11":[0, 2, 4, 6, 7, 9, 11],
    "whole_tone":   [0, 2, 4, 6, 8, 10],
    "messiaen_m3":  [0, 2, 3, 5, 6, 8, 9, 11],
    "phrygian":     [0, 1, 3, 5, 7, 8, 10],
    "locrian":      [0, 1, 3, 5, 6, 8, 10],
    "hijaz":        [0, 1, 4, 5, 7, 8, 11],
    "pelog5":       [0, 1, 3, 7, 8],
    "dorian_b2":    [0, 1, 3, 5, 7, 9, 10],
}

NOTE_NAMES_SHARP = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

# Mapping from pitch class (0=C … 11=B) to MIDI key signature "sf" values.
# The "sf" parameter represents the number of sharps (positive) or flats
# (negative) for major keys in the circle of fifths. Values range from -7 to 7.
PC_TO_SF = {
    0: 0,   # C major
    1: -5,  # Db major
    2: 2,   # D major
    3: -3,  # Eb major
    4: 4,   # E major
    5: -1,  # F major
    6: 6,   # F# major
    7: 1,   # G major
    8: -4,  # Ab major
    9: 3,   # A major
    10: -2, # Bb major
    11: 5,  # B major
}


def pitch_class_to_sf(pc: int) -> int:
    """Return MIDI key signature "sf" for a given pitch class."""
    return PC_TO_SF[pc % 12]


def pick_theme(prng_names: Xoshiro256StarStar, requested: Optional[str]) -> str:
    if requested and requested.lower() in ("glass", "salt"):
        return requested.lower()
    return prng_names.choice(["glass", "salt"])


def pick_key(prng: Xoshiro256StarStar, theme: str) -> Tuple[int, str]:
    if theme == "glass":
        roots = [(2, 0.34), (6, 0.33), (9, 0.33)]
        modes = [("lydian",0.38), ("whole_tone",0.22), ("messiaen_m3",0.18),
                 ("ionian_sharp11",0.12), ("lydian",0.10)]
    else:
        roots = [(0, 0.34), (5, 0.33), (8, 0.33)]
        modes = [("phrygian",0.33), ("locrian",0.22), ("hijaz",0.20),
                 ("pelog5",0.15), ("dorian_b2",0.10)]
    root = prng.weighted_choice(roots)
    mode = prng.weighted_choice(modes)
    return root, mode


def scale_pitches(root_pc: int, mode_name: str) -> List[int]:
    scale = MODES[mode_name]
    return [(root_pc + x) % 12 for x in scale]


def midi_note_in_scale(prng: Xoshiro256StarStar, scale_pcs: List[int], min_note=36, max_note=84) -> int:
    while True:
        note = prng.randint(min_note, max_note)
        if (note % 12) in scale_pcs:
            return note


# ----------------------------- Length Engine -----------------------------
def sample_duration_seconds(
    prng_form: Xoshiro256StarStar, theme: str, override: Optional[str]
) -> Tuple[int, str]:
    bucket_p = {"short": 0.70, "med": 0.25, "long": 0.05}
    if override in bucket_p:
        bucket = override
    else:
        r = prng_form.uniform()
        if r < bucket_p["short"]:
            bucket = "short"
        elif r < bucket_p["short"] + bucket_p["med"]:
            bucket = "med"
        else:
            bucket = "long"
        if theme == "glass" and bucket != "long":
            if prng_form.uniform() < 0.20:
                bucket = "med" if bucket == "short" else "long"
        elif theme == "salt" and bucket != "short":
            if prng_form.uniform() < 0.20:
                bucket = "short" if bucket == "med" else "med"

    mu_sigma = {
        "short": (math.log(60.0), 0.45),
        "med": (math.log(600.0), 0.50),
        "long": (math.log(5400.0), 0.60),
    }
    mu, sigma = mu_sigma[bucket]
    secs = int(max(20.0, min(3 * 3600.0, math.exp(prng_form.normal(mu, sigma)))))
    return secs, bucket


# ----------------------------- Form Generation -----------------------------
FORM_PATHS: Dict[str, List[Tuple[List[str], float]]] = {
    "glass": [
        (["INV", "PRC", "TRN", "PRC", "SHR"], 0.32),
        (["INV", "PRC", "BRK", "TRN", "PRC", "SHR"], 0.28),
        (["INV", "CLM", "PRC", "TRN", "SHR"], 0.18),
        (["INV", "PRC", "CLM", "SHR"], 0.12),
        (["INV", "TRN", "PRC", "TRN", "SHR"], 0.10),
    ],
    "salt": [
        (["INV", "PRC", "CLM", "SHR"], 0.34),
        (["INV", "CLM", "PRC", "CLM", "SHR"], 0.26),
        (["INV", "PRC", "BRK", "CLM", "SHR"], 0.22),
        (["INV", "CLM", "TRN", "CLM", "SHR"], 0.18),
    ],
}

SECTION_PCT_RANGES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "glass": {
        "INV": (12.0, 20.0),
        "PRC": (20.0, 35.0),
        "BRK": (8.0, 15.0),
        "TRN": (10.0, 18.0),
        "CLM": (10.0, 25.0),
        "SHR": (8.0, 15.0),
    },
    "salt": {
        "INV": (10.0, 18.0),
        "PRC": (18.0, 30.0),
        "BRK": (6.0, 12.0),
        "TRN": (8.0, 15.0),
        "CLM": (22.0, 40.0),
        "SHR": (8.0, 15.0),
    },
}


def pick_form(
    prng_form: Xoshiro256StarStar, theme: str, total_sec: int
) -> Tuple[List[str], List[Tuple[str, float, float]], List[float]]:
    """Select a form path and allocate section durations.

    Returns the list of nodes, timeline with absolute seconds, and
    corresponding percentage allocations per section.
    """
    path = prng_form.weighted_choice(FORM_PATHS[theme])
    pct_ranges = SECTION_PCT_RANGES[theme]

    percentages: List[float] = []
    remaining = 100.0
    for idx, node in enumerate(path):
        min_pct, max_pct = pct_ranges[node]
        remaining_nodes = path[idx + 1 :]
        min_remaining = sum(pct_ranges[n][0] for n in remaining_nodes)
        max_remaining = sum(pct_ranges[n][1] for n in remaining_nodes)
        if idx == len(path) - 1:
            pct = remaining
        else:
            lower = max(min_pct, remaining - max_remaining)
            upper = min(max_pct, remaining - min_remaining)
            if lower > upper:
                lower, upper = min_pct, max_pct
            pct = prng_form.uniform() * (upper - lower) + lower
        percentages.append(pct)
        remaining -= pct

    timeline: List[Tuple[str, float, float]] = []
    t = 0.0
    for node, pct in zip(path, percentages):
        dur = total_sec * (pct / 100.0)
        timeline.append((node, t, t + dur))
        t += dur
    if timeline:
        node, start, _ = timeline[-1]
        timeline[-1] = (node, start, float(total_sec))
    return path, timeline, percentages


# ----------------------------- Euclidean Rhythm -----------------------------
def euclidean_rhythm(k: int, n: int) -> List[int]:
    if k <= 0:
        return [0]*n
    if k >= n:
        return [1]*n
    pattern = []
    counts = []
    remainders = []
    divisor = n - k
    remainders.append(k)
    level = 0
    while True:
        counts.append(divisor // remainders[level])
        remainders.append(divisor % remainders[level])
        divisor = remainders[level]
        level += 1
        if remainders[level] <= 1:
            break
    counts.append(divisor)

    def build(level):
        if level == -1:
            return [0]
        if level == -2:
            return [1]
        seq = []
        for _ in range(counts[level]):
            seq += build(level-1)
        if remainders[level] != 0:
            seq += build(level-2)
        return seq

    pattern = build(level)
    while pattern[0] != 1:
        pattern = pattern[1:] + pattern[:1]
    return pattern[:n]


# ----------------------------- CA Gate -----------------------------
def ca_gate(rule: int, n: int, seed_bits: int) -> List[int]:
    row = [(seed_bits >> i) & 1 for i in range(32)]
    row = row + [0] * max(0, n - len(row))
    row = row[:n]
    out = []
    for i in range(n):
        l = row[i-1] if i-1 >= 0 else row[-1]
        c = row[i]
        r = row[(i+1) % n]
        idx = (l<<2) | (c<<1) | r
        bit = (rule >> idx) & 1
        out.append(bit)
    return out


# ----------------------------- Composition Helpers -----------------------------
# Theme-specific time signature tables and shift probabilities
TIME_SIGNATURE_TABLES: Dict[str, List[Tuple[Tuple[int, int], float]]] = {
    "glass": [
        ((4, 4), 0.35),
        ((5, 4), 0.22),
        ((7, 8), 0.18),
        ((9, 8), 0.12),
        ((11, 8), 0.08),
        ((13, 8), 0.05),
    ],
    "salt": [
        ((7, 8), 0.24),
        ((5, 4), 0.22),
        ((3, 4), 0.18),
        # Additive (2+2+3)/8 grouped as another 7/8 weight
        ((7, 8), 0.18),
        ((4, 4), 0.18),
    ],
}

TIME_SIGNATURE_SHIFT: Dict[str, float] = {"glass": 0.15, "salt": 0.20}


def pick_time_signature(
    prng: Xoshiro256StarStar,
    theme: str,
    chaotic: bool = False,
    count: int = 4,
) -> Union[Tuple[int, int], List[Tuple[int, int]]]:
    """Pick a time signature for the given theme.

    When ``chaotic`` is False, return a single weighted choice from the
    theme's table. When True, return a sequence ``count`` bars long where each
    subsequent bar may re-roll the meter with probability ``p_shift``.
    """

    table = TIME_SIGNATURE_TABLES[theme]
    if not chaotic:
        return prng.weighted_choice(table)

    p_shift = TIME_SIGNATURE_SHIFT[theme]
    seq: List[Tuple[int, int]] = []
    current = prng.weighted_choice(table)
    for _ in range(max(1, count)):
        seq.append(current)
        if prng.uniform() < p_shift:
            current = prng.weighted_choice(table)
    return seq


def build_melody(
    prng_mel: Xoshiro256StarStar,
    scale_pcs: List[int],
    sections: List[Tuple[int, int, int]],
    ppq: int,
    density: float,
    base_octave: int,
    sigil_pcs: List[int],
) -> List[Tuple[int, int, int, int]]:
    events: List[Tuple[int, int, int, int]] = []
    tick_offset = 0
    for bars, ts_num, ts_den in sections:
        ticks_per_bar = ppq * ts_num * 4 // ts_den
        beats_per_bar = ts_num * 4 // ts_den
        ticks_per_beat = ticks_per_bar // beats_per_bar if beats_per_bar else 0
        for bar in range(bars):
            for beat in range(beats_per_bar):
                if prng_mel.uniform() < density:
                    note = midi_note_in_scale(
                        prng_mel, scale_pcs, base_octave * 12, (base_octave + 2) * 12
                    )
                    start = tick_offset + bar * ticks_per_bar + beat * ticks_per_beat
                    dur = int(ticks_per_beat * prng_mel.uniform() * 0.8) or ticks_per_beat // 2
                    vel = prng_mel.randint(60, 120)
                    events.append((start, dur, note, vel))
        tick_offset += bars * ticks_per_bar
    return events


def build_bass(
    prng: Xoshiro256StarStar,
    scale_pcs: List[int],
    sections: List[Tuple[int, int, int]],
    ppq: int,
) -> List[Tuple[int, int, int, int]]:
    events: List[Tuple[int, int, int, int]] = []
    tick_offset = 0
    for bars, ts_num, ts_den in sections:
        ticks_per_bar = ppq * ts_num * 4 // ts_den
        for bar in range(bars):
            note = midi_note_in_scale(prng, scale_pcs, 36, 60)
            start = tick_offset + bar * ticks_per_bar
            dur = ticks_per_bar
            vel = prng.randint(70, 110)
            events.append((start, dur, note, vel))
        tick_offset += bars * ticks_per_bar
    return events


def build_perc(
    prng: Xoshiro256StarStar,
    sections: List[Tuple[int, int, int]],
    ppq: int,
    theme: str,
) -> List[Tuple[int, int, int, int]]:
    events: List[Tuple[int, int, int, int]] = []
    tick_offset = 0
    for bars, ts_num, ts_den in sections:
        ticks_per_bar = ppq * ts_num * 4 // ts_den
        beats_per_bar = ts_num * 4 // ts_den
        pattern = euclidean_rhythm(beats_per_bar, ticks_per_bar)
        ticks_per_beat = ticks_per_bar // beats_per_bar if beats_per_bar else 0
        for bar in range(bars):
            for i, bit in enumerate(pattern):
                if bit:
                    start = tick_offset + bar * ticks_per_bar + i
                    vel = prng.randint(80, 120)
                    events.append((start, ticks_per_beat // 4 or 1, 0, vel))
        tick_offset += bars * ticks_per_bar
    return events

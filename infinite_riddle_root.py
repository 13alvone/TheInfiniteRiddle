#!/usr/bin/env python3
"""
The Infinite Riddle — v0.1 Core
Implements:
- Entropy → HKDF domain keys → xoshiro256** PRNG streams
- Length engine (heavy-tailed)
- Form grammar (Glass/Refraction, Salt/Sealing)
- Harmony/Rhythm/Melody generators → MIDI (Type-1, multitrack)
- Simple synth rack (FM lead, WT pad, sub bass, noise perc) → streamed WAV (48kHz, 24-bit, stereo)
- Sidecar JSON attestation + SHA-256 hashing
- SQLite Vault schema + inserts
- Mythic variants: Backmask, Ashen (bitcrush), MirrorSalt (M/S microphase), Liminal (drone bed), CipherSpray (ultrasonic watermark)

Notes:
- Standard library only (no external deps). Uses hashlib.blake2b in place of BLAKE3.
- Focus: correctness, determinism-by-seed, and robustness — not maximal DSP complexity yet.
"""
import argparse
import audioop
import datetime as dt
import hashlib
import hmac
import json
import logging
import math
import os
import secrets
import sqlite3
import struct
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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
    h = hashlib.blake2b()
    h.update(data)
    return h.digest()


def hkdf_sha256(ikm: bytes, salt: bytes, info: bytes, length: int) -> bytes:
    """RFC 5869 HKDF-SHA256."""
    prk = hmac.new(salt, ikm, hashlib.sha256).digest()
    t = b""
    okm = b""
    i = 1
    while len(okm) < length:
        t = hmac.new(prk, t + info + bytes([i]), hashlib.sha256).digest()
        okm += t
        i += 1
    return okm[:length]


class Xoshiro256StarStar:
    """xoshiro256** PRNG (64-bit)."""
    __slots__ = ("s0", "s1", "s2", "s3")

    def __init__(self, seed32: bytes):
        if len(seed32) != 32:
            raise ValueError("seed32 must be 32 bytes")
        self.s0, self.s1, self.s2, self.s3 = struct.unpack("<QQQQ", seed32)

        # Avoid all-zero state
        if self.s0 == 0 and self.s1 == 0 and self.s2 == 0 and self.s3 == 0:
            self.s0 = 0x9E3779B97F4A7C15

    @staticmethod
    def _rotl(x: int, k: int) -> int:
        return ((x << k) & ((1 << 64) - 1)) | (x >> (64 - k))

    def next64(self) -> int:
        s0 = self.s0
        s1 = self.s1
        s2 = self.s2
        s3 = self.s3

        result = self._rotl(s1 * 5 & ((1 << 64) - 1), 7)
        result = (result * 9) & ((1 << 64) - 1)

        t = (s1 << 17) & ((1 << 64) - 1)

        s2 ^= s0
        s3 ^= s1
        s1 ^= s2
        s0 ^= s3

        s2 ^= t
        s3 = self._rotl(s3, 45)

        self.s0, self.s1, self.s2, self.s3 = s0, s1, s2, s3
        return result

    def randbits(self, k: int) -> int:
        if k <= 0:
            return 0
        out = 0
        bits_needed = k
        while bits_needed > 0:
            r = self.next64()
            take = min(bits_needed, 64)
            out = (out << take) | (r >> (64 - take))
            bits_needed -= take
        return out

    def uniform(self) -> float:
        # 53-bit precision uniform in [0,1)
        return (self.next64() >> 11) * (1.0 / (1 << 53))

    def choice(self, seq):
        idx = int(self.uniform() * len(seq))
        if idx == len(seq):
            idx -= 1
        return seq[idx]

    def weighted_choice(self, items_weights: List[Tuple[object, float]]):
        total = sum(w for _, w in items_weights)
        if total <= 0:
            raise ValueError("Non-positive total weight")
        r = self.uniform() * total
        upto = 0.0
        for item, w in items_weights:
            upto += w
            if r <= upto:
                return item
        return items_weights[-1][0]

    def normal(self, mu: float, sigma: float) -> float:
        # Box-Muller
        u1 = max(self.uniform(), 1e-12)
        u2 = self.uniform()
        z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * u2)
        return mu + z0 * sigma

    def randint(self, a: int, b: int) -> int:
        if a > b:
            a, b = b, a
        span = b - a + 1
        r = int(self.uniform() * span)
        if r == span:
            r -= 1
        return a + r


def gather_entropy() -> bytes:
    # Multiple local sources; all opt-in and ephemeral
    t_ns = time.perf_counter_ns().to_bytes(8, "little", signed=False)
    pid = os.getpid().to_bytes(4, "little", signed=False)
    rnd = secrets.token_bytes(32)
    fstats = str(os.stat(__file__) if "__file__" in globals() else "nofile").encode()
    mix = blake2b_digest(t_ns + pid + rnd + fstats)
    return mix


def domain_prngs(root_seed: bytes) -> Dict[str, Xoshiro256StarStar]:
    salt = b"RIDDLE.HKDF.SALT.v0"
    domains = [
        "form",
        "harmony",
        "rhythm",
        "melody",
        "synth",
        "ctrl",
        "video",
        "mythic",
        "names",
    ]
    prngs = {}
    for d in domains:
        seed32 = hkdf_sha256(root_seed, salt, d.encode(), 32)
        prngs[d] = Xoshiro256StarStar(seed32)
    return prngs


# ----------------------------- Music Theory Helpers -----------------------------
# Mode definitions as semitone sets relative to tonic (0=C)
MODES = {
    "lydian":       [0, 2, 4, 6, 7, 9, 11],
    "ionian_sharp11":[0, 2, 4, 6, 7, 9, 11],
    "whole_tone":   [0, 2, 4, 6, 8, 10],
    "messiaen_m3":  [0, 2, 3, 5, 6, 8, 9, 11],  # approximation
    "phrygian":     [0, 1, 3, 5, 7, 8, 10],
    "locrian":      [0, 1, 3, 5, 6, 8, 10],
    "hijaz":        [0, 1, 4, 5, 7, 8, 11],     # Phrygian dominant-ish
    "pelog5":       [0, 1, 3, 7, 8],            # stylized pentachord
    "dorian_b2":    [0, 1, 3, 5, 7, 9, 10],
}

NOTE_NAMES_SHARP = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]


def pick_theme(prng_names: Xoshiro256StarStar, requested: Optional[str]) -> str:
    if requested and requested.lower() in ("glass","salt"):
        return requested.lower()
    return prng_names.choice(["glass", "salt"])


def pick_key(prng: Xoshiro256StarStar, theme: str) -> Tuple[int, str]:
    # Returns (root_pc, mode_name)
    if theme == "glass":
        roots = [(2, 0.34), (6, 0.33), (9, 0.33)]  # D, F#, A
        modes = [("lydian",0.38), ("whole_tone",0.22), ("messiaen_m3",0.18), ("ionian_sharp11",0.12), ("lydian",0.10)]
    else:
        roots = [(0, 0.34), (5, 0.33), (8, 0.33)]  # C, F, G#
        modes = [("phrygian",0.33), ("locrian",0.22), ("hijaz",0.20), ("pelog5",0.15), ("dorian_b2",0.10)]

    root = prng.weighted_choice(roots)
    mode = prng.weighted_choice(modes)
    return root, mode


def scale_pitches(root_pc: int, mode_name: str) -> List[int]:
    scale = MODES[mode_name]
    return [(root_pc + x) % 12 for x in scale]


def midi_note_in_scale(prng: Xoshiro256StarStar, scale_pcs: List[int], min_note=36, max_note=84) -> int:
    # pick octave+pc in range
    while True:
        note = prng.randint(min_note, max_note)
        if (note % 12) in scale_pcs:
            return note


# ----------------------------- Length Engine -----------------------------
def sample_duration_seconds(prng_form: Xoshiro256StarStar, theme: str, override: Optional[str]) -> int:
    bucket_p = {"short":0.70, "med":0.25, "long":0.05}
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

    mu_sigma = {
        "short": (math.log(60.0), 0.45),
        "med":   (math.log(600.0), 0.50),
        "long":  (math.log(5400.0), 0.60),
    }
    # Theme clamps
    if theme == "glass":
        if prng_form.uniform() < 0.20:
            bucket = "med" if bucket == "short" else "long"
    else:
        if prng_form.uniform() < 0.20:
            bucket = "short" if bucket == "med" else "med"

    mu, sigma = mu_sigma[bucket]
    T = int(round(math.exp(prng_form.normal(mu, sigma))))
    # Clamp sanity
    T = max(20, min(T, 3*3600))
    return T


# ----------------------------- Form Grammar -----------------------------
FORM_PATHS = {
    "glass": [
        (("INV","PRC","TRN","PRC","SHR"), 0.32),
        (("INV","PRC","BRK","TRN","PRC","SHR"), 0.28),
        (("INV","CLM","PRC","TRN","SHR"), 0.18),
        (("INV","PRC","CLM","SHR"), 0.12),
        (("INV","TRN","PRC","TRN","SHR"), 0.10),
    ],
    "salt": [
        (("INV","PRC","CLM","SHR"), 0.34),
        (("INV","CLM","PRC","CLM","SHR"), 0.26),
        (("INV","PRC","BRK","CLM","SHR"), 0.22),
        (("INV","CLM","TRN","CLM","SHR"), 0.18),
    ],
}

SECTION_RANGES = {
    "glass": {"INV":(0.12,0.20),"PRC":(0.20,0.35),"BRK":(0.08,0.15),"TRN":(0.10,0.18),"CLM":(0.10,0.25),"SHR":(0.08,0.15)},
    "salt":  {"INV":(0.10,0.18),"PRC":(0.18,0.30),"BRK":(0.06,0.12),"TRN":(0.08,0.15),"CLM":(0.22,0.40),"SHR":(0.08,0.15)},
}


def pick_form(prng_form: Xoshiro256StarStar, theme: str, total_sec: int) -> Tuple[List[str], List[Tuple[str, float, float]]]:
    path = prng_form.weighted_choice(FORM_PATHS[theme])
    ranges = SECTION_RANGES[theme]
    # sample raw shares, then normalize to sum 1.0
    shares = []
    for node in path:
        lo, hi = ranges[node]
        shares.append(lo + prng_form.uniform() * (hi - lo))
    ssum = sum(shares)
    shares = [x / ssum for x in shares]
    # build timeline
    timeline = []
    t = 0.0
    for node, share in zip(path, shares):
        dur = total_sec * share
        timeline.append((node, t, t + dur))
        t += dur
    return list(path), timeline


# ----------------------------- Rhythm Generators -----------------------------
def euclidean_rhythm(k: int, n: int) -> List[int]:
    """
    Returns a list of length n with 1s as pulses distributed as evenly as possible.
    Bjorklund algorithm (simple variant).
    """
    if k <= 0:
        return [0] * n
    if k >= n:
        return [1] * n
    pattern = []
    counts = []
    remainders = []
    divisor = n - k
    remainders.append(k)
    while True:
        counts.append(divisor // remainders[-1])
        remainders.append(divisor % remainders[-1])
        divisor = remainders[-2]
        if remainders[-1] <= 1:
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
    pattern = build(len(counts)-1)
    # rotate so it starts with a pulse
    while pattern[0] != 1:
        pattern = pattern[1:] + pattern[:1]
    return pattern[:n]


def ca_gate(rule: int, n: int, seed_bits: int) -> List[int]:
    """
    Elementary cellular automaton row (one step) used as a gating mask.
    Deterministic from seed_bits; produce n bits.
    """
    # Build initial row from seed bits
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


# ----------------------------- MIDI Writer -----------------------------
class MidiBuilder:
    """
    Minimal SMF Type-1 writer (PPQ-based), standard lib only.
    """
    def __init__(self, ppq: int = 480):
        self.ppq = ppq
        self.tracks: List[bytearray] = []

    @staticmethod
    def _vlq(n: int) -> bytes:
        """Variable-length quantity encoding."""
        if n == 0:
            return b"\x00"
        buf = []
        while n > 0:
            buf.append(n & 0x7F)
            n >>= 7
        out = bytearray()
        for i, b7 in enumerate(reversed(buf)):
            if i < len(buf) - 1:
                out.append(0x80 | b7)
            else:
                out.append(b7)
        return bytes(out)

    @staticmethod
    def _meta(delta: int, mtype: int, data: bytes) -> bytes:
        return MidiBuilder._vlq(delta) + bytes([0xFF, mtype]) + MidiBuilder._vlq(len(data)) + data

    @staticmethod
    def _ev(delta: int, status: int, data1: int, data2: int) -> bytes:
        return MidiBuilder._vlq(delta) + bytes([status, data1 & 0x7F, data2 & 0x7F])

    def new_track(self) -> int:
        self.tracks.append(bytearray())
        return len(self.tracks) - 1

    def tempo(self, trk: int, delta: int, bpm: float) -> None:
        mpqn = int(60_000_000 / max(1.0, bpm))
        self.tracks[trk] += self._meta(delta, 0x51, struct.pack(">I", mpqn)[1:])

    def time_signature(self, trk: int, delta: int, nn: int, dd_pow: int) -> None:
        data = bytes([nn & 0xFF, dd_pow & 0xFF, 24, 8])
        self.tracks[trk] += self._meta(delta, 0x58, data)

    def key_signature(self, trk: int, delta: int, sf: int, minor: int) -> None:
        self.tracks[trk] += self._meta(delta, 0x59, bytes([sf & 0xFF, minor & 0x01]))

    def prog_change(self, trk: int, delta: int, ch: int, program: int) -> None:
        self.tracks[trk] += self._vlq(delta) + bytes([0xC0 | (ch & 0x0F), program & 0x7F])

    def note_on(self, trk: int, delta: int, ch: int, note: int, vel: int) -> None:
        self.tracks[trk] += self._ev(delta, 0x90 | (ch & 0x0F), note, vel)

    def note_off(self, trk: int, delta: int, ch: int, note: int, vel: int) -> None:
        self.tracks[trk] += self._ev(delta, 0x80 | (ch & 0x0F), note, vel)

    def controller(self, trk: int, delta: int, ch: int, cc: int, val: int) -> None:
        self.tracks[trk] += self._ev(delta, 0xB0 | (ch & 0x0F), cc, val)

    def end_track(self, trk: int, delta: int = 0) -> None:
        self.tracks[trk] += self._meta(delta, 0x2F, b"")

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            # Header: MThd, length=6, format=1, ntrks, division=ppq
            f.write(b"MThd")
            f.write(struct.pack(">I", 6))
            f.write(struct.pack(">HHH", 1, len(self.tracks), self.ppq))
            for tr in self.tracks:
                f.write(b"MTrk")
                f.write(struct.pack(">I", len(tr)))
                f.write(tr)


# ----------------------------- Simple Synth Rack -----------------------------
class StereoLimiter:
    def __init__(self, ceiling=0.97, lookahead_samples=0, release=0.005):
        self.ceiling = float(ceiling)
        self.lookahead = int(lookahead_samples)
        self.release = float(release)
        self.gain = 1.0

    def process_block(self, L: List[float], R: List[float]) -> Tuple[List[float], List[float]]:
        outL, outR = [], []
        for l, r in zip(L, R):
            peak = max(abs(l), abs(r))
            target = 1.0
            if peak * self.gain > self.ceiling:
                target = self.ceiling / max(peak, 1e-12)
            # simple one-pole smoothing toward target
            self.gain += (target - self.gain) * self.release
            outL.append(l * self.gain)
            outR.append(r * self.gain)
        return outL, outR


def sine(phase: float) -> float:
    return math.sin(phase)


def tri(phase: float) -> float:
    # phase in radians; convert to [0,1) saw then triangle
    x = (phase / (2*math.pi)) % 1.0
    return 4.0 * abs(x - 0.5) - 1.0


def softsat(x: float, drive: float = 1.0) -> float:
    return math.tanh(drive * x)


def hz_from_midi(note: int) -> float:
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


class Voice:
    """
    Simple poly voice for FM lead or WT pad.
    """
    def __init__(self, sr: int, mode: str):
        self.sr = sr
        self.mode = mode
        self.phase_c = 0.0
        self.phase_m = 0.0
        self.env = 0.0
        self.active = False
        self.release = False
        self.note = 60
        self.freq = hz_from_midi(60)
        # params differ by mode
        if mode == "fm_lead":
            self.mod_idx = 2.0
            self.ratio = 2.0
            self.a, self.d, self.s, self.r = 0.01, 0.20, 0.5, 0.30
        elif mode == "wt_pad":
            self.mod_idx = 0.4      # used as WT depth
            self.ratio = 0.5
            self.a, self.d, self.s, self.r = 0.6, 0.8, 0.7, 1.2
        else:
            self.mod_idx = 1.0
            self.ratio = 1.0
            self.a, self.d, self.s, self.r = 0.01, 0.1, 0.7, 0.2

    def note_on(self, note: int, vel: float):
        self.note = note
        self.freq = hz_from_midi(note)
        self.env = 0.0
        self.active = True
        self.release = False
        self.vel = vel

    def note_off(self):
        self.release = True

    def _adsr(self, dt: float):
        if not self.release:
            if self.env < 1.0:
                self.env = min(1.0, self.env + dt / max(self.a, 1e-5))
            else:
                # decay toward sustain
                if self.env > self.s:
                    self.env = max(self.s, self.env - dt / max(self.d, 1e-5))
        else:
            self.env = max(0.0, self.env - dt / max(self.r, 1e-5))
            if self.env <= 1e-4:
                self.active = False

    def render(self, n: int, sr: int, lfo_val: float) -> List[float]:
        out = []
        dt = 1.0 / sr
        for _ in range(n):
            self._adsr(dt)
            if self.mode == "fm_lead":
                self.phase_m += 2*math.pi * (self.freq * self.ratio) * dt
                mod = math.sin(self.phase_m) * self.mod_idx * (0.6 + 0.4 * lfo_val)
                self.phase_c += 2*math.pi * self.freq * dt + mod * dt
                s = math.sin(self.phase_c)
            else:  # wt_pad (very simple "wavetable" morphing: sine->triangle crossfade)
                self.phase_c += 2*math.pi * self.freq * dt
                wt = 0.5 + 0.5 * math.sin(lfo_val * 2*math.pi)  # 0..1 morph
                s = (1.0 - wt) * math.sin(self.phase_c) + wt * tri(self.phase_c)
            out.append(self.env * self.vel * s)
        return out


class SubBass:
    def __init__(self, sr: int):
        self.sr = sr
        self.active_notes: Dict[int, float] = {}  # note -> phase

    def note_on(self, note: int, vel: float):
        self.active_notes[note] = 0.0

    def note_off(self, note: int):
        self.active_notes.pop(note, None)

    def render(self, n: int) -> List[float]:
        out = [0.0] * n
        dt = 1.0 / self.sr
        for note, phase in list(self.active_notes.items()):
            f = hz_from_midi(note) / 2.0  # subharmonic
            for i in range(n):
                phase += 2*math.pi * f * dt
                out[i] += 0.4 * math.sin(phase)
            self.active_notes[note] = phase
        return out


class NoisePerc:
    def __init__(self, sr: int):
        self.sr = sr
        self.env = 0.0
        self.decay = 0.08
        self.trig = False

    def hit(self, strength: float = 1.0):
        self.env = min(1.0, self.env + 0.8 * strength)
        self.trig = True

    def render(self, n: int) -> List[float]:
        out = []
        for _ in range(n):
            # xorshift32 for noise
            rnd = secrets.randbits(32) & 0xFFFFFFFF
            rnd = (rnd ^ (rnd << 13)) & 0xFFFFFFFF
            rnd = (rnd ^ (rnd >> 17)) & 0xFFFFFFFF
            rnd = (rnd ^ (rnd << 5)) & 0xFFFFFFFF
            noise = ((rnd / 0xFFFFFFFF) * 2.0) - 1.0
            self.env = max(0.0, self.env - (1.0 / (self.sr * self.decay)))
            out.append(self.env * 0.5 * noise)
        return out


# ----------------------------- Composition Engines -----------------------------
def pick_time_signature(prng: Xoshiro256StarStar, theme: str) -> Tuple[int, int]:
    if theme == "glass":
        choices = [((4,4),0.35), ((5,4),0.22), ((7,8),0.18), ((9,8),0.12), ((11,8),0.08), ((13,8),0.05)]
    else:
        choices = [((7,8),0.24), ((5,4),0.22), ((3,4),0.18), (("2+2+3",8),0.18), ((4,4),0.18)]
    ts = prng.weighted_choice(choices)
    if isinstance(ts[0], str):
        # additive, approximate to 7/8 for MIDI time sig
        return (7, 8)
    return ts


def build_melody(prng_mel: Xoshiro256StarStar, scale_pcs: List[int], bars: int, ppq: int, ts_num: int, ts_den: int,
                 channel: int, density: float, base_octave: int, sigil_pcs: List[int]) -> List[Tuple[int,int,int,int]]:
    """
    Returns list of (tick, duration_ticks, note, velocity)
    """
    events = []
    beats_per_bar = ts_num * (4 / ts_den)
    ticks_per_bar = int(ppq * beats_per_bar)
    min_note = 48 + (base_octave-4)*12
    max_note = 84 + (base_octave-4)*12
    for b in range(bars):
        t0 = b * ticks_per_bar
        # simple rhythmic grid: 8th notes subdivision
        div = 2
        step = ticks_per_bar // (int(beats_per_bar*div))
        for i in range(int(beats_per_bar*div)):
            if prng_mel.uniform() < density:
                pc_pool = sigil_pcs if prng_mel.uniform() < 0.65 else scale_pcs
                note = midi_note_in_scale(prng_mel, pc_pool, min_note, max_note)
                dur = int(step * (1 + prng_mel.uniform()*1.5))
                vel = 64 + int(prng_mel.uniform()*50)
                events.append((t0 + i*step, dur, note, vel))
    return events


def build_bass(prng: Xoshiro256StarStar, scale_pcs: List[int], bars: int, ppq: int, ts_num: int, ts_den: int) -> List[Tuple[int,int,int,int]]:
    events = []
    beats_per_bar = ts_num * (4 / ts_den)
    ticks_per_bar = int(ppq * beats_per_bar)
    for b in range(bars):
        t0 = b * ticks_per_bar
        root = midi_note_in_scale(prng, scale_pcs, 36, 52)
        vel = 60 + int(prng.uniform()*30)
        dur = int(ppq * (beats_per_bar * 0.75))
        events.append((t0, dur, root, vel))
    return events


def build_perc(prng: Xoshiro256StarStar, bars: int, ppq: int, ts_num: int, ts_den: int, theme: str) -> List[Tuple[int,int,int,int]]:
    events = []
    beats_per_bar = ts_num * (4 / ts_den)
    ticks_per_bar = int(ppq * beats_per_bar)
    # Choose Euclidean pattern and gate it
    if theme == "glass":
        ek, en = (5, 8)
        rule = 90
    else:
        ek, en = (3, 8)
        rule = 150
    e = euclidean_rhythm(ek, en)
    gate = ca_gate(rule, en, prng.randbits(32))
    patt = [1 if (e[i] & gate[i]) else 0 for i in range(en)]
    step = ticks_per_bar // en
    for b in range(bars):
        base = b * ticks_per_bar
        for i, on in enumerate(patt):
            if on:
                t = base + i*step
                # percussion channel (10): midi note 42 (closed hat) or 36 (kick) occasionally
                note = 42 if prng.uniform() < 0.8 else 36
                vel = 50 + int(prng.uniform()*40)
                events.append((t, int(step*0.9), note, vel))
    return events


# ----------------------------- WAV Rendering -----------------------------
def render_audio(output_path: Path, midi_events: Dict[str, List[Tuple[int,int,int,int]]], bpm: float,
                 ppq: int, sr: int, total_sec: int, limiter_ceiling: float = 0.97) -> None:
    """Very simple, streamed renderer based on MIDI-like events."""
    # Convert BPM to seconds per tick
    s_per_tick = 60.0 / bpm / ppq
    # Build event timeline per sample
    # Gather note-ons/offs per track
    lead = midi_events.get("lead", [])
    pad  = midi_events.get("pad", [])
    bass = midi_events.get("bass", [])
    perc = midi_events.get("perc", [])

    # Active voices
    lead_voices: List[Voice] = []
    pad_voices: List[Voice] = []
    sub_bass = SubBass(sr)
    drums = NoisePerc(sr)
    limiter = StereoLimiter(ceiling=limiter_ceiling)

    # Convert event ticks to absolute seconds
    def expand(events):
        seq = []
        for t, dur, note, vel in events:
            seq.append(("on", t*s_per_tick, note, vel))
            seq.append(("off", (t+dur)*s_per_tick, note, 0))
        return sorted(seq, key=lambda x: x[1])

    events_timed = {
        "lead": expand(lead),
        "pad":  expand(pad),
        "bass": expand(bass),
        "perc": [( "hit", t*s_per_tick, note, vel) for (t, d, note, vel) in perc],  # only hits
    }

    # Streaming render
    block = 1024
    total_samples = total_sec * sr
    t_sec = 0.0
    evt_idx = {k:0 for k in events_timed.keys()}

    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(3)  # 24-bit
        wf.setframerate(sr)
        # Main loop
        while int(t_sec * sr) < total_samples:
            # trigger any events within this block
            t_end = t_sec + block / sr
            for track in ("lead","pad","bass","perc"):
                evts = events_timed[track]
                i = evt_idx[track]
                while i < len(evts) and evts[i][1] < t_end:
                    typ, tt, note, vel = evts[i]
                    i += 1
                    if track == "lead":
                        if typ == "on":
                            v = Voice(sr, "fm_lead")
                            v.note_on(note, vel/127.0)
                            lead_voices.append(v)
                        else:
                            # note-off: release nearest same-note voice
                            for v in lead_voices:
                                if v.note == note and v.active:
                                    v.note_off()
                                    break
                    elif track == "pad":
                        if typ == "on":
                            v = Voice(sr, "wt_pad")
                            v.note_on(note, vel/127.0 * 0.6)
                            pad_voices.append(v)
                        else:
                            for v in pad_voices:
                                if v.note == note and v.active:
                                    v.note_off()
                                    break
                    elif track == "bass":
                        if typ == "on":
                            sub_bass.note_on(note - 12, vel/127.0)
                        else:
                            sub_bass.note_off(note - 12)
                    elif track == "perc" and typ == "hit":
                        drums.hit(vel/127.0)
                evt_idx[track] = i

            # render block
            L = [0.0]*block
            R = [0.0]*block

            # very slow LFO for pad morph
            lfo = math.sin(2*math.pi * 0.02 * t_sec)

            # sum voices
            # lead
            alive_leads = []
            for v in lead_voices:
                buf = v.render(block, sr, lfo)
                if v.active:
                    alive_leads.append(v)
                for i, s in enumerate(buf):
                    L[i] += 0.22 * s
                    R[i] += 0.22 * s
            lead_voices = alive_leads

            # pad
            alive_pads = []
            for v in pad_voices:
                buf = v.render(block, sr, lfo)
                if v.active:
                    alive_pads.append(v)
                for i, s in enumerate(buf):
                    L[i] += 0.18 * s
                    R[i] += 0.18 * s
            pad_voices = alive_pads

            # bass
            bbuf = sub_bass.render(block)
            for i, s in enumerate(bbuf):
                L[i] += 0.20 * s
                R[i] += 0.20 * s

            # perc
            pbuf = drums.render(block)
            for i, s in enumerate(pbuf):
                L[i] += 0.20 * s
                R[i] += 0.20 * s

            # soft saturation and limiter
            for i in range(block):
                L[i] = softsat(L[i], 1.2)
                R[i] = softsat(R[i], 1.2)

            L, R = limiter.process_block(L, R)

            # interleave 24-bit
            frames = bytearray()
            for i in range(block):
                l = max(-0.999999, min(0.999999, L[i]))
                r = max(-0.999999, min(0.999999, R[i]))
                li = int(l * 8388607.0)
                ri = int(r * 8388607.0)
                frames += struct.pack("<i", li)[0:3]
                frames += struct.pack("<i", ri)[0:3]
            wf.writeframes(frames)
            t_sec = t_end


# ----------------------------- Mythic Variants -----------------------------
def mythic_backmask(src_wav: Path, dst_wav: Path) -> None:
    with wave.open(str(src_wav), "rb") as r:
        params = r.getparams()
        frames = r.readframes(r.getnframes())
    # Reverse frames at sample granularity (24-bit stereo)
    # Convert to 16-bit via audioop for safe reverse; then back to 24-bit
    sampwidth = params.sampwidth
    nch = params.nchannels
    if sampwidth != 3:
        raise ValueError("Expected 24-bit WAV")
    # downconvert to 16-bit for reversal convenience
    pcm16 = audioop.lin2lin(frames, 3, 2)
    rev16 = audioop.reverse(pcm16, 2)
    # back to 24-bit
    frames24 = audioop.lin2lin(rev16, 2, 3)
    with wave.open(str(dst_wav), "wb") as w:
        w.setparams(params)
        w.writeframes(frames24)


def mythic_ashen_bitcrush(src_wav: Path, dst_wav: Path, bits: int = 12) -> None:
    with wave.open(str(src_wav), "rb") as r:
        params = r.getparams()
        frames = r.readframes(r.getnframes())
    if params.sampwidth != 3:
        raise ValueError("Expected 24-bit WAV")
    pcm16 = audioop.lin2lin(frames, 3, 2)
    # Quantize
    step = 1 << (16 - bits)
    crushed = audioop.bias(audioop.mul(pcm16, 2, 1.0), 2, 0)
    # manual quantization
    arr = bytearray(crushed)
    for i in range(0, len(arr), 2):
        s = int.from_bytes(arr[i:i+2], "little", signed=True)
        s = int(round(s / step) * step)
        s = max(-32768, min(32767, s))
        arr[i:i+2] = int(s).to_bytes(2, "little", signed=True)
    frames24 = audioop.lin2lin(bytes(arr), 2, 3)
    with wave.open(str(dst_wav), "wb") as w:
        w.setparams(params)
        w.writeframes(frames24)


def mythic_mirrorsalt_ms(src_wav: Path, dst_wav: Path) -> None:
    with wave.open(str(src_wav), "rb") as r:
        params = r.getparams()
        frames = r.readframes(r.getnframes())
    if params.nchannels != 2 or params.sampwidth != 3:
        raise ValueError("Expected 24-bit stereo WAV")
    # Convert to 16-bit for MS ops
    pcm16 = audioop.lin2lin(frames, 3, 2)
    # Split L/R
    L = audioop.tomono(pcm16, 2, 1.0, 0.0)
    R = audioop.tomono(pcm16, 2, 0.0, 1.0)
    # M/S transform and microphase wobble by mixing slightly
    M = audioop.add(L, R, 2)
    S = audioop.sub(L, R, 2)
    # swap M/S -> (M stays, S inverted slightly)
    S = audioop.mul(S, 2, -0.98)
    # back to L/R: L = (M+S)/2, R = (M-S)/2
    L2 = audioop.mul(audioop.add(M, S, 2), 2, 0.5)
    R2 = audioop.mul(audioop.sub(M, S, 2), 2, 0.5)
    stereo = audioop.tostereo(L2, 2, 1.0, 0.0)
    stereo = audioop.add(stereo, audioop.tostereo(R2, 2, 0.0, 1.0), 2)
    frames24 = audioop.lin2lin(stereo, 2, 3)
    with wave.open(str(dst_wav), "wb") as w:
        w.setparams(params)
        w.writeframes(frames24)


def mythic_liminal_bed(src_wav: Path, dst_wav: Path, gain_db: float = -20.0, max_minutes: int = 60) -> None:
    with wave.open(str(src_wav), "rb") as r:
        params = r.getparams()
        sr = r.getframerate()
        nframes = min(r.getnframes(), max_minutes*60*sr)
        frames = r.readframes(nframes)
    if params.sampwidth != 3:
        raise ValueError("Expected 24-bit WAV")
    pcm16 = audioop.lin2lin(frames, 3, 2)
    gain = 10 ** (gain_db / 20.0)
    quiet = audioop.mul(pcm16, 2, gain)
    frames24 = audioop.lin2lin(quiet, 2, 3)
    with wave.open(str(dst_wav), "wb") as w:
        w.setparams((params.nchannels, 3, params.framerate, 0, params.comptype, params.compname))
        w.writeframes(frames24)


def mythic_cipherspray_watermark(src_wav: Path, dst_wav: Path, seed_hex: str) -> None:
    """
    Embed an ultrasonic single-tone watermark (e.g., 19kHz) modulated by seed bits at low amplitude.
    """
    with wave.open(str(src_wav), "rb") as r:
        params = r.getparams()
        sr = r.getframerate()
        nframes = r.getnframes()
        frames = r.readframes(nframes)
    if params.sampwidth != 3 or params.nchannels != 2:
        raise ValueError("Expected 24-bit stereo WAV")
    pcm16 = audioop.lin2lin(frames, 3, 2)

    # Build watermark
    duration = nframes / sr
    tone_hz = 19000.0 if sr >= 48000 else sr*0.39  # stay high but safe
    amp = 0.02  # very low
    # seed bits
    bits = bin(int(seed_hex, 16))[2:].zfill(64)
    # We'll modulate amplitude every 0.25s based on bits
    segment = 0.25
    seg_frames = int(segment * sr)
    out = bytearray()
    phase = 0.0
    for i in range(0, len(pcm16), 4):  # 2ch * 2 bytes
        idx = i // 4
        k = (idx // seg_frames) % len(bits)
        a = amp if bits[k] == "1" else amp*0.4
        phase += 2*math.pi * tone_hz / sr
        s = int(max(-32768, min(32767, int(math.sin(phase) * a * 32767))))
        # mix into both channels lightly
        l = int.from_bytes(pcm16[i:i+2], "little", signed=True)
        r = int.from_bytes(pcm16[i+2:i+4], "little", signed=True)
        l = max(-32768, min(32767, l + s))
        r = max(-32768, min(32767, r + s))
        out += int(l).to_bytes(2, "little", signed=True)
        out += int(r).to_bytes(2, "little", signed=True)

    frames24 = audioop.lin2lin(bytes(out), 2, 3)
    with wave.open(str(dst_wav), "wb") as w:
        w.setparams(params)
        w.writeframes(frames24)


# ----------------------------- Vault (SQLite) -----------------------------
def ensure_vault(db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS runs (
          id INTEGER PRIMARY KEY,
          started_utc TEXT NOT NULL,
          theme TEXT NOT NULL,
          seed_commitment TEXT NOT NULL UNIQUE,
          duration_sec INTEGER NOT NULL,
          form_path TEXT NOT NULL,
          bpm_base REAL NOT NULL,
          lufs_target REAL,
          coherence REAL,
          presence REAL,
          hostility REAL,
          obliquity REAL
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS run_harmony (
          run_id INTEGER REFERENCES runs(id),
          section_idx INTEGER,
          key_root TEXT,
          mode TEXT,
          start_sec REAL,
          end_sec REAL,
          PRIMARY KEY (run_id, section_idx)
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS run_rhythm (
          run_id INTEGER REFERENCES runs(id),
          section_idx INTEGER,
          time_sig TEXT,
          euclid TEXT,
          ca_rule INTEGER,
          swing REAL,
          PRIMARY KEY (run_id, section_idx)
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS artifacts (
          id INTEGER PRIMARY KEY,
          run_id INTEGER REFERENCES runs(id),
          kind TEXT CHECK(kind IN ('wav','midi','mp4','json','stem','mythic')),
          path TEXT NOT NULL,
          sha256 TEXT NOT NULL,
          duration_sec REAL,
          bpm_est REAL,
          key_hint TEXT,
          mythic_type TEXT
        );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_kind ON artifacts(kind);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_runs_theme ON runs(theme);")
        conn.commit()
    finally:
        conn.close()


def vault_insert_run(db_path: Path, run) -> int:
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute("""
        INSERT INTO runs(started_utc, theme, seed_commitment, duration_sec, form_path, bpm_base, lufs_target, coherence, presence, hostility, obliquity)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run["started_utc"], run["theme"], run["seed_commitment"], run["duration_sec"],
            run["form_path"], run["bpm_base"], run["lufs_target"], run["coherence"],
            run["presence"], run["hostility"], run["obliquity"]
        ))
        run_id = cur.lastrowid
        for idx, sec in enumerate(run["sections"]):
            cur.execute("""
            INSERT INTO run_harmony(run_id, section_idx, key_root, mode, start_sec, end_sec)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (run_id, idx, sec["key_root"], sec["mode"], sec["start_sec"], sec["end_sec"]))
            cur.execute("""
            INSERT INTO run_rhythm(run_id, section_idx, time_sig, euclid, ca_rule, swing)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (run_id, idx, sec["time_sig"], sec["euclid"], sec["ca_rule"], sec["swing"]))
        conn.commit()
        return run_id
    finally:
        conn.close()


def vault_insert_artifact(db_path: Path, run_id: int, kind: str, path: Path, duration_sec: float,
                          bpm_est: Optional[float], key_hint: Optional[str], mythic_type: Optional[str]) -> None:
    sha256 = sha256_of_file(path)
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute("""
        INSERT INTO artifacts(run_id, kind, path, sha256, duration_sec, bpm_est, key_hint, mythic_type)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (run_id, kind, str(path), sha256, duration_sec, bpm_est, key_hint, mythic_type))
        conn.commit()
    finally:
        conn.close()


# ----------------------------- Utilities -----------------------------
def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def seed_commitment(root_seed: bytes) -> str:
    return hashlib.blake2b(root_seed, digest_size=32).hexdigest()


def timestamp_now_utc() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


# ----------------------------- Main Pipeline -----------------------------
def run_riddle(theme_req: Optional[str], outdir: Path, db_path: Path, duration_bucket: Optional[str],
               stems: bool, mythic_max: int, lufs_target: float, verbosity: int) -> None:
    try:
        setup_logging(verbosity)
        logging.info("[i] Starting The Infinite Riddle v0.1")
        ensure_vault(db_path)

        # Entropy & PRNG split
        entropy = gather_entropy()
        root_seed = blake2b_digest(entropy)
        commit = seed_commitment(root_seed)
        prngs = domain_prngs(root_seed)

        # Theme, duration, form
        theme = pick_theme(prngs["names"], theme_req)
        total_sec = sample_duration_seconds(prngs["form"], theme, duration_bucket)
        form_nodes, timeline = pick_form(prngs["form"], theme, total_sec)
        logging.info("[i] Theme=%s Duration=%ds Form=%s", theme, total_sec, "→".join(form_nodes))

        # BPM base
        if theme == "glass":
            bpm_base = 96.0 + (prngs["rhythm"].uniform()-0.5)*30.0
        else:
            bpm_base = 64.0 + (prngs["rhythm"].uniform()-0.5)*24.0
        bpm_base = max(36.0, min(160.0, bpm_base))

        # Key/Mode per run (single for v0.1)
        key_root, mode_name = pick_key(prngs["harmony"], theme)
        key_pc_name = NOTE_NAMES_SHARP[key_root]
        scale_pcs = scale_pitches(key_root, mode_name)
        sigil_pcs = prngs["melody"].choice([scale_pcs, scale_pcs[:max(2, len(scale_pcs)//2)]])
        logging.info("[i] Key=%s %s; BPM≈%.1f", key_pc_name, mode_name, bpm_base)

        # MIDI composition
        ppq = 480
        # Resolve bars per section at chosen time signature
        ts_num, ts_den = pick_time_signature(prngs["rhythm"], theme)
        beats_per_bar = ts_num * (4/ts_den)
        bars_total = max(1, int((total_sec / 60.0) * (bpm_base / beats_per_bar)))
        # Split bars across sections by proportion
        section_bars = []
        sec_len = [end-start for (_, start, end) in timeline]
        total_len = sum(sec_len)
        for (node, start, end) in timeline:
            share = (end - start) / total_len
            section_bars.append(max(1, int(round(bars_total * share))))

        # Build per-track events over total bars
        bars = sum(section_bars)
        lead_events = build_melody(prngs["melody"], scale_pcs, bars, ppq, ts_num, ts_den, channel=0,
                                   density=0.35 if theme=="glass" else 0.25, base_octave=5 if theme=="glass" else 4,
                                   sigil_pcs=sigil_pcs)
        pad_events  = build_melody(prngs["melody"], scale_pcs, bars, ppq, ts_num, ts_den, channel=1,
                                   density=0.20 if theme=="glass" else 0.18, base_octave=4, sigil_pcs=sigil_pcs)
        bass_events = build_bass(prngs["melody"], scale_pcs, bars, ppq, ts_num, ts_den)
        perc_events = build_perc(prngs["rhythm"], bars, ppq, ts_num, ts_den, theme)

        # MIDI build
        mb = MidiBuilder(ppq=ppq)
        tr_meta = mb.new_track()
        mb.tempo(tr_meta, 0, bpm_base)
        mb.time_signature(tr_meta, 0, ts_num, int(round(math.log2(ts_den))))
        # Key signature meta (approx; sharps positive, flats negative)
        sharps = [1,3,6,8,10]  # crude mapping to prefer sharps
        sf = sharps.index(key_root) + 1 if key_root in sharps else -( [0,2,5,7,9,11].index(key_root) + 1 )
        mb.key_signature(tr_meta, 0, sf, 0)

        def write_track(events, ch, program_hint):
            t = mb.new_track()
            mb.prog_change(t, 0, ch, program_hint)
            events_sorted = sorted(events, key=lambda x: x[0])
            last_tick = 0
            for tick, dur, note, vel in events_sorted:
                delta = max(0, tick - last_tick)
                mb.note_on(t, delta, ch, note, vel)
                mb.note_off(t, dur, ch, note, max(32, vel//2))
                last_tick = tick + dur
            mb.end_track(t, 0)

        write_track(lead_events, 0, 11)   # vibraphone-ish
        write_track(pad_events, 1, 89)    # pad
        write_track(bass_events, 2, 34)   # electric bass
        write_track(perc_events, 9, 0)    # standard kit

        # Paths & filenames
        now = dt.datetime.now()
        seed8 = commit[:8]
        form_str = "→".join(form_nodes)
        keymode = f"{NOTE_NAMES_SHARP[key_root]}_{mode_name}"
        mm = total_sec // 60
        ss = total_sec % 60
        len_tag = f"{mm:02d}-{ss:02d}"
        base = f"{now:%Y%m%d_%H%M%S}_RIDDLE_{'Glass' if theme=='glass' else 'Salt'}_{form_str}_BPM{int(round(bpm_base))}_{keymode}_LEN{len_tag}_{seed8}"

        outdir.mkdir(parents=True, exist_ok=True)
        json_path = outdir / f"{base}.riddle.json"
        midi_path = outdir / f"{base}_MIDI_{seed8}.mid"
        wav_path  = outdir / f"{base}.wav"

        # Save MIDI
        mb.save(midi_path)
        logging.info("[i] MIDI written: %s", midi_path.name)

        # Render WAV (streamed)
        midi_dict = {"lead":lead_events, "pad":pad_events, "bass":bass_events, "perc":perc_events}
        render_audio(wav_path, midi_dict, bpm_base, ppq, 48000, total_sec)
        logging.info("[i] WAV written: %s", wav_path.name)

        # Build sidecar JSON
        sidecar = {
            "seed_commitment": commit,
            "theme": theme,
            "form_nodes": form_nodes,
            "form_path": form_str,
            "durations": [{"node":n, "start":float(s), "end":float(e)} for (n,s,e) in timeline],
            "bpm_base": bpm_base,
            "key_mode": {"root_pc": key_root, "root_name": key_pc_name, "mode": mode_name},
            "sigil_pcs": sigil_pcs,
            "artifact_hashes": {
                "midi": sha256_of_file(midi_path),
                "wav": sha256_of_file(wav_path),
            },
            "started_utc": timestamp_now_utc(),
            "loudness_target": lufs_target,
            "crest_factor_est": None,
            "true_peak_est": None,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(sidecar, f, indent=2)
        logging.info("[i] Sidecar JSON written: %s", json_path.name)

        # Vault inserts
        run_obj = {
            "started_utc": sidecar["started_utc"],
            "theme": theme,
            "seed_commitment": commit,
            "duration_sec": total_sec,
            "form_path": form_str,
            "bpm_base": bpm_base,
            "lufs_target": lufs_target,
            "coherence": 0.7 if theme=="glass" else 0.6,
            "presence": 0.5,
            "hostility": 0.2,
            "obliquity": 0.6 if theme=="glass" else 0.7,
            "sections": [
                {
                    "key_root": key_pc_name,
                    "mode": mode_name,
                    "start_sec": float(s),
                    "end_sec": float(e),
                    "time_sig": f"{ts_num}/{ts_den}",
                    "euclid": "E(5,8)|E(7,12)" if theme=="glass" else "E(3,8)|E(5,12)",
                    "ca_rule": 90 if theme=="glass" else 150,
                    "swing": 0.06 if theme=="glass" else -0.04
                }
                for (n,s,e) in timeline
            ]
        }
        run_id = vault_insert_run(db_path, run_obj)
        vault_insert_artifact(db_path, run_id, "json", json_path, 0.0, None, None, None)
        vault_insert_artifact(db_path, run_id, "midi", midi_path, total_sec, bpm_base, keymode, None)
        vault_insert_artifact(db_path, run_id, "wav",  wav_path,  total_sec, bpm_base, keymode, None)

        # Mythic variants (choose up to mythic_max we can implement)
        mythics = [
            ("Backmask", mythic_backmask),
            ("Ashen", mythic_ashen_bitcrush),
            ("MirrorSalt", mythic_mirrorsalt_ms),
            ("Liminal", mythic_liminal_bed),
            ("CipherSpray", lambda src, dst: mythic_cipherspray_watermark(src, dst, commit[:16])),
        ]
        prng_m = prngs["mythic"]
        rolled = 0
        for label, fn in mythics:
            if rolled >= mythic_max:
                break
            # independent rolls with modest probability
            if prng_m.uniform() < 0.22:  # ~22% per variant
                dst = outdir / f"{base}_MYTHIC_{label}_{seed8}.wav"
                try:
                    fn(wav_path, dst)
                    vault_insert_artifact(db_path, run_id, "mythic", dst, total_sec, bpm_base, keymode, label)
                    logging.info("[i] Mythic generated: %s", dst.name)
                    rolled += 1
                except Exception as e:
                    logging.error("[x] Mythic %s failed: %s", label, e)

        logging.info("[i] Completed run. SeedCommit=%s", commit[:16])
    except Exception as e:
        logging.exception("[x] Fatal error: %s", e)
        sys.exit(1)


# ----------------------------- CLI -----------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="The Infinite Riddle — generate unique WAV/MIDI artifacts with provenance.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("theme", choices=["glass","salt","auto"], help="Select theme or auto.")
    p.add_argument("outdir", help="Output directory for artifacts.")
    p.add_argument("--db", default="riddle_vault.db", help="Path to SQLite vault.")
    p.add_argument("--bucket", choices=["short","med","long"], default=None, help="Force duration bucket.")
    p.add_argument("--stems", action="store_true", help="(Reserved) Emit stems (future).")
    p.add_argument("--mythic-max", type=int, default=2, help="Max mythic variants to attempt.")
    p.add_argument("--lufs-target", type=float, default=-14.0, help="Target loudness metadata.")
    p.add_argument("-v", "--verbose", action="count", default=1, help="Increase verbosity (-v or -vv).")
    return p.parse_args(argv)


def _resolve_artifact_paths(outdir_arg: str, db_arg: str, root: Path) -> Tuple[Path, Path]:
    outdir = Path(outdir_arg).resolve()
    db_path = Path(db_arg).resolve()
    try:
        outdir.relative_to(root)
        db_path.relative_to(root)
    except ValueError:
        raise ValueError(f"Paths must reside within {root}")
    outdir.mkdir(parents=True, exist_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return outdir, db_path


def main():
    args = parse_args()
    root = Path.cwd().resolve()
    outdir, db_path = _resolve_artifact_paths(args.outdir, args.db, root)
    theme = None if args.theme == "auto" else args.theme
    run_riddle(theme, outdir, db_path, args.bucket, args.stems, args.mythic_max, args.lufs_target, args.verbose)


if __name__ == "__main__":
    main()


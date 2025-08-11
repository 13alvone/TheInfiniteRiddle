#!/usr/bin/env python3
"""I/O helpers for The Infinite Riddle."""
import hashlib
import struct
from pathlib import Path


class MidiBuilder:
    """Minimal SMF Type-1 writer (PPQ-based)."""
    def __init__(self, ppq: int = 480):
        self.ppq = ppq
        self.tracks = []

    @staticmethod
    def _vlq(n: int) -> bytes:
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
            f.write(b"MThd")
            f.write(struct.pack(">I", 6))
            f.write(struct.pack(">HHH", 1, len(self.tracks), self.ppq))
            for tr in self.tracks:
                f.write(b"MTrk")
                f.write(struct.pack(">I", len(tr)))
                f.write(tr)


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def seed_commitment(root_seed: bytes) -> str:
    return hashlib.blake2b(root_seed, digest_size=16).hexdigest()


def timestamp_now_utc() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

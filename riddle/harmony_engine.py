#!/usr/bin/env python3
"""Simple harmony and melody generator for The Infinite Riddle.

This module exposes dataclasses for chords, voicings, melody notes, and a
``HarmonicPlan`` container.  It also provides a small command line interface
available as ``python -m riddle.harmony_engine gen`` which outputs a
JSON description of a generated harmonic plan.  The musical logic is minimal
but deterministic and sufficient for tests.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sqlite3
import sys
from dataclasses import dataclass, asdict
from hashlib import blake2b
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Logging helpers -----------------------------------------------------------
# ---------------------------------------------------------------------------

_PREFIX = {
    logging.INFO: "[i]",
    logging.WARNING: "[!]",
    logging.DEBUG: "[DEBUG]",
    logging.ERROR: "[x]",
}


class PrefixFormatter(logging.Formatter):
    """Formatter that injects required log level prefixes."""

    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - thin wrapper
        prefix = _PREFIX.get(record.levelno, "[i]")
        msg = super().format(record)
        return f"{prefix} {msg}"


def setup_logging(verbosity: int) -> None:
    """Configure logging according to -v/-vv flags."""
    level = logging.WARNING
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO
    handler = logging.StreamHandler()
    handler.setFormatter(PrefixFormatter("%(message)s"))
    logging.basicConfig(level=level, handlers=[handler], force=True)


# ---------------------------------------------------------------------------
# Dataclasses ---------------------------------------------------------------
# ---------------------------------------------------------------------------

@dataclass
class Chord:
    name: str
    pitch_classes: List[int]
    function: str
    roman: str


@dataclass
class Voicing:
    soprano: int
    alto: int
    tenor: int
    bass: int


@dataclass
class ChordEvent:
    chord: Chord
    voicing: Voicing
    duration: float


@dataclass
class MelodyNote:
    pitch: int
    start: float
    duration: float
    velocity: int


@dataclass
class HarmonicPlan:
    seed_commitment: str
    key: str
    mode: str
    complexity: int
    chords: List[ChordEvent]
    melody: List[MelodyNote]
    modulations: List[Dict[str, str]]
    cadence_map: List[int]

    def to_json(self) -> str:
        """Return the plan serialised to JSON."""
        def encode(obj):
            if isinstance(obj, (Chord, Voicing, ChordEvent, MelodyNote)):
                return asdict(obj)
            raise TypeError(f"Unserialisable: {obj}")

        data = asdict(self)
        return json.dumps(data, default=encode, indent=2)


# ---------------------------------------------------------------------------
# Music theory helpers ------------------------------------------------------
# ---------------------------------------------------------------------------

PC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NAME_TO_PC = {n: i for i, n in enumerate(PC_NAMES)}

# scale degree semitone offsets for common modes
MODES = {
    "ionian": [0, 2, 4, 5, 7, 9, 11],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "aeolian": [0, 2, 3, 5, 7, 8, 10],
    "locrian": [0, 1, 3, 5, 6, 8, 10],
}

# Functional chord families
T_CHORDS = ["I", "vi"]
PD_CHORDS = ["ii", "IV"]
D_CHORDS = ["V", "vii°"]

VOICE_RANGES = {
    "s": (60, 84),
    "a": (55, 77),
    "t": (48, 69),
    "b": (36, 55),
}

MELODY_RANGE = (62, 86)


def _pc_in_key(key_pc: int, degree: int, mode_steps: Sequence[int]) -> int:
    return (key_pc + mode_steps[degree]) % 12


def roman_numeral(degree: int, mode: str) -> str:
    """Return a basic roman numeral for the given scale degree."""
    major_like = {0, 3, 4}
    diminished = {6}
    if degree in diminished:
        return "vii°"
    if degree in major_like:
        return ["I", "II", "III", "IV", "V", "VI", "VII"][degree]
    return ["i", "ii", "iii", "iv", "v", "vi", "vii"][degree]


def build_chord(key_pc: int, roman: str, mode_steps: Sequence[int]) -> Chord:
    """Build chord definition from roman numeral."""
    if roman == "V/V":
        root_pc = (key_pc + 2) % 12
        pcs = [root_pc, (root_pc + 4) % 12, (root_pc + 7) % 12]
        return Chord("V/V", pcs, "D", "V/V")
    mapping = {"I": 0, "ii": 1, "iii": 2, "IV": 3, "V": 4, "vi": 5, "vii°": 6}
    deg = mapping[roman]
    pcs = [
        _pc_in_key(key_pc, deg, mode_steps),
        _pc_in_key(key_pc, (deg + 2) % 7, mode_steps),
        _pc_in_key(key_pc, (deg + 4) % 7, mode_steps),
    ]
    func = "T" if roman in T_CHORDS else "PD" if roman in PD_CHORDS else "D"
    return Chord(roman, pcs, func, roman_numeral(deg, ""))


def _nearest_pitch(pc: int, target: int, low: int, high: int) -> int:
    candidates = [pc + 12 * o for o in range(-2, 10)]
    candidates = [c for c in candidates if low <= c <= high]
    return min(candidates, key=lambda p: abs(p - target))


def voice_chord(pcs: Sequence[int], prev: Optional[Voicing]) -> Voicing:
    """Return a SATB voicing with minimal motion."""
    prev_vals = [
        prev.soprano if prev else None,
        prev.alto if prev else None,
        prev.tenor if prev else None,
        prev.bass if prev else None,
    ]
    voices = []
    for (low, high), prev_pitch in zip(VOICE_RANGES.values(), prev_vals):
        target = prev_pitch if prev_pitch is not None else (low + high) // 2
        pitch = _nearest_pitch(pcs[0], target, low, high)
        voices.append(pitch)
    return Voicing(*voices)


def generate_progression(
    prng: random.Random,
    length: int,
    key_pc: int,
    mode_steps: Sequence[int],
    complexity: int,
    modulate_to: Optional[str],
) -> Tuple[List[ChordEvent], List[Dict[str, str]]]:
    """Generate chord events and modulation metadata."""
    romans: List[str] = []
    state = "T"
    for _ in range(length - 2):
        if state == "T":
            state = prng.choices(["T", "PD", "D"], weights=[0.5, 0.3, 0.2])[0]
        elif state == "PD":
            state = prng.choices(["D", "T"], weights=[0.6, 0.4])[0]
        else:
            state = prng.choices(["T", "PD"], weights=[0.7, 0.3])[0]
        if state == "T":
            romans.append(prng.choice(T_CHORDS))
        elif state == "PD":
            romans.append(prng.choice(PD_CHORDS))
        else:
            if complexity > 2 and prng.random() < 0.2:
                romans.append("V/V")
            else:
                romans.append(prng.choice(D_CHORDS))
    romans.extend(["V", "I"])  # cadence

    modulations: List[Dict[str, str]] = []
    events: List[ChordEvent] = []
    modulated = False
    current_key_pc = key_pc
    for idx, rn in enumerate(romans):
        if not modulated and modulate_to and idx >= length // 2:
            current_key_pc = NAME_TO_PC[modulate_to]
            modulations.append({"index": str(idx), "key": modulate_to})
            modulated = True
        chord = build_chord(current_key_pc, rn, mode_steps)
        prev_voice = events[-1].voicing if events else None
        voicing = voice_chord(chord.pitch_classes, prev_voice)
        events.append(ChordEvent(chord, voicing, 4.0))
    return events, modulations


def generate_melody(prng: random.Random, chords: Sequence[ChordEvent]) -> List[MelodyNote]:
    notes: List[MelodyNote] = []
    time = 0.0
    prev_pitch = (MELODY_RANGE[0] + MELODY_RANGE[1]) // 2
    for ev in chords:
        pcs = ev.chord.pitch_classes
        for beat in range(4):
            if beat % 2 == 0:
                pc = prng.choice(pcs)
                pitch = _nearest_pitch(pc, prev_pitch, *MELODY_RANGE)
            else:
                step = prng.choice([-2, -1, 1, 2])
                pitch = min(max(prev_pitch + step, MELODY_RANGE[0]), MELODY_RANGE[1])
            velocity = prng.randint(68, 112)
            notes.append(MelodyNote(pitch, time, 1.0, velocity))
            prev_pitch = pitch
            time += 1.0
    return notes


# ---------------------------------------------------------------------------
# Generation orchestrator ---------------------------------------------------
# ---------------------------------------------------------------------------

class HarmonyError(Exception):
    """Raised for invalid user input."""


def _validate_args(args: argparse.Namespace) -> None:
    if args.key not in NAME_TO_PC:
        raise HarmonyError(f"Invalid key: {args.key}")
    if args.mode not in MODES:
        raise HarmonyError(f"Invalid mode: {args.mode}")
    if args.complexity < 1 or args.complexity > 5:
        raise HarmonyError("complexity must be between 1 and 5")
    if args.bucket not in {"short", "med", "long"}:
        raise HarmonyError("bucket must be short, med, or long")
    if args.modulate_to and args.modulate_to not in NAME_TO_PC:
        raise HarmonyError(f"Invalid modulation key: {args.modulate_to}")


def run_generation(args: argparse.Namespace) -> HarmonicPlan:
    _validate_args(args)
    seed_hex = args.seed or "".join(random.choice("0123456789abcdef") for _ in range(32))
    prng = random.Random(int(seed_hex, 16))
    commitment = blake2b(bytes.fromhex(seed_hex), digest_size=16).hexdigest()

    lengths = {"short": 4, "med": 8, "long": 12}
    length = lengths[args.bucket]
    key_pc = NAME_TO_PC[args.key]
    mode_steps = MODES[args.mode]

    events, modulations = generate_progression(
        prng, length, key_pc, mode_steps, args.complexity, args.modulate_to
    )
    melody = generate_melody(prng, events)
    cadence_map = [len(events) - 2]

    plan = HarmonicPlan(
        seed_commitment=commitment,
        key=args.key,
        mode=args.mode,
        complexity=args.complexity,
        chords=events,
        melody=melody,
        modulations=modulations,
        cadence_map=cadence_map,
    )
    return plan


def persist_plan(plan: HarmonicPlan, db_path: Path) -> None:
    conn = sqlite3.connect(db_path, timeout=30)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS harmony_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            seed_commitment TEXT,
            key TEXT,
            mode TEXT,
            complexity INTEGER,
            created TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS harmony_events (
            run_id INTEGER,
            idx INTEGER,
            chord TEXT,
            function TEXT,
            soprano INTEGER,
            alto INTEGER,
            tenor INTEGER,
            bass INTEGER
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS harmony_melody (
            run_id INTEGER,
            idx INTEGER,
            pitch INTEGER,
            start REAL,
            duration REAL,
            velocity INTEGER
        )
        """
    )
    cur.execute(
        "INSERT INTO harmony_runs(seed_commitment,key,mode,complexity) VALUES (?,?,?,?)",
        (plan.seed_commitment, plan.key, plan.mode, plan.complexity),
    )
    run_id = cur.lastrowid
    for idx, ev in enumerate(plan.chords):
        cur.execute(
            """
            INSERT INTO harmony_events(run_id,idx,chord,function,soprano,alto,tenor,bass)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            (
                run_id,
                idx,
                ev.chord.name,
                ev.chord.function,
                ev.voicing.soprano,
                ev.voicing.alto,
                ev.voicing.tenor,
                ev.voicing.bass,
            ),
        )
    for idx, note in enumerate(plan.melody):
        cur.execute(
            """
            INSERT INTO harmony_melody(run_id,idx,pitch,start,duration,velocity)
            VALUES (?,?,?,?,?,?)
            """,
            (run_id, idx, note.pitch, note.start, note.duration, note.velocity),
        )
    conn.commit()
    conn.close()


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="harmony_engine",
        description="Generate harmony and melody plans",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen", help="generate harmonic plan")
    g.add_argument("--bucket", default="med")
    g.add_argument("--key", default="C")
    g.add_argument("--mode", default="ionian")
    g.add_argument("--complexity", type=int, default=1)
    g.add_argument("--seed")
    g.add_argument("--modulate-to")
    g.add_argument("--json-out")
    g.add_argument("--stdout-json", action="store_true")
    g.add_argument("--db")
    g.add_argument("-v", action="count", default=0)

    args = parser.parse_args(argv)
    setup_logging(args.v)

    if args.cmd == "gen":
        try:
            plan = run_generation(args)
        except HarmonyError as exc:
            logging.error(str(exc))
            sys.exit(1)
        except Exception as exc:  # pragma: no cover - defensive
            logging.error(str(exc))
            sys.exit(1)
        json_text = plan.to_json()
        if args.json_out:
            try:
                Path(args.json_out).write_text(json_text)
            except OSError as exc:
                logging.error(f"Failed to write JSON: {exc}")
        if args.stdout_json:
            sys.stdout.write(json_text)
        if args.db:
            try:
                persist_plan(plan, Path(args.db))
            except sqlite3.Error as exc:
                logging.error(f"DB error: {exc}")
    else:  # pragma: no cover - no other subcommands
        parser.error("unknown command")


if __name__ == "__main__":
    main()

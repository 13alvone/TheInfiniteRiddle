#!/usr/bin/env python3
"""Top-level orchestrator for The Infinite Riddle."""
import argparse
import datetime as dt
import json
import logging
import math
import sqlite3
from pathlib import Path
from typing import List, Optional, Tuple

from .core import (
    setup_logging,
    blake2b_digest,
    gather_entropy,
    domain_prngs,
    pick_theme,
    pick_key,
    scale_pitches,
    sample_duration_seconds,
    pick_form,
    pick_time_signature,
    build_melody,
    build_bass,
    build_perc,
    NOTE_NAMES_SHARP,
    pitch_class_to_sf,
)
from .io import MidiBuilder, sha256_of_file, seed_commitment, timestamp_now_utc
from .synth import (
    render_audio, mythic_backmask, mythic_ashen_bitcrush,
    mythic_mirrorsalt_ms, mythic_liminal_bed, mythic_cipherspray_watermark,
)
from .vault import ensure_vault, vault_insert_run, vault_insert_artifact
from .qa import measure_peak, measure_rms, validate_lufs

__version__ = "0.2"

__all__ = [
    "__version__",
    "setup_logging", "blake2b_digest", "gather_entropy", "domain_prngs",
    "pick_theme", "pick_key", "scale_pitches", "sample_duration_seconds",
    "pick_form", "pick_time_signature", "pick_time_signatures",
    "build_melody", "build_bass", "build_perc", "NOTE_NAMES_SHARP",
    "pitch_class_to_sf",
    "MidiBuilder", "sha256_of_file",
    "seed_commitment", "timestamp_now_utc", "render_audio", "mythic_backmask",
    "mythic_ashen_bitcrush", "mythic_mirrorsalt_ms", "mythic_liminal_bed",
    "mythic_cipherspray_watermark", "ensure_vault", "vault_insert_run",
    "vault_insert_artifact", "parse_args", "_resolve_artifact_paths",
    "measure_peak", "measure_rms", "validate_lufs", "run_riddle",
]


def valid_seed(value: str) -> str:
    """Validate hexadecimal seed input for CLI."""
    if len(value) != 32:
        raise argparse.ArgumentTypeError("seed must be 32 hex characters")
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seed must be hexadecimal") from exc
    return value.lower()


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="The Infinite Riddle — generate unique WAV/MIDI artifacts with provenance.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser(
        "generate",
        help="generate WAV/MIDI artifacts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g.add_argument("theme", choices=["glass", "salt", "auto"], help="Select theme or auto.")
    g.add_argument("outdir", help="Output directory for artifacts.")
    g.add_argument("--db", default="riddle_vault.db", help="Path to SQLite vault.")
    g.add_argument("--bucket", choices=["short", "med", "long"], default=None, help="Force duration bucket.")
    g.add_argument("--stems", action="store_true", help="Render individual stems alongside mix.")
    g.add_argument("--mythic-max", type=int, default=2, help="Max mythic variants to attempt.")
    g.add_argument("--lufs-target", type=float, default=-14.0, help="Target loudness metadata.")
    g.add_argument("--seed", type=valid_seed, help="Hex seed for reproducible runs.")
    g.add_argument("-v", "--verbose", action="count", default=1, help="Increase verbosity (-v or -vv).")

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


def pick_time_signatures(prng, theme: str, form_nodes: List[str]):
    """Pick an odd-meter signature for each form node."""
    sigs = []
    for _ in form_nodes:
        ts = pick_time_signature(prng, theme)
        if isinstance(ts, list):
            ts = ts[0]
        sigs.append(ts)
    return sigs


def run_riddle(
    theme_req: Optional[str],
    outdir: Path,
    db_path: Path,
    duration_bucket: Optional[str],
    stems: bool,
    mythic_max: int,
    lufs_target: float,
    seed_hex: Optional[str],
    verbosity: int,
) -> None:
    try:
        setup_logging(verbosity)
        logging.info("[i] Starting The Infinite Riddle v%s", __version__)
        with sqlite3.connect(str(db_path)) as conn:
            ensure_vault(conn)

            if seed_hex:
                try:
                    root_seed = bytes.fromhex(seed_hex)
                except ValueError:
                    logging.error("[!] Seed must be 32 hex characters")
                    return
            else:
                entropy = gather_entropy()
                root_seed = blake2b_digest(entropy)
            commit = seed_commitment(root_seed)
            prngs = domain_prngs(root_seed)

            theme = pick_theme(prngs["names"], theme_req)
            total_sec = sample_duration_seconds(prngs["form"], theme, duration_bucket)
            form_nodes, timeline = pick_form(prngs["form"], theme, total_sec)
            logging.info("[i] Theme=%s Duration=%ds Form=%s", theme, total_sec, "→".join(form_nodes))

            if theme == "glass":
                bpm_base = 96.0 + (prngs["rhythm"].uniform() - 0.5) * 30.0
            else:
                bpm_base = 64.0 + (prngs["rhythm"].uniform() - 0.5) * 24.0
            bpm_base = max(36.0, min(160.0, bpm_base))

            key_root, mode_name = pick_key(prngs["harmony"], theme)
            key_pc_name = NOTE_NAMES_SHARP[key_root]
            scale_pcs = scale_pitches(key_root, mode_name)
            sigil_pcs = prngs["melody"].choice([scale_pcs, scale_pcs[: max(2, len(scale_pcs) // 2)]])
            logging.info("[i] Key=%s %s; BPM≈%.1f", key_pc_name, mode_name, bpm_base)

            ppq = 480
            time_sigs = pick_time_signatures(prngs["rhythm"], theme, form_nodes)
            sections: List[Tuple[int, int, int]] = []
            section_starts: List[int] = []
            tick_accum = 0
            for (node, start, end), (ts_num, ts_den) in zip(timeline, time_sigs):
                dur = end - start
                beats_per_bar = ts_num * (4 / ts_den)
                bars = max(1, int(round((dur / 60.0) * (bpm_base / beats_per_bar))))
                sections.append((bars, ts_num, ts_den))
                section_starts.append(tick_accum)
                ticks_per_bar = ppq * ts_num * 4 // ts_den
                tick_accum += bars * ticks_per_bar

            lead_events = build_melody(
                prngs["melody"],
                scale_pcs,
                sections,
                ppq,
                density=0.35 if theme == "glass" else 0.25,
                base_octave=5 if theme == "glass" else 4,
                sigil_pcs=sigil_pcs,
            )
            pad_events = build_melody(
                prngs["melody"],
                scale_pcs,
                sections,
                ppq,
                density=0.20 if theme == "glass" else 0.18,
                base_octave=4,
                sigil_pcs=sigil_pcs,
            )
            bass_events = build_bass(prngs["melody"], scale_pcs, sections, ppq)
            perc_events = build_perc(prngs["rhythm"], sections, ppq, theme)

            mb = MidiBuilder(ppq)
            trk0 = mb.new_track()
            mb.tempo(trk0, 0, bpm_base)
            first_num, first_den = time_sigs[0]
            mb.time_signature(trk0, 0, first_num, int(math.log2(first_den)))
            mb.key_signature(trk0, 0, pitch_class_to_sf(key_root), 0)
            last_tick = 0
            last_sig = (first_num, first_den)
            for tick, (ts_num, ts_den) in zip(section_starts[1:], time_sigs[1:]):
                if (ts_num, ts_den) != last_sig:
                    delta = tick - last_tick
                    mb.time_signature(trk0, delta, ts_num, int(math.log2(ts_den)))
                    last_tick = tick
                    last_sig = (ts_num, ts_den)

            def write_track(events, ch, program_hint):
                trk = mb.new_track()
                mb.prog_change(trk, 0, ch, program_hint)
                events_sorted = sorted(events, key=lambda x: x[0])
                last_tick = 0
                for tick, dur, note, vel in events_sorted:
                    delta = max(0, tick - last_tick)
                    mb.note_on(trk, delta, ch, note, vel)
                    mb.note_off(trk, dur, ch, note, max(32, vel // 2))
                    last_tick = tick + dur
                mb.end_track(trk, 0)

            write_track(lead_events, 0, 11)
            write_track(pad_events, 1, 89)
            write_track(bass_events, 2, 34)
            write_track(perc_events, 9, 0)
            mb.end_track(trk0, 0)

            now = dt.datetime.now()
            seed8 = commit[:8]
            form_str = "→".join(form_nodes)
            keymode = f"{NOTE_NAMES_SHARP[key_root]}_{mode_name}"
            mm = total_sec // 60
            ss = total_sec % 60
            len_tag = f"{mm:02d}-{ss:02d}"
            base = (
                f"{now:%Y%m%d_%H%M%S}_RIDDLE_{'Glass' if theme=='glass' else 'Salt'}_"
                f"{form_str}_BPM{int(round(bpm_base))}_{keymode}_LEN{len_tag}_{seed8}"
            )

            outdir.mkdir(parents=True, exist_ok=True)
            json_path = outdir / f"{base}.riddle.json"
            midi_path = outdir / f"{base}_MIDI_{seed8}.mid"
            wav_path = outdir / f"{base}.wav"
            stem_paths = {}
            if stems:
                stem_paths = {
                    "lead": outdir / f"{base}_STEM_LEAD_{seed8}.wav",
                    "pad": outdir / f"{base}_STEM_PAD_{seed8}.wav",
                    "bass": outdir / f"{base}_STEM_BASS_{seed8}.wav",
                    "perc": outdir / f"{base}_STEM_PERC_{seed8}.wav",
                }

            mb.save(midi_path)
            logging.info("[i] MIDI written: %s", midi_path.name)

            midi_dict = {"lead": lead_events, "pad": pad_events, "bass": bass_events, "perc": perc_events}
            render_audio(
                wav_path,
                midi_dict,
                bpm_base,
                ppq,
                48000,
                total_sec,
                prngs["synth"],
                stem_paths=stem_paths if stems else None,
            )
            logging.info("[i] WAV written: %s", wav_path.name)
            if stems:
                for p in stem_paths.values():
                    logging.info("[i] Stem written: %s", p.name)
            peak_db = measure_peak(wav_path)
            rms_db = measure_rms(wav_path)
            crest_db = peak_db - rms_db
            if not validate_lufs(rms_db, lufs_target):
                logging.warning("[!] LUFS target %.1f dB, measured %.1f dB", lufs_target, rms_db)

            sidecar = {
                "seed_commitment": commit,
                "theme": theme,
                "form_nodes": form_nodes,
                "form_path": form_str,
                "durations": [{"node": n, "start": float(s), "end": float(e)} for (n, s, e) in timeline],
                "bpm_base": bpm_base,
                "time_sigs": [f"{n}/{d}" for n, d in time_sigs],
                "key_mode": {"root_pc": key_root, "root_name": key_pc_name, "mode": mode_name},
                "sigil_pcs": sigil_pcs,
                "artifact_hashes": {
                    "midi": sha256_of_file(midi_path),
                    "wav": sha256_of_file(wav_path),
                    **({"stems": {k: sha256_of_file(p) for k, p in stem_paths.items()}} if stems else {}),
                },
                "started_utc": timestamp_now_utc(),
                "loudness_target": lufs_target,
                "crest_factor_est": crest_db,
                "true_peak_est": peak_db,
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(sidecar, f, indent=2)
            logging.info("[i] Sidecar JSON written: %s", json_path.name)

            run_obj = {
                "started_utc": sidecar["started_utc"],
                "theme": theme,
                "seed_commitment": commit,
                "duration_sec": total_sec,
                "form_path": form_str,
                "bpm_base": bpm_base,
                "lufs_target": lufs_target,
                "coherence": 0.7 if theme == "glass" else 0.6,
                "presence": 0.5,
                "hostility": 0.2,
                "obliquity": 0.6 if theme == "glass" else 0.7,
                "time_sigs": [f"{n}/{d}" for n, d in time_sigs],
                "sections": [
                    {
                        "key_root": key_pc_name,
                        "mode": mode_name,
                        "start_sec": float(s),
                        "end_sec": float(e),
                        "time_sig": f"{time_sigs[i % len(time_sigs)][0]}/{time_sigs[i % len(time_sigs)][1]}",
                        "euclid": "E(5,8)|E(7,12)" if theme == "glass" else "E(3,8)|E(5,12)",
                        "ca_rule": 90 if theme == "glass" else 150,
                        "swing": 0.06 if theme == "glass" else -0.04,
                    }
                    for i, (n, s, e) in enumerate(timeline)
                ],
            }
            run_id = vault_insert_run(conn, run_obj)
            vault_insert_artifact(conn, run_id, "json", json_path, 0.0, None, None, None)
            vault_insert_artifact(conn, run_id, "midi", midi_path, total_sec, bpm_base, keymode, None)
            vault_insert_artifact(conn, run_id, "wav", wav_path, total_sec, bpm_base, keymode, None)
            if stems:
                for name, path in stem_paths.items():
                    vault_insert_artifact(conn, run_id, "stem", path, total_sec, bpm_base, keymode, name)

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
                if prng_m.uniform() < 0.22:
                    dst = outdir / f"{base}_MYTHIC_{label}_{seed8}.wav"
                    try:
                        fn(wav_path, dst)
                        vault_insert_artifact(conn, run_id, "mythic", dst, total_sec, bpm_base, keymode, label)
                        logging.info("[i] Mythic generated: %s", dst.name)
                        rolled += 1
                    except Exception as e:
                        logging.error("[x] Mythic %s failed: %s", label, e)

            logging.info("[i] Completed run. SeedCommit=%s", commit[:16])
    except Exception as e:
        logging.exception("[x] Fatal error: %s", e)
        raise

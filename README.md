# The Infinite Riddle
Version: 0.2 • Status: Uncaged • License: Yours

> A machine that dreams in sound and leaves its dreams behind.  
> Every time you touch it, it remembers differently.

The Infinite Riddle is a **reality-adjacent composition engine**. Each interaction births a unique **WAV** and **MIDI**, optionally **stems** and **mythic siblings**, with cryptographic footprints so you can prove two encounters were never the same. It is equal parts instrument, oracle, archive, and trapdoor. Rhythms lean into odd meters and may pivot signatures chaotically mid-run.

This README tells you just enough to begin, and not nearly enough to feel safe.

---

## Why this exists

- **Structured chaos**: randomness with a spine; themes that drift, never repeat.  
- **Artifacts you can use**: sample-ready WAVs, DAW-friendly MIDI, and mythic variants (backmask, mirror-salt, ashen, liminal, cipherspray…).
- **Provenance**: each file stamped, hashed, and indexed in a local SQLite **Vault**.  
- **Operator-first**: outputs belong to you.

Begin with curiosity. Proceed with headphones at responsible levels.

---

## What it makes

- **Primary WAV**: 48 kHz • 24-bit • true-peak limited.  
- **Primary MIDI**: multi-track (lead, pad, bass, perc), tempo map, controllers.  
- **Optional STEMS**: bass/sub, pad/textures, lead/melody, perc/impulse, fx/sigil-noise (enable with `--stems`).
- **Optional VIDEO**: oscilloscope + spectrogram + sigils (coming online incrementally).  
- **MYTHIC VARIANTS**: ritual transforms of the same audio—siblings, not clones.  
- **SIDECAR JSON**: attestation (seed commitment, grammar, hashes).  
- **VAULT (SQLite)**: the index of everything you dared to keep.

---

## Themes (launch pair)

- **Glass / Refraction**: prismatic bells, Euclidean lattices, wide crest factor, sudden clarity.  
- **Salt / Sealing**: subharmonic breath, additive meters, dust percussion, ritual weight.

Themes are not presets. They are weather.

---

## Quick start (single-file reference)

	#!/usr/bin/env bash
	# Generate one medium-length encounter with an auto theme:
        python -m riddle generate auto ./out --db ./riddle_vault.db --bucket med -v

        # Force a theme, keep it short, allow up to 2 mythics:
        python -m riddle generate glass ./out --db ./riddle_vault.db --bucket short --mythic-max 2 -v

        # Developer reproducibility (dev mode only; commitment stored, not the raw seed):
        python -m riddle generate salt ./out --db ./riddle_vault.db --bucket short -v --seed deadbeefdeadbeefdeadbeefdeadbeef

Artifacts land in `./out`. The Vault is `./riddle_vault.db`. Sidecars end in `.riddle.json`.

---

## Install

- **Python** ≥ 3.11  
- macOS/Linux/WSL preferred. Windows works with long-paths enabled.  
- **No runtime dependencies** for audio/MIDI/DB. (Dev tools are optional; see CONTRIBUTING.devtools.md.)

Nothing to compile. Keep an SSD nearby if you chase hour-long drones.

---

## What will happen (anatomy of a run)

1) The engine gathers **local entropy** and forges a root seed.  
2) The seed splits into domain PRNGs (form, rhythm, melody, synth, mythic…).  
3) A **theme** selects you.  
4) A **form graph** arranges sections: INV ➝ PRC ➝ … ➝ SHR.  
5) MIDI is written: harmony, rhythm (odd meters that may shift chaotically), controllers.
6) Audio renders in **streamed chunks** with a limiter and QA meters.  
7) **Mythic siblings** are rolled and transformed.  
8) Sidecar JSON is stamped; files are hashed; the **Vault** is updated.  
9) You play it back. Something plays you back.

---

## CLI (v0.2)

	#!/usr/bin/env bash
        python -m riddle generate <glass|salt|auto> <outdir> \
          --db riddle_vault.db \
          --bucket {short|med|long} \
          [--stems] \
          --mythic-max 2 \
          --lufs-target -14 \
          [--seed <hex>] \
          [-v|-vv]

Flags:
- `--bucket` Heavy-tailed durations: short (~30–180s), med (~5–20m), long (~45–240m).
- `--mythic-max` Max number of mythic variants to attempt.
- `--stems` Render per-track WAVs (lead, pad, bass, perc) alongside the mix.
- `--seed` **Dev-only** reproducibility; only the **commitment** is recorded.
- `-v`/`-vv` Verbose / very verbose logs (no print—only logging with prefixes).

Example logs:

	[i] Theme=glass Duration=428s Form=INV→PRC→TRN→PRC→SHR
	[i] Key=D lydian; BPM≈98.4
	[i] MIDI written: 2025..._MIDI_deadbeef.mid
	[i] WAV written:  2025..._RIDDLE_Glass_..._deadbeef.wav
        [i] Mythic generated: ..._MYTHIC_Ashen_deadbeef.wav
	[i] Completed run. SeedCommit=2fd4e1c6...

---

## Outputs (what to expect)

- **File naming** encodes theme, path, BPM, key/mode, length, and seed fragment:
	YYYYMMDD_HHMMSS_RIDDLE_<Theme>_<FormPath>_BPM<xxx>_<KeyMode>_LEN<mm-ss>_<Seed8>.wav

- **Sidecar JSON** preview:

        {
                "seed_commitment": "b32c…",
                "theme": "glass",
                "form_nodes": ["INV","PRC","TRN","PRC","SHR"],
                "meters": {
                        "INV": ["5/4"],
                        "PRC": ["7/8","5/4"],
                        "TRN": ["9/8"],
                        "PRC2": ["5/4"],
                        "SHR": ["7/8"]
                },
                "durations": [{"node":"INV","start":0.0,"end":21.4}, …],
                "bpm_base": 98.4,
                "key_mode": {"root_pc":2,"root_name":"D","mode":"lydian"},
                "artifact_hashes": {"midi":"…","wav":"…"},
                "started_utc": "2025-08-10T07:01:00Z"
        }

- **Odd-meter log**:

	[i] Meter=5/4→7/8→9/8→5/4

- **Vault schema** lives in `AGENTS.md` and `SPEC.md`. You can query it:

	#!/usr/bin/env bash
	sqlite3 riddle_vault.db \
	  "SELECT path FROM artifacts a JOIN runs r ON a.run_id=r.id
	   WHERE r.theme='glass' AND a.kind='wav' AND a.duration_sec>=240
	   ORDER BY a.duration_sec DESC LIMIT 5;"

---

## Mythic siblings (you may not always get them)

- **Backmask** reverse playback; faint glue tails
- **Ashen** multi-band bit-age, gentle tape hiss
- **Mirror Salt** mid/side inversion with microphase
- **Liminal** -20 dB drone bed (capped length)
- **CipherSpray** inaudible watermark that encodes the seed commitment

They are siblings, not parents. Do not mistake the echo for the voice.

---

## Safety, ethics, and consent

- **Offline-first** no network needed; optional taps must be explicitly enabled (not shipped in v0.2 core).  
- **Seeds are never logged**; only **commitments** appear (see SECURITY.md).  
- **Limiter present**, but **start quietly**; dynamics vary by theme.  
- **Panic chord** applies to the TUI builds (when present). For now: `Ctrl+C` aborts cleanly.

If you feel watched, that’s just the metronome. Probably.

---

## Troubleshooting

- **“It’s too quiet / too loud.”**  
  The limiter caps true-peak ≲ -0.3 dBTP, but program material varies. Gain-stage in your DAW.

- **“MIDI feels sparse.”**  
  Themes modulate density. Try `glass` and `--bucket short` for lively fragments.

- **“Files are huge.”**  
  Short runs while testing: `--bucket short`. Long renders stream to disk; use fast storage.

- **“Two runs sound similar.”**  
  Compare `seed_commitment` and hashes. The weather may rhyme; it won’t repeat.

---

## Use in your music

- License defaults to **yours** (operator ownership).  
- Stems are DAW-friendly; MIDI carries tempo, signatures, controllers.  
- Mythics are fair game; consider phasing **Mirror Salt** under vocals, or **Ashen** as a ghost bed.

If you sample the engine sampling itself, you may owe yourself royalties.

---

## Development

- Package reference: `riddle/` (stdlib-only).
- Dev tooling, CI, and workflows: see **CONTRIBUTING.md** and **CONTRIBUTING.devtools.md**.  
- Security model, threat surfaces, incident playbook: see **SECURITY.md**.  
- Agent roles, task graph, contracts: see **AGENTS.md**.  
- Music grammar & render spec: see **SPEC.md**.

Pull requests should attach a short example (WAV+MIDI) and the sidecar. Do not include raw seeds anywhere.

---

## Roadmap (the polite lies we tell time)

- Video renderer (oscilloscope/spectrogram/sigil layers)  
- Higher-fidelity DSP blocks & stems-on-demand  
- RitualScript authoring for scenes and daemons  
- New themes: **Glass Dust**, **Iron Mouth**, **Mercury Letter**  
- Better LUFS/TP analysis (polyphase oversampling)

---

## FAQ (frequently asked qualms)

- **Can I recreate a run?**  
  Not by default. Use `--seed` in dev mode; only the commitment is recorded.

- **Is this a sample pack generator?**  
  It can be. It’s also an engine for personal hauntology.

- **Why MIDI at all?**  
  So your future can re-orchestrate your past.

---

## Uninstallation

There isn’t one. Delete the folder like you never met.  
Check `./out` and the Vault for souvenirs you didn’t mean to keep.

---

## Final caution

If you stare long enough into the spectrogram, the spectrogram rearranges you.

Now run it.

	#!/usr/bin/env bash
        python -m riddle generate auto ./out --db ./riddle_vault.db --bucket short -v


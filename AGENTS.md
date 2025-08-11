# AGENTS.md — The Infinite Riddle (Collaborator Playbook)
Version: 0.2 (Collaboration-Ready)
Status: Living Document
Owner: The Infinite Riddle Core
Audience: Human + Software Agents contributing to generation, QA, archiving, and release flows.

--------------------------------------------------------------------------------
## 0) Purpose

This document defines how collaborating AGENTS (human or automated) extend, operate, and audit **The Infinite Riddle** system. It establishes:
- roles & responsibilities,
- task graph & interfaces,
- data contracts (files, JSON, DB),
- operational guardrails (privacy, ethics),
- QA gates & observability,
- branching/versioning workflow,
- reproducibility and provenance rules.

All examples below use **tabs** to indicate code blocks.

--------------------------------------------------------------------------------
## 1) System Ethos (Non-Negotiables)

- **Unrepeatable yet coherent**: every run is unique, thematically bound.
- **Operator-first**: outputs belong to the local operator by default.
- **Consent & privacy**: no network/mic/cam taps without explicit opt-in; air-gapped mode always supported.
- **Provable lineage**: every artifact is hash-stamped and traceable via the Vault (SQLite) + sidecar JSON.
- **Useful artifacts**: every interaction yields **WAV** and **MIDI**, optional video/stems, plus occasional **mythic** siblings.
- **Observability**: QA metrics & logs must be informative without leaking secrets.

--------------------------------------------------------------------------------
## 2) Agent Roles & Responsibilities

### 2.1 Core Agent Types
- **Planner**  
  Decides session parameters (theme, length bucket, form graph) within policy bounds. Ensures seeds/PRNG separation.
- **Composer**  
  Generates harmony, rhythm, melody events; emits MIDI; assigns controller curves.
- **Renderer**  
  Renders audio (master + stems), applies limiter & QA meters; produces optional video frames; streams to disk safely.
- **Mythicizer**  
  Creates variant siblings (Backmask, Ashen, MirrorSalt, Liminal, CipherSpray) with lineage-safe transforms.
- **Archivist**  
  Writes sidecar JSON, computes hashes, populates Vault, and enforces naming conventions & retention policy.
- **Auditor**  
  Validates QA gates, checks headroom/phase/true-peak/RMS floor; can quarantine failed outputs.
- **Orchestrator**  
  Runs the task graph, enforces rate limits, handles retries/back-pressure, triggers notifications.
- **Custodian (Security/Ethics)**  
  Enforces consent gates, offline mode, policy compliance, and redlines forbidden behaviors.

### 2.2 RACI (Simplified)
- **R** (Responsible): executing; **A** (Accountable): final owner; **C** (Consulted); **I** (Informed)

| Task | Planner | Composer | Renderer | Mythicizer | Archivist | Auditor | Orchestrator | Custodian |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Seed & PRNG split | R | I | I | I | C | I | A | C |
| Theme/Length/Form | A | C | I | I | I | I | R | I |
| MIDI generation | I | A | I | I | I | C | R | I |
| Audio render | I | C | A | I | C | C | R | I |
| Mythic variants | I | I | C | A | C | C | R | I |
| Sidecar & Vault | I | I | I | C | A | C | R | I |
| QA gates | I | C | C | C | C | A | R | I |
| Policy/Consent | I | I | I | I | I | I | C | A |

--------------------------------------------------------------------------------
## 3) Task Graph (Reference Flow)

- **Inputs**: local entropy bundle → root seed → domain PRNGs
- **Decisions**: theme ∈ {Glass, Salt}, duration bucket ∈ {short, med, long}, form path
- **Outputs**: WAV (master), MIDI (type-1), optional stems, optional MP4, mythic siblings, sidecar JSON, Vault rows

	# Minimal orchestration graph (YAML-esque)
	pipeline:
	  - id: plan
	    agent: planner
	    out: {theme, duration_sec, form_nodes, bpm_base, key_mode}
	  - id: compose
	    needs: [plan]
	    agent: composer
	    out: {midi_events, controller_curves}
	  - id: render
	    needs: [compose]
	    agent: renderer
	    out: {wav_master, stems?, meters}
	  - id: mythic
	    needs: [render]
	    agent: mythicizer
	    out: {mythics[]}
	  - id: archive
	    needs: [render, mythic]
	    agent: archivist
	    out: {sidecar_json, vault_rows}
	  - id: audit
	    needs: [archive]
	    agent: auditor
	    decision: {accept|quarantine}
	  - id: notify
	    needs: [audit]
	    agent: orchestrator

--------------------------------------------------------------------------------
## 4) Interfaces & Contracts

### 4.1 CLI Contracts (v0.2 Core)
- **Generate** one session:
	#!/usr/bin/env bash
	riddle generate auto ./out \
	  --db ./riddle_vault.db \
	  --bucket med \
	  --mythic-max 2 \
	  --lufs-target -14 \
	  -vv

- **Search** artifacts (sample filters; Orchestrator may add richer queries):
	#!/usr/bin/env bash
	riddle search --db ./riddle_vault.db --theme glass --min-bpm 80 --max-bpm 100 --min-sec 240

### 4.2 File Naming (must match)
	YYYYMMDD_HHMMSS_RIDDLE_<Theme>_<FormPath>_BPM<xxx>_<KeyMode>_LEN<mm-ss>_<Seed8>.wav
	YYYYMMDD_HHMMSS_RIDDLE_<Theme>_MIDI_<Seed8>.mid
	YYYYMMDD_HHMMSS_RIDDLE_<Theme>_VIDEO_<Seed8>.mp4
	YYYYMMDD_HHMMSS_RIDDLE_<Theme>_STEMS_<part>_<Seed8>.wav
	YYYYMMDD_HHMMSS_RIDDLE_<Theme>_MYTHIC_<Type>_<Seed8>.wav
	YYYYMMDD_HHMMSS_RIDDLE_<Theme>_<Seed8>.riddle.json

### 4.3 Sidecar JSON (core fields)
	{
		"seed_commitment": "<hex blake2b>",
		"theme": "glass|salt",
		"form_nodes": ["INV","PRC","..."],
		"form_path": "INV→PRC→...→SHR",
		"durations": [{"node":"INV","start":0.0,"end":21.4}, ...],
		"bpm_base": 96.0,
		"key_mode": {"root_pc":2,"root_name":"D","mode":"lydian"},
		"sigil_pcs": [2,6,9],
		"artifact_hashes": {"midi":"<sha256>","wav":"<sha256>"},
		"started_utc":"2025-08-10T07:01:00Z",
		"loudness_target": -14.0,
		"crest_factor_est": null,
		"true_peak_est": null
	}

### 4.4 Vault (SQLite) — Core Schema

	-- runs / sessions
	CREATE TABLE runs (
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

	-- musical metadata per section
	CREATE TABLE run_harmony (
	  run_id INTEGER REFERENCES runs(id),
	  section_idx INTEGER,
	  key_root TEXT,
	  mode TEXT,
	  start_sec REAL,
	  end_sec REAL,
	  PRIMARY KEY (run_id, section_idx)
	);

	CREATE TABLE run_rhythm (
	  run_id INTEGER REFERENCES runs(id),
	  section_idx INTEGER,
	  time_sig TEXT,
	  euclid TEXT,
	  ca_rule INTEGER,
	  swing REAL,
	  PRIMARY KEY (run_id, section_idx)
	);

	-- artifacts catalog
	CREATE TABLE artifacts (
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

	CREATE INDEX idx_artifacts_kind ON artifacts(kind);
	CREATE INDEX idx_runs_theme ON runs(theme);

### 4.5 Example Queries
- *All Glass > 4 minutes between 80–100 BPM*:
	-- min-sec=240
	SELECT a.path, a.duration_sec, a.bpm_est
	FROM artifacts a
	JOIN runs r ON a.run_id=r.id
	WHERE r.theme='glass' AND a.duration_sec>=240
	  AND a.bpm_est BETWEEN 80 AND 100
	  AND a.kind='wav'
	ORDER BY a.duration_sec DESC;

- *Find WAVs in D Lydian with mythic=Shatter*:
	SELECT a.path
	FROM artifacts a
	JOIN runs r ON a.run_id=r.id
	JOIN run_harmony h ON h.run_id=r.id
	WHERE a.kind IN ('wav','mythic')
	  AND a.mythic_type='Shatter'
	  AND h.key_root='D' AND h.mode='lydian';

--------------------------------------------------------------------------------
## 5) Logging & Error Policy

- **Prefixes**:  
  `logging.info` → “[i] …”  
  `logging.warning` → “[!] …”  
  `logging.debug` → “[DEBUG] …”  
  `logging.error` → “[x] …”

- **Examples**
	#!/usr/bin/env python3
	import logging
	logging.info("[i] Theme=%s Duration=%ds Form=%s", theme, total_sec, "→".join(form_nodes))
	logging.warning("[!] Falling back to default limiter settings")
	logging.debug("[DEBUG] PRNG splits established for domains=%s", list(prngs.keys()))
	logging.error("[x] Mythic %s failed: %s", label, exc)

- **Basic error handling**  
  - Fail a stage → attempt localized retry (idempotent where possible).  
  - Catastrophic failure → write partial sidecar with `"failed_stage": "<name>"` and quarantine outputs.  
  - Never drop logs for seed/PRNG digests; store commitments, not raw secrets.

--------------------------------------------------------------------------------
## 6) Determinism, Seeds, and Reproducibility

- Default runs are **unrepeatable** (entropy bundle).  
- Dev mode may accept `--seed <hex>` to reproduce; this must:  
  - populate **domain PRNGs** via HKDF,  
  - print `seed_commitment` to logs,  
  - mark sidecar with `"reproducible": true`.

- **Domain separation**: `k_form`, `k_harmony`, `k_rhythm`, `k_melody`, `k_synth`, `k_ctrl`, `k_video`, `k_mythic`, `k_noise`, `k_names`.  
- **Rule**: Agents **must not** share PRNG instances across domains.

--------------------------------------------------------------------------------
## 7) Quality Gates (Auditor Checklist)

- **No-silence**: RMS floor ≥ −45 dBFS; if not, inject minimal bed & re-limit.
- **Clipping guard**: pre-limit ≤ −1.0 dBFS; post-limit true-peak ≤ −0.3 dBTP.
- **Phase**: |mid/side correlation| ≤ 0.98 sustained; narrow width if violated.
- **MIDI utility**: at least one track per section with note density > 0.2 notes/sec.
- **Metadata completeness**: sidecar and Vault rows present; hashes match filesystem.
- **Mythic safety**: enforce forbidden pairs (Backmask↔Liminal, MirrorSalt↔Ashen).

- **Quarantine procedure**
	#!/usr/bin/env bash
	mkdir -p ./quarantine
	mv ./out/*_Seed8bad* ./quarantine/
	echo "[!] Quarantined 3 artifacts due to phase/TP violations" >> ./audit.log

--------------------------------------------------------------------------------
## 8) Mythic Variants (Policy & Combinations)

- Roll up to `mythic_max` siblings per run.  
- Recommended probabilities (independent rolls):  
  Backmask .28, Seal Encoding .22, Chalice .18, MirrorSalt .14, Shatter .12, Ashen .11, Liminal .07, CipherSpray .20

- **Forbidden combos**  
  - Backmask with Liminal (redundant time intent + risk of silent stretches)  
  - MirrorSalt with Ashen (phase integrity under heavy quantization)

- **Post-transform QA**  
  Re-check headroom/TP; apply limiter if necessary; re-hash; update Vault.

--------------------------------------------------------------------------------
## 9) Security, Privacy, and Ethics

- **Consent gates** for any external taps. Defaults OFF.  
- **Air-gap mode** must be feature-complete (no network dependencies).  
- **No sensitive data** storage; sidecar holds commitments & hashes only.  
- **Operator license**: generated outputs belong to the local operator unless overridden.  
- **Panic chord** (e.g., `Ctrl+Alt+~` in TUI): freezes daemons and displays status pane.  
- **Do Not**: exfiltrate local file contents, modify external systems, scare via harm; lean on ambiguity, symbolism, and timing.

--------------------------------------------------------------------------------
## 10) Project Structure (Suggested)

	/infinite-riddle/
	  AGENTS.md                 # this file
	  SPEC.md                   # Audio/AV & Grammar spec (design-freeze excerpts)
	  SECURITY.md               # extended threat model & audits
	  CONTRIBUTING.md           # coding & review discipline
	  LICENSE
	  riddle/                   # python package
	    __main__.py             # CLI entry
	    core/                   # seeds, PRNGs, forms, engines
	    synth/                  # DSP blocks, limiter, mythic transforms
	    io/                     # wav/midi/json, video (later)
	    vault/                  # sqlite adapters, queries
	    qa/                     # meters, validators
	  tests/
	    test_smoke.py
	    test_qagates.py
	  out/                      # artifacts
	  tools/                    # helper scripts (e.g., batch search/export)

--------------------------------------------------------------------------------
## 11) Branching, Reviews, and Versioning

- **Branching**: `main` (stable), `dev/*` (feature), `exp/*` (experiments, can be squashed).  
- **PR template** must include: seed_commitment (or “unrepeatable”), QA metrics, sample spectrogram, Vault diff.  
- **Versioning**: SemVer-ish for CLI (0.x pre-stable), CalVer tags allowed for releases (YYYY.MM.patch).  
- **Reviews**: at least one Auditor-signoff for DSP/QA changes; Custodian-signoff for policy/security changes.

--------------------------------------------------------------------------------
## 12) Testing Strategy

- **Unit**: PRNG separation, Euclidean/CA correctness, file naming, sidecar schema.  
- **Property-based**: nonzero note density, crest factor ranges by theme, no identical seeds across 1k runs.  
- **Golden**: MIDI-only fixtures to assert musical grammar (forms, signature distributions).  
- **Soak**: 8-hour batch renders stream-to-disk, measure dropouts, memory growth.  
- **Mythic fuzz**: random pairs, ensure forbidden combos are rejected, verify phase and TP after transform.

- **Example test (Python, tabs used)**
	#!/usr/bin/env python3
	import unittest, json, os
	from pathlib import Path
	class TestSidecar(unittest.TestCase):
		def test_sidecar_fields(self):
			j = json.load(open("./out/sample.riddle.json"))
			for k in ["seed_commitment","theme","form_nodes","artifact_hashes"]:
				self.assertIn(k, j)
	if __name__ == "__main__":
		unittest.main()

--------------------------------------------------------------------------------
## 13) Observability & Telemetry (Local)

- **Metrics (local CSV/SQLite)**:  
  - render_time_sec, cpu_load_est, crest_factor, tp_db, rms_db, notes_total, mythic_count, stems_count
- **Dash (optional)**: local HTML page summarizing last N runs (no network).  
- **Event tags**: theme, form_path, bpm_base, key_mode, duration_bucket.

--------------------------------------------------------------------------------
## 14) Extension Points

- **Themes**: add new theme module (scale set, interval profile, rhythm defaults, timbre recipes, video palette).  
- **Engines**: swap FM/WT/Granular with higher-fidelity blocks; preserve the same controller surfaces.  
- **Video**: add oscilloscope/spectrogram compositor and sigil layers; couple to MIDI controllers.  
- **RitualScript**: author scenes declaratively; compiler enforces bounds & instrument mapping.

--------------------------------------------------------------------------------
## 15) Playbooks

### 15.1 Add a New Mythic Variant
1. Define transform (pure function WAV→WAV, no side effects).  
2. Add forbidden pair checks.  
3. Implement post-transform QA (TP/headroom).  
4. Update probabilities table (keep sums unconstrained; independent rolls).  
5. Extend sidecar lineage and Vault inserts.  
6. Add tests: phase integrity, re-limit correctness, hash changes.

### 15.2 Batch-Generate for a Sampler Pack
	#!/usr/bin/env bash
	set -euo pipefail
	OUT=./out/pack_$(date +"%Y%m%d_%H%M")
	DB=./riddle_vault.db
	mkdir -p "$OUT"
	for i in $(seq 1 20); do
		riddle generate auto "$OUT" --db "$DB" --bucket short --mythic-max 2 -v
	done
	riddle search --db "$DB" --theme glass --min-bpm 80 --max-bpm 120 > "$OUT/manifest.txt"
	echo "[i] Pack complete: $OUT"

### 15.3 Quarantine & Reprocess
	#!/usr/bin/env bash
	set -e
	DB=./riddle_vault.db
	find ./out -name "*MYTHIC_MirrorSalt*" -size -50k -print -exec mv {} ./quarantine/ \;
	# Re-run mythicizer on quarantined sources (not shown): ensure new hashes recorded.

--------------------------------------------------------------------------------
## 16) Environment & Config

- **Env Vars**
	- `RIDDLE_DB` — path to the Vault (default: `./riddle_vault.db`)
	- `RIDDLE_OUTDIR` — default artifact directory
	- `RIDDLE_LUFS_TARGET` — loudness target metadata (default: `-14`)
	- `RIDDLE_CPU_BUDGET` — `low|med|high` (affects synth complexity)
	- `RIDDLE_VIDEO` — `on|off`

- **CLI Defaults**  
  `--mythic-max 2`, limiter ceiling `-0.3 dBTP`, sample rate `48k`, sample width `24-bit`.

--------------------------------------------------------------------------------
## 17) Glossary

- **Form Graph**: sequence of sections (INV/PRC/BRK/TRN/CLM/SHR).  
- **Sigil tones**: seed-chosen pitch classes recurring as leitmotifs.  
- **Mythic**: ritual transform sibling of the primary WAV, lineage-linked.  
- **Sidecar**: JSON attestation with session metadata & artifact hashes.  
- **Vault**: SQLite catalog of runs, sections, and artifacts.

--------------------------------------------------------------------------------
## 18) Roadmap Cues (v0.3+)

- Add video renderer (oscilloscope + spectrogram + sigils).  
- Expand QA to integrated LUFS and better TP (polyphase oversampling).  
- Stems-on-demand with consistent bus processing.  
- RitualScript compiler & scene library.  
- Additional themes: **Glass Dust**, **Iron Mouth**, **Mercury Letter**.

--------------------------------------------------------------------------------
## 19) Contact & Ownership

- **Core Maintainers**: The Infinite Riddle Core  
- **Security/Ethics**: Custodian agent (rotating)  
- **Audit Queue**: PR label `needs-audit` + attach sidecar, WAV hash, and QA metrics

--------------------------------------------------------------------------------
## 20) Appendices

### A) Minimal Reproducer (Dev Mode)
	#!/usr/bin/env bash
	# Generates a reproducible run using a fixed seed (dev only)
	SEED=deadbeefdeadbeefdeadbeefdeadbeef
	riddle generate glass ./out --db ./riddle_vault.db --bucket short --mythic-max 1 -v --seed "$SEED"

### B) Safe Defaults to Ship
- `max_render_hours=2`, `stems=on`, `video=off`, `mythic_max=2`, `microtonal=on (Glass only v1)`

### C) Style Conventions
- Python shebang at top of scripts:
	#!/usr/bin/env python3
- Bash scripts:
	#!/usr/bin/env bash
	set -euo pipefail
- Logging prefixes per §5; **never** use print() for operational logs.

--------------------------------------------------------------------------------
## 21) Final Notes

This is a living document. Propose edits via PRs with concrete diffs to workflows, schemas, or QA gates. Preserve backward compatibility for sidecar and Vault where possible; if breaking, provide a deterministic migrator.

Stay weird, stay coherent. The Riddle should feel alive—yet be auditable.


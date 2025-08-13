# CONTRIBUTING.md — The Infinite Riddle
Version: 0.2
Status: Collaboration-Ready
Owner: The Infinite Riddle Core

--------------------------------------------------------------------------------
## 0) Welcome

Thanks for contributing to **The Infinite Riddle**. This project generates unique, thematically coherent audio/MIDI artifacts with provenance, mythic variants, and an auditable vault. Contributions are expected to preserve the system ethos: *structured chaos, privacy-first, reproducible when explicitly requested, and musically useful outputs.*

This document explains:
- how to set up your dev environment,
- coding & logging standards,
- branching, commits, and PR review,
- test & QA expectations,
- release flow,
- and a small PR template you can paste into your description.

--------------------------------------------------------------------------------
## 1) Quick Start

### 1.1 Requirements
- Python ≥ 3.11 (standard library only for core runtime)
- macOS/Linux/WSL recommended (Windows works; ensure long path support)
- `ffmpeg` only if working on video renderer (optional for v0.2)
- Disk space for long renders; SSD strongly recommended

### 1.2 Clone & Run (single-file script path)
	#!/usr/bin/env bash
	git clone https://example.com/infinite-riddle.git
	cd infinite-riddle
	python3 riddle_v0_2.py generate auto ./out --db ./riddle_vault.db --bucket short -v

### 1.3 Package Layout (when using the package structure)
	/infinite-riddle/
	  AGENTS.md
	  CONTRIBUTING.md
	  SECURITY.md
	  SPEC.md
	  riddle/
	    __main__.py
	    core/      # seeds, PRNGs, forms, grammar
	    synth/     # DSP blocks, limiter, mythics
	    io/        # wav/midi/json, video (later)
	    vault/     # sqlite adapters, queries
	    qa/        # meters, validators
	  tests/
	  out/
	  tools/

Run as a module:
	#!/usr/bin/env bash
	python3 -m riddle generate auto ./out --db ./riddle_vault.db -v

--------------------------------------------------------------------------------
## 2) Coding Standards

### 2.1 General
- Prefer **argparse** over hard-coded parameters. Required args should be positional; optional as flags.
- Include shebangs in runnable scripts:
	#!/usr/bin/env python3
- For bash helpers:
	#!/usr/bin/env bash
	set -euo pipefail

### 2.2 Logging (no print)
- Use `logging` only, with prefixes in message strings:
	# Examples
	logging.info("[i] Theme=%s Duration=%ds", theme, total_sec)
	logging.warning("[!] Falling back to default limiter")
	logging.debug("[DEBUG] PRNG split domains=%s", list(prngs.keys()))
	logging.error("[x] Mythic %s failed: %s", label, exc)

### 2.3 Data & Storage
- Prefer **SQLite** over CSV/TXT for persistent outputs/indices.
- Sidecar JSON is the attestation for every run; never store secrets (only commitments/hashes).
- Do not write outside the chosen `outdir` and DB path provided by the CLI.

### 2.4 Determinism & Seeds
- Default runs: *unrepeatable*. Use local entropy bundle.
- Dev/repro runs: add `--seed <hex>`; derive domain PRNGs via HKDF; never log raw seeds.

### 2.5 Style & Structure
- Keep functions small and pure where possible; pure transforms are easiest to test.
- Boundary checks & basic error handling on all I/O (file existence, permissions, partial writes).
- Robustness first: chunked rendering for long durations; graceful recoveries on transient failures.

--------------------------------------------------------------------------------
## 3) Branching, Commits, and PRs

### 3.1 Branches
- `main` — stable
- `dev/*` — features (e.g., `dev/stems-render`)
- `exp/*` — experiments (squash on merge)

### 3.2 Commit Messages (Conventional-ish)
- Format: `<type>(scope): summary`
- Types: `feat`, `fix`, `perf`, `refactor`, `docs`, `test`, `build`, `ci`, `sec`
- Examples:
	feat(audio): add streamed stem rendering with limiter bus
	fix(qa): correct RMS floor computation on silent edges
	sec(core): mask seed in crash path and redact from logs

### 3.3 PR Checklist (paste into your PR)
	- [ ] Code uses argparse; no hard-coded paths/params
	- [ ] Logging uses prefixes: [i]/[!]/[DEBUG]/[x]; no print()
	- [ ] Sidecar fields populated; no secrets; hashes verified
	- [ ] Vault rows persisted; search works for new fields
	- [ ] QA metrics pass: no clipping, TP ≤ -0.3 dBTP, RMS floor ≥ -45 dBFS
        - [ ] Tests added/updated; `pytest` passes locally
	- [ ] Docs (AGENTS.md/SPEC.md/SECURITY.md) updated if behavior changed
	- [ ] For security-impacting changes: Custodian sign-off included
	- [ ] Attach example artifacts (WAV+MIDI) + sidecar + seed commitment

--------------------------------------------------------------------------------
## 4) Testing & QA

### 4.1 Unit Tests
- Grammar generators (forms/modes), Euclidean rhythm, CA gates
- File naming & sidecar schema
- Mythic transforms are pure functions WAV→WAV (use small fixtures)

Run:
	#!/usr/bin/env bash
	pip install -r requirements-dev.txt
	pytest

### 4.2 Property Tests (recommended)
- No two seeds collide across 10k runs
- Note density > 0.2 notes/sec per section
- Crest factor target by theme
- Mythic forbidden pairs are rejected

### 4.3 Soak/Burn-In
- Long renders (≥ 1 hour) streamed to disk; no dropouts; constant memory profile

### 4.4 QA Gates (Auditor)
- RMS floor ≥ −45 dBFS
- True-peak ≤ −0.3 dBTP (approximate OK in v0.2)
- Phase correlation within bounds for M/S operations
- MIDI usefulness & controller curves present

--------------------------------------------------------------------------------
## 5) Security & Privacy (Dev Duties)

- Respect **consent gates** for mic/cam/network taps; defaults OFF.
- No secret seeds in logs; store **commitments** only (e.g., blake2b).
- Guard against path traversal: join + normalize; restrict to `outdir`.
- Never run untrusted code. No dynamic `eval/exec`.
- See `SECURITY.md` for threat model and disclosure process.

--------------------------------------------------------------------------------
## 6) Release Process

### 6.1 Tagging
- CalVer tags preferred `YYYY.MM.patch` (e.g., `2025.08.0`)
- Update `SPEC.md`/`AGENTS.md` version headers on release

### 6.2 Release QA Bundle
- Include:
	- 1 short, 1 medium session (WAV+MIDI) for each theme
	- sidecar JSONs + vault diff
	- QA summary (RMS/TP/crest factor)
	- changelog and migration notes (if schema changed)

### 6.3 Optional Signing (artifacts bundle)
- Use `minisign` or `cosign` to sign a tarball; include signature & public key in release

--------------------------------------------------------------------------------
## 7) Developer Environment Tips

- Prefer venvs; keep runtime stdlib-only, but dev tools (pytest, ruff) are fine
- Use `time` and `ps`/`top` to watch long renders
- SSDs matter; long audio writes are bandwidth-heavy
- Keep `out/` tidy; use subfolders per batch job

--------------------------------------------------------------------------------
## 8) Contact

- General discussions: Issues with label `discussion`
- Security concerns: follow `SECURITY.md` (private reporting)
- Maintainers: The Infinite Riddle Core

--------------------------------------------------------------------------------
## 9) PR Template (copy/paste)

	### Summary
	<what changed and why>

	### Artifacts
	- WAV: <path> (sha256: …)
	- MIDI: <path> (sha256: …)
	- Sidecar: <path> (sha256: …)
	- Seed commitment: <hex> (no raw seeds)

	### QA
	- TP: <value dBTP> (≤ -0.3 dBTP)
	- RMS floor: <value dBFS> (≥ -45 dBFS)
	- Crest factor: <dB>
	- Notes density: <n/sec>

	### Risk / Security
	- Impacted modules:
	- Secrets handled correctly: yes/no
	- Consent gates unaffected: yes/no

	### Tests
	- Unit: passing
	- Property: passing
	- Soak (if relevant): <duration>

	### Docs
	- Updated AGENTS.md / SPEC.md / SECURITY.md: yes/no

	### Reviewers
	@auditor @custodian @core


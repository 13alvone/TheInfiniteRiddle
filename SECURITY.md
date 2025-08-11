# SECURITY.md — The Infinite Riddle
Version: 0.2
Status: Policy
Owner: Custodian (Security/Ethics)

--------------------------------------------------------------------------------
## 0) Overview

**The Infinite Riddle** is an offline-first, audio/MIDI generator that emphasizes uniqueness, provenance, and privacy. This document defines:
- supported versions and scope,
- threat model & surfaces,
- security & privacy controls,
- reporting & disclosure process,
- hardening guidance for users and contributors.

--------------------------------------------------------------------------------
## 1) Supported Versions

We accept security reports for:
- `main` branch
- the latest tagged release
Older releases may be advised to upgrade rather than patched.

--------------------------------------------------------------------------------
## 2) Threat Model

### 2.1 Goals
- Prevent leakage of sensitive material (seeds, PII)
- Prevent arbitrary code execution and path traversal
- Ensure artifacts’ provenance without revealing secrets
- Maintain integrity of the Vault and sidecar JSON

### 2.2 Non-Goals
- DRM for generated artifacts (outputs belong to the operator)
- Network-layer hardening for remote attack vectors (project is offline-first)

### 2.3 Attack Surfaces
- **CLI inputs**: arguments, file paths, environment variables
- **Filesystem**: writing WAV/MIDI/JSON, SQLite Vault
- **Mythic transforms**: WAV→WAV pipelines must be pure; no external commands
- **Logging**: must not contain secrets (raw seeds, tokens)
- **Optional features**: future video export; optional online taps (gated)

--------------------------------------------------------------------------------
## 3) Security Controls

### 3.1 Seeds & PRNG
- Default runs are seeded from a local entropy bundle; the **raw seed is never logged or stored**.
- Sidecar & logs only contain **seed commitments** (e.g., blake2b hex).
- Reproducible runs accept `--seed <hex>`; still **store commitment only**.

### 3.2 Filesystem Safety
- All writes must be under the user-specified `outdir` and DB path.
- Sanitize paths: resolve realpath, reject `..` escapes, create directories with restrictive permissions.
- Use atomic writes where feasible (temp + rename).

### 3.3 Vault (SQLite)
- Use parameterized queries only.
- Unique constraints on `seed_commitment`.
- Never store secrets or raw seeds; store hash commitments, digests, and metrics only.

### 3.4 Logging Policy
- **No print()**; use `logging` with prefixes in messages:
	[i] info, [!] warning, [DEBUG] debug, [x] error
- Redact anything that could identify the operator or leak seeds.
- On unhandled exceptions: write minimal crash context; offer to quarantine artifacts.

### 3.5 Mythic Transforms
- Pure, deterministic functions on local WAV files.
- Forbidden combinations enforced (e.g., MirrorSalt + Ashen).
- Post-transform QA re-applies limiter checks; recompute hashes.

### 3.6 Offline-First / Consent
- No network access is required for core functionality.
- Any optional online taps (e.g., weather, headlines) must:
	- default to OFF,
	- require explicit flags/consent,
	- cache minimally and store no PII.

### 3.7 Cryptography Notes
- HKDF-SHA256 for domain key separation; BLAKE2b for commitments.
- xoshiro256** is **not** a cryptographically secure PRNG; used for artistic determinism only.
- Do not rely on it for confidentiality; rely on system RNG for initial entropy.

--------------------------------------------------------------------------------
## 4) Hardening Guidance

### 4.1 Operator Runtime
- Run under a non-privileged user account.
- Use a dedicated output directory (e.g., `./out/<date>`).
- Keep Vault on a trusted filesystem; back it up regularly.

### 4.2 Long Renders
- Ensure stable power; prefer laptop on AC or UPS for desktops.
- Use SSDs; avoid network filesystems for real-time streaming.

### 4.3 Audio Safety
- A final limiter enforces headroom, but always check levels before headphones.
- Disable system sound enhancements that may add gain unexpectedly.

### 4.4 Build & Supply Chain
- Core runtime is stdlib-only; for dev tooling (pytest, linters), pin versions in a local requirements file.
- Never add runtime dependencies that fetch code at runtime.
- No dynamic import of untrusted files; no `eval/exec` on user-provided strings.

--------------------------------------------------------------------------------
## 5) Reporting a Vulnerability

### 5.1 Private Disclosure
- Email: `security@infinite-riddle.dev`
- Alternatively, open a **GitHub Security Advisory** (private) to the maintainers.
- Do not open public issues for exploitable vulnerabilities.

### 5.2 What to Include
- A clear description and impact
- Reproducer (minimal CLI) and environment details
- Logs with sensitive data **redacted**
- Suggested remediation if known

### 5.3 Acknowledgement & Timeline
- We aim to acknowledge within **72 hours**.
- Triage and fix target within **30 days** for high-severity issues.
- Public disclosure by mutual agreement, or **90 days** after acknowledgement if unresolved.

--------------------------------------------------------------------------------
## 6) Severity Guide (Informal)

- **Critical**: Code execution, file write outside outdir, seed disclosure, path traversal resulting in overwrite
- **High**: Denial-of-service with trivial input, integrity break in Vault/sidecar hashes
- **Medium**: Info leak in logs (non-seed), persistent crash on common configs
- **Low**: Minor input validation gaps, doc errors causing unsafe usage

--------------------------------------------------------------------------------
## 7) Quarantine & Incident Response

- On QA failure or suspected vulnerability:
	- Move artifacts to `./quarantine/<timestamp>/`.
	- Add a minimal incident JSON with hashes and reason.
	- Avoid deleting evidence; do not share externally until redacted.

	#!/usr/bin/env bash
	mkdir -p ./quarantine/$(date +"%Y%m%d_%H%M%S")
	mv ./out/* ./quarantine/$(date +"%Y%m%d_%H%M%S")/

--------------------------------------------------------------------------------
## 8) Responsible Disclosure Credits

We credit reporters in release notes unless anonymity is requested. Provide a preferred handle when reporting.

--------------------------------------------------------------------------------
## 9) Contact

- Security: `security@infinite-riddle.dev`
- Maintainers: The Infinite Riddle Core
- Ethics/Custodian: rotating reviewer; tag PRs with `needs-custodian`

--------------------------------------------------------------------------------
## 10) Appendix — Secure CLI Patterns

- Use argparse with positional required args and explicit optional flags:
	#!/usr/bin/env python3
	import argparse
	p = argparse.ArgumentParser()
	p.add_argument("theme", choices=["glass","salt","auto"])
	p.add_argument("outdir")
	p.add_argument("--db", default="riddle_vault.db")
	p.add_argument("--seed")  # dev only; stored as commitment
	args = p.parse_args()

- Normalize and validate output paths:
	#!/usr/bin/env python3
	from pathlib import Path
	outdir = Path(args.outdir).resolve()
	outdir.mkdir(parents=True, exist_ok=True)

- Parameterized queries for SQLite:
	#!/usr/bin/env python3
	cur.execute("INSERT INTO runs (started_utc, theme, seed_commitment, ...) VALUES (?,?,?,?,...)", tuple_vals)

- Logging with prefixes; no secrets:
	#!/usr/bin/env python3
	import logging
	logging.info("[i] Seed commitment: %s", seed_commitment[:16])


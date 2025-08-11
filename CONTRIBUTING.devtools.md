# CONTRIBUTING.devtools.md — The Infinite Riddle
Version: 0.2
Status: Dev Workflow Guide
Owner: The Infinite Riddle Core

--------------------------------------------------------------------------------
## 0) Purpose

This companion to CONTRIBUTING.md focuses on **developer tooling**: local environment, tests, linters, mutation testing, coverage, audio QA utilities, and CI examples. Runtime remains stdlib-only; dev tools are optional but recommended.

--------------------------------------------------------------------------------
## 1) Local Environment

### 1.1 Python & venv
	#!/usr/bin/env bash
	python3 -m venv .venv
	. .venv/bin/activate
	python -m pip install --upgrade pip

### 1.2 Recommended Dev Dependencies
- `pytest` — unit/property tests
- `pytest-cov` — coverage report
- `ruff` — lint/format (PEP8-ish, fast)
- `mypy` — optional static typing
- `mutmut` — mutation testing (long-running)
- `soundfile` (dev-only utilities; not used at runtime)
- `numpy` (optional for faster QA scripts; not used at runtime)

	#!/usr/bin/env bash
	pip install pytest pytest-cov ruff mypy mutmut
	# optional helpers used by dev-only QA scripts
	pip install soundfile numpy

### 1.3 Suggested Folder Layout
	/infinite-riddle/
	  riddle/                  # package (if using modular layout)
	  tests/                   # unit & property tests
	  tools/                   # dev-only utilities (scripts, QA)
	  out/                     # generated artifacts (gitignored)
	  .github/workflows/       # CI pipelines (optional)
	  .ruff.toml               # linter config (optional)
	  pyproject.toml           # tool configs (optional)

--------------------------------------------------------------------------------
## 2) Running the Generator Locally

### 2.1 Single Run (auto theme)
	#!/usr/bin/env bash
	python riddle_v0_2.py generate auto ./out --db ./riddle_vault.db --bucket short -v

### 2.2 Reproducible Dev Run (fixed seed)
	#!/usr/bin/env bash
	python riddle_v0_2.py generate glass ./out --db ./riddle_vault.db --bucket short -v --seed deadbeefdeadbeefdeadbeefdeadbeef

--------------------------------------------------------------------------------
## 3) Tests

### 3.1 Unit Tests
        #!/usr/bin/env bash
        pytest -q --seed deadbeefdeadbeefdeadbeefdeadbeef

Tests include a `--seed` option (see `tests/conftest.py`) for deterministic runs.

### 3.2 Coverage
	#!/usr/bin/env bash
	pytest --cov=riddle --cov-report=term-missing

### 3.3 Property Tests (examples)
- Distinct seed commitments across N runs
- Euclidean rhythm correctness
- Note density >= threshold per section

	#!/usr/bin/env python3
	import json, subprocess, tempfile, os, pathlib
	def test_unique_commitments(tmp_path: pathlib.Path):
		seen = set()
		for _ in range(10):
			outdir = tmp_path / "o"
			outdir.mkdir(exist_ok=True)
			subprocess.run(["python","riddle_v0_2.py","generate","auto",str(outdir),"--db",str(tmp_path/"db.sqlite"),"--bucket","short","-v"], check=True)
			sidecars = list(outdir.glob("*.riddle.json"))
			assert sidecars, "no sidecar"
			j = json.loads(sidecars[-1].read_text())
			assert j["seed_commitment"] not in seen
			seen.add(j["seed_commitment"])

--------------------------------------------------------------------------------
## 4) Linting & Typing

### 4.1 Ruff (lint + format)
	#!/usr/bin/env bash
	ruff check .
	ruff format .

### 4.2 Mypy (optional)
	#!/usr/bin/env bash
	mypy riddle

### 4.3 Sample .ruff.toml
	# Place at repo root
	[tool.ruff]
	line-length = 100
	target-version = "py311"
	[tool.ruff.lint]
	select = ["E","F","I","W"]
	ignore = ["E203","E266","E501"]

--------------------------------------------------------------------------------
## 5) Mutation Testing (optional, slow)

	#!/usr/bin/env bash
	mutmut run
	mutmut results

Focus on core logic: PRNG separation, form grammar, rhythm generators, file naming, and mythic transforms.

--------------------------------------------------------------------------------
## 6) Audio QA Utilities (dev-only)

### 6.1 Quick Meter Script (RMS/peak/approx TP)
	#!/usr/bin/env python3
	import wave, audioop, math, sys
	def db(x): return -120.0 if x <= 1e-12 else 20*math.log10(x)
	path = sys.argv[1]
	with wave.open(path,"rb") as r:
		nc, sw, sr, nf, *_ = r.getparams()
		raw = r.readframes(nf)
		pcm16 = audioop.lin2lin(raw, sw, 2)
		rms = audioop.rms(pcm16, 2)/32767.0
		peak = max(abs(audioop.max(pcm16,2)/32767.0), abs(audioop.minmax(pcm16,2)[0]/32767.0))
		print(f"RMS {db(rms):.1f} dBFS | Peak {db(peak):.1f} dBFS")

	# Usage:
	# python tools/meter.py ./out/2025...wav

### 6.2 Batch QA for a Folder
	#!/usr/bin/env bash
	for f in ./out/*.wav; do
	  python tools/meter.py "$f"
	done

--------------------------------------------------------------------------------
## 7) Git Hooks (optional)

### 7.1 pre-commit sample (.pre-commit-config.yaml)
	# install: pip install pre-commit && pre-commit install
	repos:
	  - repo: https://github.com/astral-sh/ruff-pre-commit
	    rev: v0.5.7
	    hooks:
	      - id: ruff
	      - id: ruff-format
	  - repo: https://github.com/pre-commit/mirrors-mypy
	    rev: v1.10.0
	    hooks:
	      - id: mypy

--------------------------------------------------------------------------------
## 8) CI Examples (GitHub Actions)

### 8.1 Python CI
	# .github/workflows/ci.yml
	name: CI
	on:
	  push: { branches: [ main ] }
	  pull_request:
	permissions: { contents: read }
	jobs:
	  test:
	    runs-on: ubuntu-latest
	    steps:
	      - uses: actions/checkout@v4
	      - uses: actions/setup-python@v5
	        with: { python-version: '3.11' }
	      - run: python -m pip install --upgrade pip
	      - run: pip install pytest pytest-cov ruff
	      - run: ruff check .
	      - run: pytest --cov=riddle --cov-report=term-missing

### 8.2 Long-Render Soak (nightly)
	# .github/workflows/soak.yml
	name: Soak
	on:
	  schedule: [{ cron: "0 5 * * *" }]
	jobs:
	  soak:
	    runs-on: ubuntu-latest
	    steps:
	      - uses: actions/checkout@v4
	      - uses: actions/setup-python@v5
	        with: { python-version: '3.11' }
	      - run: python riddle_v0_2.py generate auto ./out --db ./riddle_vault.db --bucket long -v
	      - run: ls -lh ./out

--------------------------------------------------------------------------------
## 9) Release Helper Scripts (optional)

### 9.1 Build a Sampler Pack
	#!/usr/bin/env bash
	set -euo pipefail
	OUT=./out/pack_$(date +"%Y%m%d_%H%M")
	mkdir -p "$OUT"
	for i in $(seq 1 20); do
		python riddle_v0_2.py generate auto "$OUT" --db ./riddle_vault.db --bucket short -v
	done
	ls -1 "$OUT" > "$OUT/manifest.txt"
	echo "[i] Pack complete → $OUT"

--------------------------------------------------------------------------------
## 10) Troubleshooting

- **Audio too quiet**: check limiter ceiling and your OS enhancement chain.
- **Large files**: use `--bucket short` during dev.
- **Identical outputs suspicion**: compare `seed_commitment` and artifact hashes; collisions should not occur.
- **Slow QA**: skip mutation tests locally; rely on CI/nightly.

--------------------------------------------------------------------------------
## 11) Contact

- Dev tool questions: open an issue with label `devtools`
- Security: see SECURITY.md


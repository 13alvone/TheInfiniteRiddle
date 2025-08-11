# The Infinite Riddle — Audio/AV Relic Forge & Music Grammar Spec
Version: 0.1 (Design Freeze Candidate)
Status: Draft
Owner: The Infinite Riddle Core
Scope: Audio/MIDI/Video generation & provenance; form/harmony/rhythm grammar; mythic transforms; vault schema.

---

## 0) Session Math (seed → PRNGs)

- Root seed: 256-bit (BLAKE3 of entropy bundle).
- Entropy bundle (examples; all opt-in & privacy-preserving): high-res monotonic clock jitter, CPU/GPU clock drift, keystroke timing jitter, filesystem microstats deltas, optional mic-room noise energy (never stored).
- Stream split: derive subkeys with HKDF:
  - `k_form`, `k_harmony`, `k_rhythm`, `k_melody`, `k_synth`, `k_ctrl`, `k_video`, `k_mythic`.
- PRNG: xoshiro256** instance per subkey (domain separation). Each generator advances only in its domain.
- Micro-gestures: low-rate chaotic maps (logistic r∈[3.7,3.98]) sampled per section; reseeded from domain PRNG.

---

## 1) Length Engine (heavy-tail durations)

- Bucket draw: P(short)=0.70, P(med)=0.25, P(long)=0.05.
- Within bucket, sample T = round(exp(N(μ,σ))):
  - short: μ=ln(60), σ=0.45 → ~30–180 s
  - med:   μ=ln(600),σ=0.50 → ~5–20 min
  - long:  μ=ln(5400),σ=0.60 → ~45–240 min
- Theme clamps:
  - Glass/Refraction: +20% chance to upshift bucket; permits very long drones.
  - Salt/Sealing: −20% chance to downshift bucket; prefers shorter rites.
- Render: stream to disk (chunk = 8192 frames) with 2-chunk lookahead for FX tails.

---

## 2) Form Graph Grammar

### Common nodes
- INV (Invocation): establish motif/texture
- PRC (Procession): develop rhythm/harmony
- BRK (Break): texture collapse, sparse events
- TRN (Transfiguration): mode/pivot modulation
- CLM (Calm/Stillness): low-density drone
- SHR (Shatter/Coda): decisive texture event, exit tail

### Glass / Refraction (form rules)
- Allowed paths (≈1.0 total):
  - INV→PRC→TRN→PRC→SHR (0.32)
  - INV→PRC→BRK→TRN→PRC→SHR (0.28)
  - INV→CLM→PRC→TRN→SHR (0.18)
  - INV→PRC→CLM→SHR (0.12)
  - INV→TRN→PRC→TRN→SHR (0.10)
- Section durations (% of T):
  - INV 12–20, PRC 20–35, BRK 8–15, TRN 10–18, CLM 10–25, SHR 8–15 (soft constraints to sum 100).

### Salt / Sealing (form rules)
- Allowed paths:
  - INV→PRC→CLM→SHR (0.34)
  - INV→CLM→PRC→CLM→SHR (0.26)
  - INV→PRC→BRK→CLM→SHR (0.22)
  - INV→CLM→TRN→CLM→SHR (0.18)
- Section durations (% of T):
  - INV 10–18, PRC 18–30, BRK 6–12, TRN 8–15, CLM 22–40, SHR 8–15.

---

## 3) Harmony & Keyspace

### Key selection
- Base root ∈ {C..B}, biases:
  - Glass: favor D, F#, A.
  - Salt: favor C, F, G#.
- Long pieces: 0–2 pivots:
  - Glass: 60% chance of 1 pivot, 10% of 2.
  - Salt: 35% chance of 1 pivot, 5% of 2.

### Mode sets
- Glass: Lydian (0.38), Whole-tone (0.22), Messiaen Mode 3 (0.18), Ionian+#11 (0.12), 19-EDO ornament (0.10 over diatonic scaffold).
- Salt: Phrygian (0.33), Locrian (0.22), Hijaz/Phrygian dominant (0.20), Pelog-like 5-note (0.15), Dorian♭2 (0.10).

### Voice-leading constraints
- Bass moves ≤ P4 most steps; leaps require contrary motion in pad/lead.
- Chordal skeleton bias:
  - Glass: I – II – V (Lydian color emphasis).
  - Salt: i – ♭II – v (dark gravity).
- Mode slips at TRN:
  - Glass: pivot key vs pivot mode = 0.55 / 0.45.
  - Salt: 0.40 / 0.60.

---

## 4) Rhythm Engine

### Time signatures (per node)
- Glass: 4/4 (0.35), 5/4 (0.22), 7/8 (0.18), 9/8 (0.12), 11/8 (0.08), 13/8 (0.05).
- Salt: 7/8 (0.24), 5/4 (0.22), 3/4 (0.18), additive (2+2+3)/8 (0.18), 4/4 (0.18).

### Generators
- Euclidean pulses E(k,n) per track; swing s∈[−0.12,+0.12]:
  - Glass defaults: E(5,8), E(7,12), E(9,16).
  - Salt defaults: E(3,8), E(5,12), E(7,12) + additive groupings.
- Cellular Automata gating:
  - Glass: Rule 90, p_refresh=0.15/bar.
  - Salt: Rule 150, p_refresh=0.25/bar.
- Downbeat stability (coherence c):
  - Glass: P(keep beat-1)=0.5+0.5c.
  - Salt: 0.3+0.6c.

---

## 5) Melody Engine (Leitmotifs & Intervals)

### Sigil tones (leitmotif)
- Choose 2–3 pitch classes from current scale; guarantee ≥1 appearance per node.
- Recurrence target: ≥65% of phrases touch a sigil tone.

### Interval class preference
- Glass (probabilities): m2 0.10, M2 0.22, m3 0.18, M3 0.12, P4 0.10, TT 0.12, P5 0.10, 8ve 0.06.
- Salt (probabilities): m2 0.24, M2 0.14, m3 0.20, M3 0.09, P4 0.09, TT 0.10, P5 0.08, 8ve 0.06.

### Phrase shapes
- Length: Glass 4–12 beats; Salt 5–9 beats.
- Contours: arc, ramp, stair, spike, mirror.
  - Glass favors mirror; Salt favors ramp/arc.
- Cadences: 55% to sigil tone, 20% deceptive, 25% open.

---

## 6) Orchestration & Synthesis Mapping

### Stem roles
- bass/sub (mono-lean), pad/textures (stereo), lead/melody, perc/impulse, fx/sigil-noise.

### Glass timbres
- Lead: 2–3 op FM bell (ratios ~1:2, 2:3, 3:5), inharmonicity 0.01–0.04.
- Pad: wavetable sweep across “refraction” tables, width 0.3–0.7, LFO 0.01–0.05 Hz.
- FX: comb-delays 2.5–12 ms, shimmer (+12 st, send 0.15–0.35).

### Salt timbres
- Bass: sub + subharmonic (÷2) mix 0.15–0.35, gentle drive.
- Pad: granular from self-generated impulses/noise; density 3–12 grains/s; window 50–120 ms.
- Perc: filtered noise taps; convolution with metal/shaker IRs; decay 40–140 ms.

### Dynamics & mix
- Pre-limiter headroom: ≤ −3.0 dBFS.
- Crest factor target: Glass 12–16 dB; Salt 8–12 dB.
- True-peak limiter: ceiling −0.3 dBTP; lookahead 5 ms; soft knee.

---

## 7) MIDI Emission

- Tempo map:
  - Glass base 68–128 BPM (mode ~96), macro drift ±3–8%; rubato on cadences ±1.5–3%.
  - Salt base 40–92 BPM (mode ~64), macro drift ±2–6%.
- Time/key signatures written per node; mode in meta text.
- Channels:
  - ch1 lead, ch2 pad (chords/arps), ch3 bass, ch10 perc, ch11 fx gates.
- Controllers:
  - CC1 timbre (FM index / grain density)
  - CC11 expression (global dynamics)
  - CC74 brightness (filter cutoff / wavetable position)
  - CC64 sustain as “ritual gate” (pads; long holds)
- Program hints in meta text (e.g., `glass_bell_1`, `salt_drone_2`).

---

## 8) Controller Curves (Gesture Fields)

- Curves per section sampled at:
  - Audio side: 10–30 Hz
  - MIDI side: per quarter note

### Glass mappings
- CC74 ← slow sine + pink noise (peak-to-peak 18–28)
- CC1  ← logistic (r≈3.86), EMA(α=0.1), range 32–96
- CC11 ← normalized event density envelope (48–104)

### Salt mappings
- CC74 ← 1/f drift (range 10–22) with rare BRK spikes
- CC1  ← very slow sine (0.01–0.03 Hz) + dither (24–72)
- CC11 ← long triangle swells (15–90 s), range 36–100

- Automation at TRN: 2–5 s morphs of 2–3 params (Glass: comb feedback↑; Salt: subharmonic depth↑).

---

## 9) Output Contract (every run)

- Primary WAV: 48 kHz, 24-bit, mono/stereo by theme; limiter ceiling −0.3 dBTP; −14 LUFS target (configurable).
- Primary MIDI: multi-track with tempo map, key/time sigs, controllers per §7.
- Optional Video: MP4/AV1 (48 kHz audio); 1080p30 (short/med), 720p30 (long).
- Stems (optional): drums/perc, bass/sub, lead/melody, pad/textures, fx/sigil-noise → individual WAVs; MIDI tracks where applicable.
- Sidecar JSON: full session attestation & musical metadata (see §11).
- File naming:
  - `YYYYMMDD_HHMMSS_RIDDLE_<Theme>_<FormPath>_BPM<xxx>_<KeyMode>_LEN<mm-ss>_<Seed8>.wav`
  - `YYYYMMDD_HHMMSS_RIDDLE_<Theme>_MIDI_<Seed8>.mid`
  - `YYYYMMDD_HHMMSS_RIDDLE_<Theme>_VIDEO_<Seed8>.mp4`
  - `YYYYMMDD_HHMMSS_RIDDLE_<Theme>_STEMS_<part>_<Seed8>.wav`
  - `YYYYMMDD_HHMMSS_RIDDLE_<Theme>_MYTHIC_<Type>_<Seed8>.wav`
  - `YYYYMMDD_HHMMSS_RIDDLE_<Theme>_<Seed8>.riddle.json`

---

## 10) Mythic Variants (ritual transforms)

- Roll 0–3 independent variants (max 3) using `k_mythic`. Probabilities:
  - Backmask (0.28)
  - Seal Encoding (FLAC + metadata sigil + stereo micro-decorrelation) (0.22)
  - Chalice (varispeed −7% + −12 st sub layer) (0.18)
  - Mirror Salt (M/S swap + ±0.2° phase wobble) (0.14)
  - Shatter (granular reassembly at φ points) (0.12)
  - Ashen (multi-band 8–12-bit with soft tape hiss) (0.11)
  - Liminal (−20 dB drone extension, cap 60 m) (0.07)
  - CipherSpray (ultrasonic seed watermark) (0.20)
- Safety matrix:
  - Don’t combine Backmask with Liminal.
  - Don’t combine Mirror Salt with Ashen.
- Post-transform checks:
  - Re-measure headroom/TP; re-limit if needed.
  - Maintain provenance links in sidecar.

---

## 11) Provenance & Sidecar JSON (attestation)

- Fields (core):
  - `seed_commitment` (BLAKE3 of root seed)
  - `prng_epoch` (domain seeds digests)
  - `theme`, `form_nodes`, `form_path`, `durations`
  - `bpm_curve`, `key_mode_timeline`
  - `leitmotif_pcs`, `interval_profile`
  - `rhythm_specs` (time sig, Euclid, CA rule, swing)
  - `synth_rack` (chosen recipes & params)
  - `ctrl_curves_digest` (hash of automation tracks)
  - `artifact_hashes` (sha256 per file)
  - `mythic_lineage`
  - `loudness`, `crest_factor`, `true_peak`
- License statement (default): generated works belong to the local operator.

---

## 12) Vault (SQLite) Schema

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
	  euclid TEXT,     -- e.g., "E(5,8)|E(7,12)"
	  ca_rule INTEGER, -- 90 or 150
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

	-- indexes
	CREATE INDEX idx_artifacts_kind ON artifacts(kind);
	CREATE INDEX idx_runs_theme ON runs(theme);

- Example queries the app will support:
	- All Glass artifacts > 4 min between 80–100 BPM.
	- Find WAVs in D Lydian with mythic=Shatter.
	- Export last 10 runs’ MIDI + stems.

---

## 13) Video Mapping (optional renderer)

- Visual tracks:
  - Oscilloscope line following summed stem.
  - Spectrogram ribbon (FFT 1024/2048, hop 256).
  - Sigil overlays generated from seed; fade near sigil-tone events.
  - Whisper text: 1–3 frame acrostic flickers on cadences (opacity ≤ 0.12).
- Motion coupling:
  - Line thickness ← CC11 EMA.
  - Hue shift (subtle) ← CC74 normalized.
  - Stutter/glitch frames on SHR downbeats (Glass only).
- Export:
  - 1080p30 (short/med), 720p30 (long); AAC 192 kbps mux.
  - Filename lineage mirrors audio names.

---

## 14) QA Gates (robustness)

- No-silence rule: min RMS floor −45 dBFS; inject low-bed if below.
- Clipping guard: pre-limit max ≤ −1.0 dBFS; post-limit true-peak ≤ −0.3 dBTP.
- Phase guard: M/S correlation |ρ| ≤ 0.98 sustained; reduce width if violated.
- MIDI usefulness: ≥1 track per node with note density > 0.2 notes/s.
- Export attestation: hash-record all files; mismatch → re-render section only.
- Long-render watchdog: chunked streaming; recover from transient synth underruns.

---

## 15) Theme→Sound Mapping (Launch Themes)

### Glass / Refraction
- Scales/Modes: Lydian, Whole-tone, Messiaen M3; rare 19-EDO ornaments.
- Rhythm: Euclidean E(5,8), E(7,12); polymeters 5:4, 7:6; crystalline stutters.
- Timbre: FM/wavetable bells, inharmonic partials; comb filters; shimmer verbs.
- Dynamics: wide crest factor; long decays.
- Video: prismatic glyphs, diffraction lines, spectrogram blooms.

### Salt / Sealing
- Scales/Modes: Phrygian, Locrian, Hijaz; Pelog-like pentachords.
- Rhythm: slow ostinati; additive meters (3+2+2 / 2+2+3); tape-age swing.
- Timbre: bowed metal, granular sand hiss, subharmonic oscillators, convolved IRs.
- Dynamics: creeping swells; distant thumps.
- Video: seal-circles, salt lines; faint runes.

---

## 16) RitualScript (authoring DSL; illustrative)

- Primitives: `sigil(text…)`, `bind(symbol…)`, `whisper(text…)`, `gate(name){ … }`, `mask(rule)`, `portent(filter)`.
- Constraints: `ensures(theme_tag>=0.7)`, `drift(rate=0.03)`.
- Agents: `invoke(Archivist|Saboteur|Mirror, intensity=…)`.
- Outputs: `emit(image|audio|text, transmuters=[…])`.

	// Glass Invocation
	gate INV {
	  bind theme="Glass/Refraction"
	  ensure motif.sigils>=2
	  emit pad using wavetable(refraction), width~U(0.4,0.7)
	  emit lead using fm(bell), iclass_profile="glass_default"
	  rhythm euclid E(5,8) with ca=90 swing~N(0,0.04)
	  ctrl CC74 <- sine(0.02Hz) + pink(±10)
	  ensure form.next in {PRC,CLM}
	}

	// Salt Calm
	gate CLM {
	  bind theme="Salt/Sealing"
	  emit pad using granular(sand_hiss), density~U(3,8)
	  emit bass using sub + subharmonic mix~U(0.18,0.32)
	  rhythm additive "2+2+3/8" with ca=150
	  ctrl CC11 <- triangle(period=45s, range=36..96)
	  ensure mode_slip probability=0.6 at TRN
	}

---

## 17) Signature Experiences (targets)

- The Closed Book: single pane; letters rearrange while reading; screenshots render an alternate page (capture hook).
- The False Echo: your typed sentence is answered by whispers using only letters you have not pressed this session.
- Equinox Debt: on equinox, Vault offers a daemon in exchange for forgetting prior run fragments (logged; reversible via counter-puzzle).

---

## 18) Defaults (shipping)

- max_render_hours = 2
- stems = on
- video = off (toggle on for med/long)
- mythic_max = 2
- loudness_target = −14 LUFS, limiter ceiling −0.3 dBTP
- microtonal = on (Glass only v1)
- cpu_budget = med (FM + WT + light granular)
- export_formats = ["wav","midi","json"] (add "mp4","flac" via toggle)

---

## 19) Build Order (implementation)

1) PRNG split + length engine + form grammar  
2) MIDI-only path (harmony/rhythm/melody) → musical verification  
3) Synth rack (FM/WT/Granular) + controller curves → audio render  
4) Theme→Sound mapping (Glass/Salt)  
5) Mythic transforms (offline) + attestation + Vault writes  
6) Optional video renderer (spectrogram + sigils)  
7) CLI/TUI UX + search/export tooling

---

## 20) Two Curated First Experiences (reference)

- Refraction Hymn (Glass)
  - Form: INV→PRC→TRN→PRC→SHR; 7:40; 96→104 BPM
  - D Lydian; pivot to Whole-tone at TRN (≈90 s)
  - Euclid E(5,8)/E(7,12), Rule 90, swing +0.06
  - Mythics: Shatter at φ·len; Seal Encoding sibling
  - Outcome: bell constellation; crisp MIDI; airy, mix-ready.

- Sealing Breath (Salt)
  - Form: INV→CLM→PRC→CLM→SHR; 4:28; 62 BPM
  - A Phrygian; brief Dorian♭2 micro-slip at TRN
  - Additive 2+2+3/8, Rule 150, swing −0.04
  - Mythic: Chalice (−7% varispeed + −12 st layer)
  - Outcome: weighty drone + ritual taps; stems ideal under vocals.

---

## 21) Ethical & Operational Notes

- Consent gates for any mic/cam/network taps; default OFF; air-gap mode fully supported.
- No collection of personal data; “presence” is inferred only from local interaction rhythms.
- Panic chord (e.g., Ctrl+Alt+~) freezes daemons and reveals status pane.
- License: generated outputs are yours.

---


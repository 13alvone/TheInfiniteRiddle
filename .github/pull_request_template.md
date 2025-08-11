# .github/pull_request_template.md — The Infinite Riddle
Version: 0.2
Status: Template

### Summary
<what changed and why; link issues>

### Artifacts (attach or link paths)
- WAV: <path>  
- MIDI: <path>  
- Sidecar JSON: <path>  
- Seed commitment: `<hex>` (do **not** include raw seed)

### QA (paste values)
- True-peak: `<dBTP>` (≤ -0.3 dBTP)
- RMS floor: `<dBFS>` (≥ -45 dBFS)
- Crest factor: `<dB>`
- Note density (min per section): `<n/sec>`
- Mythic variants generated: `<list or none>`
- Phase correlation (if M/S ops): `<value>`

### Reproducibility
- Dev seed used? `yes/no`  
- If yes, provide **commitment only** and store raw seed securely.

### Security/Privacy
- Changes touch seeds/logging/paths/db? `yes/no`
- Consent gates impacted? `yes/no`
- Any network I/O added? `yes/no` (must default OFF and be optional)

### Tests
- Unit tests updated: `yes/no`
- Property tests updated: `yes/no`
- Soak test (if long-render feature): `<duration or n/a>`
- Coverage delta (if applicable): `+/- x%`

### Docs
- Updated: `AGENTS.md / SPEC.md / SECURITY.md / CONTRIBUTING.md / devtools` (circle)

### Checklist
- [ ] Uses argparse; required args positional; no hard-coded paths
- [ ] Logging uses prefixes `[i]/[!]/[DEBUG]/[x]`; no print()
- [ ] Sidecar fields populated; no secrets; hashes verified
- [ ] Vault rows inserted; queries succeed
- [ ] QA gates pass (TP/RMS/phase)
- [ ] Forbidden mythic combos rejected (if relevant)
- [ ] Backward-compatible file naming
- [ ] CI green

### Screens/Audio (optional)
<attach spectrogram screenshots or short audio clips>

### Reviewers
@auditor @custodian @core


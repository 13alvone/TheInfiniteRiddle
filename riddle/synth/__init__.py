#!/usr/bin/env python3
"""Synthesis, rendering, and mythic transforms for The Infinite Riddle."""
import audioop
import math
import struct
import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..core import Xoshiro256StarStar

# ----------------------------- Simple Synth Rack -----------------------------
class StereoLimiter:
    def __init__(self, ceiling=0.97, lookahead_samples=0, release=0.005):
        self.ceiling = float(ceiling)
        self.lookahead = int(lookahead_samples)
        self.release = float(release)
        self.gain = 1.0

    def process_block(self, L: List[float], R: List[float]) -> Tuple[List[float], List[float]]:
        outL, outR = [], []
        for l, r in zip(L, R):
            peak = max(abs(l), abs(r))
            target = 1.0
            if peak * self.gain > self.ceiling:
                target = self.ceiling / max(peak, 1e-12)
            self.gain += (target - self.gain) * self.release
            outL.append(l * self.gain)
            outR.append(r * self.gain)
        return outL, outR


def sine(phase: float) -> float:
    return math.sin(phase)


def tri(phase: float) -> float:
    x = (phase / (2*math.pi)) % 1.0
    return 4.0 * abs(x - 0.5) - 1.0


def softsat(x: float, drive: float = 1.0) -> float:
    return math.tanh(drive * x)


def hz_from_midi(note: int) -> float:
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


class Voice:
    def __init__(self, sr: int, mode: str):
        self.sr = sr
        self.mode = mode
        self.phase_c = 0.0
        self.phase_m = 0.0
        self.env = 0.0
        self.active = False
        self.release = False
        self.note = 60
        self.freq = hz_from_midi(60)
        if mode == "fm_lead":
            self.mod_idx = 2.0
            self.ratio = 2.0
            self.a, self.d, self.s, self.r = 0.01, 0.20, 0.5, 0.30
        elif mode == "wt_pad":
            self.mod_idx = 0.4
            self.ratio = 0.5
            self.a, self.d, self.s, self.r = 0.6, 0.8, 0.7, 1.2
        else:
            self.mod_idx = 1.0
            self.ratio = 1.0
            self.a, self.d, self.s, self.r = 0.01, 0.1, 0.7, 0.2

    def note_on(self, note: int, vel: float):
        self.note = note
        self.freq = hz_from_midi(note)
        self.env = 0.0
        self.active = True
        self.release = False
        self.vel = vel

    def note_off(self):
        self.release = True

    def _adsr(self, dt: float):
        if not self.release:
            if self.env < 1.0:
                self.env = min(1.0, self.env + dt / max(self.a, 1e-5))
            else:
                if self.env > self.s:
                    self.env = max(self.s, self.env - dt / max(self.d, 1e-5))
        else:
            self.env = max(0.0, self.env - dt / max(self.r, 1e-5))
            if self.env <= 1e-4:
                self.active = False

    def render(self, n: int, sr: int, lfo_val: float) -> List[float]:
        out = []
        dt = 1.0 / sr
        for _ in range(n):
            self._adsr(dt)
            if self.mode == "fm_lead":
                self.phase_m += 2*math.pi * (self.freq * self.ratio) * dt
                mod = math.sin(self.phase_m) * self.mod_idx * (0.6 + 0.4 * lfo_val)
                self.phase_c += 2*math.pi * self.freq * dt + mod * dt
                s = math.sin(self.phase_c)
            else:
                self.phase_c += 2*math.pi * self.freq * dt
                wt = 0.5 + 0.5 * math.sin(lfo_val * 2*math.pi)
                s = (1.0 - wt) * math.sin(self.phase_c) + wt * tri(self.phase_c)
            out.append(self.env * self.vel * s)
        return out


class SubBass:
    def __init__(self, sr: int):
        self.sr = sr
        self.active_notes: Dict[int, float] = {}

    def note_on(self, note: int, vel: float):
        self.active_notes[note] = 0.0

    def note_off(self, note: int):
        self.active_notes.pop(note, None)

    def render(self, n: int) -> List[float]:
        out = [0.0] * n
        dt = 1.0 / self.sr
        for note, phase in list(self.active_notes.items()):
            f = hz_from_midi(note) / 2.0
            for i in range(n):
                phase += 2*math.pi * f * dt
                out[i] += 0.4 * math.sin(phase)
            self.active_notes[note] = phase
        return out


class NoisePerc:
    def __init__(self, sr: int, prng: Xoshiro256StarStar):
        self.sr = sr
        self.prng = prng
        self.env = 0.0
        self.decay = 0.08

    def hit(self, strength: float = 1.0):
        self.env = min(1.0, self.env + 0.8 * strength)

    def render(self, n: int) -> List[float]:
        if self.env <= 0.0:
            return [0.0] * n

        out = []
        for _ in range(n):
            rnd = self.prng.randbits(32) & 0xFFFFFFFF
            rnd = (rnd ^ (rnd << 13)) & 0xFFFFFFFF
            rnd = (rnd ^ (rnd >> 17)) & 0xFFFFFFFF
            rnd = (rnd ^ (rnd << 5)) & 0xFFFFFFFF
            noise = ((rnd / 0xFFFFFFFF) * 2.0) - 1.0
            self.env = max(0.0, self.env - (1.0 / (self.sr * self.decay)))
            out.append(self.env * 0.5 * noise)
            if self.env <= 0.0:
                out.extend([0.0] * (n - len(out)))
                break
        return out


# ----------------------------- Renderer -----------------------------
def render_audio(output_path: Path, midi_events: Dict[str, List[Tuple[int,int,int,int]]], bpm: float,
                 ppq: int, sr: int, total_sec: int, prng: Xoshiro256StarStar,
                 limiter_ceiling: float = 0.97,
                 stem_paths: Optional[Dict[str, Path]] = None) -> None:
    s_per_tick = 60.0 / bpm / ppq
    lead = midi_events.get("lead", [])
    pad  = midi_events.get("pad", [])
    bass = midi_events.get("bass", [])
    perc = midi_events.get("perc", [])

    lead_voices: List[Voice] = []
    pad_voices: List[Voice] = []
    sub_bass = SubBass(sr)
    drums = NoisePerc(sr, prng)
    limiter = StereoLimiter(ceiling=limiter_ceiling)
    state = prng.s.copy()

    def expand(events):
        seq = []
        for t, dur, note, vel in events:
            seq.append(("on", t*s_per_tick, note, vel))
            seq.append(("off", (t+dur)*s_per_tick, note, 0))
        return sorted(seq, key=lambda x: x[1])

    events_timed = {
        "lead": expand(lead),
        "pad":  expand(pad),
        "bass": expand(bass),
        "perc": [("hit", t*s_per_tick, note, vel) for (t, d, note, vel) in perc],
    }

    block = 1024
    total_samples = total_sec * sr
    t_sec = 0.0
    evt_idx = {k:0 for k in events_timed.keys()}

    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(3)
        wf.setframerate(sr)
        while int(t_sec * sr) < total_samples:
            n = min(block, total_samples - int(t_sec * sr))
            t_end = t_sec + n / sr
            for track in ("lead","pad","bass","perc"):
                evts = events_timed[track]
                i = evt_idx[track]
                while i < len(evts) and evts[i][1] < t_end:
                    typ, tt, note, vel = evts[i]
                    i += 1
                    if track == "lead":
                        if typ == "on":
                            v = Voice(sr, "fm_lead")
                            v.note_on(note, vel/127.0)
                            lead_voices.append(v)
                        else:
                            for v in lead_voices:
                                if v.note == note and v.active:
                                    v.note_off()
                                    break
                    elif track == "pad":
                        if typ == "on":
                            v = Voice(sr, "wt_pad")
                            v.note_on(note, vel/127.0 * 0.6)
                            pad_voices.append(v)
                        else:
                            for v in pad_voices:
                                if v.note == note and v.active:
                                    v.note_off()
                                    break
                    elif track == "bass":
                        if typ == "on":
                            sub_bass.note_on(note - 12, vel/127.0)
                        else:
                            sub_bass.note_off(note - 12)
                    elif track == "perc" and typ == "hit":
                        drums.hit(vel/127.0)
                evt_idx[track] = i

            L = [0.0] * n
            R = [0.0] * n
            lfo = math.sin(2*math.pi * 0.02 * t_sec)
            alive_leads = []
            for v in lead_voices:
                buf = v.render(n, sr, lfo)
                if v.active:
                    alive_leads.append(v)
                for i, s in enumerate(buf):
                    L[i] += 0.22 * s
                    R[i] += 0.22 * s
            lead_voices = alive_leads

            alive_pads = []
            for v in pad_voices:
                buf = v.render(n, sr, lfo)
                if v.active:
                    alive_pads.append(v)
                for i, s in enumerate(buf):
                    L[i] += 0.18 * s
                    R[i] += 0.18 * s
            pad_voices = alive_pads

            bbuf = sub_bass.render(n)
            for i, s in enumerate(bbuf):
                L[i] += 0.20 * s
                R[i] += 0.20 * s

            pbuf = drums.render(n)
            for i, s in enumerate(pbuf):
                L[i] += 0.20 * s
                R[i] += 0.20 * s

            for i in range(n):
                L[i] = softsat(L[i], 1.2)
                R[i] = softsat(R[i], 1.2)

            L, R = limiter.process_block(L, R)

            frames = bytearray()
            for i in range(n):
                l = max(-0.999999, min(0.999999, L[i]))
                r = max(-0.999999, min(0.999999, R[i]))
                li = int(l * 8388607.0)
                ri = int(r * 8388607.0)
                frames += struct.pack("<i", li)[0:3]
                frames += struct.pack("<i", ri)[0:3]
            wf.writeframes(frames)
            t_sec = t_end

    if stem_paths:
        for track, path in stem_paths.items():
            prng.s = state.copy()
            sub_events = {track: midi_events.get(track, [])}
            render_audio(path, sub_events, bpm, ppq, sr, total_sec, prng, limiter_ceiling, stem_paths=None)


# ----------------------------- Mythic Variants -----------------------------
def mythic_backmask(src_wav: Path, dst_wav: Path) -> None:
    with wave.open(str(src_wav), "rb") as r:
        params = r.getparams()
        frames = r.readframes(r.getnframes())
    sampwidth = params.sampwidth
    if sampwidth != 3:
        raise ValueError("Expected 24-bit WAV")
    pcm16 = audioop.lin2lin(frames, 3, 2)
    rev16 = audioop.reverse(pcm16, 2)
    frames24 = audioop.lin2lin(rev16, 2, 3)
    with wave.open(str(dst_wav), "wb") as w:
        w.setparams(params)
        w.writeframes(frames24)


def mythic_ashen_bitcrush(src_wav: Path, dst_wav: Path, bits: int = 12) -> None:
    with wave.open(str(src_wav), "rb") as r:
        params = r.getparams()
        frames = r.readframes(r.getnframes())
    if params.sampwidth != 3:
        raise ValueError("Expected 24-bit WAV")
    pcm16 = audioop.lin2lin(frames, 3, 2)
    step = 1 << (16 - bits)
    crushed = audioop.bias(audioop.mul(pcm16, 2, 1.0), 2, 0)
    arr = bytearray(crushed)
    for i in range(0, len(arr), 2):
        s = int.from_bytes(arr[i:i+2], "little", signed=True)
        s = int(round(s / step) * step)
        s = max(-32768, min(32767, s))
        arr[i:i+2] = int(s).to_bytes(2, "little", signed=True)
    frames24 = audioop.lin2lin(bytes(arr), 2, 3)
    with wave.open(str(dst_wav), "wb") as w:
        w.setparams(params)
        w.writeframes(frames24)


def mythic_mirrorsalt_ms(src_wav: Path, dst_wav: Path) -> None:
    with wave.open(str(src_wav), "rb") as r:
        params = r.getparams()
        frames = r.readframes(r.getnframes())
    if params.nchannels != 2 or params.sampwidth != 3:
        raise ValueError("Expected 24-bit stereo WAV")
    pcm16 = audioop.lin2lin(frames, 3, 2)
    L = audioop.tomono(pcm16, 2, 1.0, 0.0)
    R = audioop.tomono(pcm16, 2, 0.0, 1.0)
    M = audioop.add(L, R, 2)
    S = audioop.sub(L, R, 2)
    S = audioop.mul(S, 2, -0.98)
    L2 = audioop.mul(audioop.add(M, S, 2), 2, 0.5)
    R2 = audioop.mul(audioop.sub(M, S, 2), 2, 0.5)
    stereo = audioop.tostereo(L2, 2, 1.0, 0.0)
    stereo = audioop.add(stereo, audioop.tostereo(R2, 2, 0.0, 1.0), 2)
    frames24 = audioop.lin2lin(stereo, 2, 3)
    with wave.open(str(dst_wav), "wb") as w:
        w.setparams(params)
        w.writeframes(frames24)


def mythic_liminal_bed(src_wav: Path, dst_wav: Path, gain_db: float = -20.0, max_minutes: int = 60) -> None:
    with wave.open(str(src_wav), "rb") as r:
        params = r.getparams()
        sr = r.getframerate()
        nframes = min(r.getnframes(), max_minutes*60*sr)
        frames = r.readframes(nframes)
    if params.sampwidth != 3:
        raise ValueError("Expected 24-bit WAV")
    pcm16 = audioop.lin2lin(frames, 3, 2)
    gain = 10 ** (gain_db / 20.0)
    quiet = audioop.mul(pcm16, 2, gain)
    frames24 = audioop.lin2lin(quiet, 2, 3)
    with wave.open(str(dst_wav), "wb") as w:
        w.setparams((params.nchannels, 3, params.framerate, 0, params.comptype, params.compname))
        w.writeframes(frames24)


def mythic_cipherspray_watermark(src_wav: Path, dst_wav: Path, seed_hex: str) -> None:
    with wave.open(str(src_wav), "rb") as r:
        params = r.getparams()
        sr = r.getframerate()
        nframes = r.getnframes()
        frames = r.readframes(nframes)
    if params.sampwidth != 3 or params.nchannels != 2:
        raise ValueError("Expected 24-bit stereo WAV")
    pcm16 = audioop.lin2lin(frames, 3, 2)
    tone_hz = 19000.0 if sr >= 48000 else sr*0.39
    amp = 0.02
    bits = bin(int(seed_hex, 16))[2:].zfill(64)
    segment = 0.25
    seg_frames = int(segment * sr)
    out = bytearray()
    phase = 0.0
    for i in range(0, len(pcm16), 4):
        idx = i // 4
        k = (idx // seg_frames) % len(bits)
        a = amp if bits[k] == "1" else amp*0.4
        phase += 2*math.pi * tone_hz / sr
        s = int(max(-32768, min(32767, int(math.sin(phase) * a * 32767))))
        l = int.from_bytes(pcm16[i:i+2], "little", signed=True)
        r = int.from_bytes(pcm16[i+2:i+4], "little", signed=True)
        l = max(-32768, min(32767, l + s))
        r = max(-32768, min(32767, r + s))
        out += int(l).to_bytes(2, "little", signed=True)
        out += int(r).to_bytes(2, "little", signed=True)
    frames24 = audioop.lin2lin(bytes(out), 2, 3)
    with wave.open(str(dst_wav), "wb") as w:
        w.setparams(params)
        w.writeframes(frames24)

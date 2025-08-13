#!/usr/bin/env python3
import audioop
import sys
import wave
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import riddle as irr


def create_wav(path: Path, nframes: int = 1000, sr: int = 48000) -> None:
    pcm16 = bytearray()
    for i in range(nframes):
        sample = (i % 1000) - 500
        pcm16 += int(sample).to_bytes(2, "little", signed=True) * 2
    pcm24 = audioop.lin2lin(bytes(pcm16), 2, 3)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(3)
        w.setframerate(sr)
        w.writeframes(pcm24)


def test_duration_sampling_deterministic(root_seed: bytes):
    prng1 = irr.domain_prngs(root_seed)["form"]
    prng2 = irr.domain_prngs(root_seed)["form"]
    d1, b1 = irr.sample_duration_seconds(prng1, "glass", None)
    d2, b2 = irr.sample_duration_seconds(prng2, "glass", None)
    assert d1 == d2 and b1 == b2
    assert 20 <= d1 <= 3 * 3600


def test_form_generation_deterministic(root_seed: bytes):
    prng1 = irr.domain_prngs(root_seed)["form"]
    total, bucket = irr.sample_duration_seconds(prng1, "glass", None)
    path1, timeline1 = irr.pick_form(prng1, "glass", total)
    prng2 = irr.domain_prngs(root_seed)["form"]
    total2, bucket2 = irr.sample_duration_seconds(prng2, "glass", None)
    path2, timeline2 = irr.pick_form(prng2, "glass", total2)
    assert total == total2 and bucket == bucket2
    assert path1 == path2
    assert timeline1 == timeline2
    assert abs(timeline1[-1][2] - total) < 1e-6


def test_mythic_variants(tmp_path: Path):
    src = tmp_path / "src.wav"
    create_wav(src)
    with wave.open(str(src), "rb") as r:
        nframes = r.getnframes()
        frames = r.readframes(nframes)
        params = r.getparams()
    dst = tmp_path / "backmask.wav"
    irr.mythic_backmask(src, dst)
    with wave.open(str(dst), "rb") as r:
        assert r.getparams() == params
        out_frames = r.readframes(nframes)
    expected = audioop.reverse(audioop.lin2lin(frames, 3, 2), 2)
    assert audioop.lin2lin(out_frames, 3, 2) == expected
    variants = [
        ("ashen", lambda s, d: irr.mythic_ashen_bitcrush(s, d, bits=12)),
        ("liminal", irr.mythic_liminal_bed),
        ("cipherspray", lambda s, d: irr.mythic_cipherspray_watermark(s, d, "feedfacefeedface")),
    ]
    if hasattr(audioop, "sub"):
        variants.insert(1, ("mirrorsalt", irr.mythic_mirrorsalt_ms))
    for label, fn in variants:
        out = tmp_path / f"{label}.wav"
        fn(src, out)
        assert out.exists()
        with wave.open(str(out), "rb") as r:
            assert r.getnframes() == nframes


def test_pick_time_signature_odd_sequence(root_seed: bytes):
    prng_single = irr.domain_prngs(root_seed)["rhythm"]
    ts = irr.pick_time_signature(prng_single, "glass")
    assert ts[0] % 2 == 1

    prng_seq = irr.domain_prngs(root_seed)["rhythm"]
    seq = irr.pick_time_signature(prng_seq, "glass", chaotic=True)
    assert isinstance(seq, list) and seq
    for num, _ in seq:
        assert num % 2 == 1
    assert len(set(seq)) > 1


def test_prng_ctrl_deterministic(root_seed: bytes):
    prng1 = irr.domain_prngs(root_seed)["ctrl"]
    prng2 = irr.domain_prngs(root_seed)["ctrl"]
    vals1 = [prng1.uniform() for _ in range(3)]
    vals2 = [prng2.uniform() for _ in range(3)]
    assert vals1 == vals2


def test_prng_video_deterministic(root_seed: bytes):
    prng1 = irr.domain_prngs(root_seed)["video"]
    prng2 = irr.domain_prngs(root_seed)["video"]
    vals1 = [prng1.randbits(64) for _ in range(3)]
    vals2 = [prng2.randbits(64) for _ in range(3)]
    assert vals1 == vals2

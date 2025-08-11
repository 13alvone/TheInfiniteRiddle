#!/usr/bin/env python3
import audioop
import sys
import tempfile
import unittest
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import riddle as irr


class TestStemMixMatch(unittest.TestCase):
    def test_percussion_stems_sum_to_mix(self):
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        try:
            mix_path = root / "mix.wav"
            stem_paths = {
                "lead": root / "lead.wav",
                "pad": root / "pad.wav",
                "bass": root / "bass.wav",
                "perc": root / "perc.wav",
            }
            prng = irr.domain_prngs(bytes.fromhex("feedface" * 8))["synth"]
            midi = {
                "lead": [],
                "pad": [],
                "bass": [],
                "perc": [(t, 1, 0, 120) for t in range(0, 192, 2)],
            }
            irr.render_audio(
                mix_path,
                midi,
                120.0,
                96,
                8000,
                1,
                prng,
                stem_paths=stem_paths,
            )

            def pcm32(p: Path) -> bytes:
                with wave.open(str(p), "rb") as w:
                    frames = w.readframes(w.getnframes())
                return audioop.lin2lin(frames, 3, 4)

            mix_pcm = pcm32(mix_path)
            summed = bytes(len(mix_pcm))
            for p in stem_paths.values():
                summed = audioop.add(summed, pcm32(p), 4)
            self.assertEqual(mix_pcm, summed)
        finally:
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()

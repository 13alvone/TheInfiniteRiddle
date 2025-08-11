#!/usr/bin/env python3
import audioop
import unittest


class TestPCMConversion(unittest.TestCase):
    def test_16_to_24(self):
        frames = bytes.fromhex("0080ffff00000100ff7fc7cf39300000")
        expected = bytes.fromhex("00008000ffff00000000010000ff7f00c7cf003930000000")
        result = audioop.lin2lin(frames, 2, 3)
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()

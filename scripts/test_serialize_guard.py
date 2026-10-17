#!/usr/bin/env python3
"""Tests of scripts/serialize_guard.py on throwaway trees."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import serialize_guard  # noqa: E402

GATED = '#[derive(Debug)]\n#[cfg_attr(feature = "serde-raw", derive(Serialize, Deserialize))]\npub struct Fixed(pub i32);\n'
PLAIN = "#[derive(Debug, Clone, Serialize, Deserialize)]\npub struct Count(u64);\n"
REASON = "integer counts, no fixed-point value"


class SerializeGuardTest(unittest.TestCase):
    def tree(self, files: dict[str, str], toml: str) -> Path:
        d = Path(tempfile.mkdtemp())
        (d / "src").mkdir()
        (d / "scripts").mkdir()
        for name, text in files.items():
            (d / "src" / name).write_text(text, encoding="utf-8")
        (d / "scripts" / "serialize_types.toml").write_text(toml, encoding="utf-8")
        return d

    def test_gated_and_listed_pass(self):
        d = self.tree({"a.rs": GATED + PLAIN},
                      f'raw_only = 1\n[non_durable]\n"src/a.rs::Count" = "{REASON}"\n')
        self.assertEqual(serialize_guard.check(d), [])

    def test_ungated_unlisted_fails(self):
        d = self.tree({"a.rs": GATED + PLAIN}, "raw_only = 1\n[non_durable]\n")
        self.assertTrue(any("src/a.rs::Count" in p and "not listed" in p
                            for p in serialize_guard.check(d)))

    def test_removing_the_gate_fails(self):
        ungated = GATED.replace('#[cfg_attr(feature = "serde-raw", derive(Serialize, Deserialize))]\n', "") \
                       .replace("#[derive(Debug)]", "#[derive(Debug, Serialize, Deserialize)]")
        d = self.tree({"a.rs": ungated + PLAIN},
                      f'raw_only = 1\n[non_durable]\n"src/a.rs::Count" = "{REASON}"\n')
        problems = serialize_guard.check(d)
        self.assertTrue(any("src/a.rs::Fixed" in p for p in problems), problems)
        self.assertTrue(any("raw_only = 1" in p for p in problems), problems)

    def test_listed_but_gated_fails(self):
        d = self.tree({"a.rs": GATED},
                      f'raw_only = 1\n[non_durable]\n"src/a.rs::Fixed" = "{REASON}"\n')
        self.assertTrue(any("behind serde-raw" in p for p in serialize_guard.check(d)))

    def test_stale_entry_fails(self):
        d = self.tree({"a.rs": GATED},
                      f'raw_only = 1\n[non_durable]\n"src/a.rs::Gone" = "{REASON}"\n')
        self.assertTrue(any("stale" in p for p in serialize_guard.check(d)))

    def test_short_reason_fails(self):
        d = self.tree({"a.rs": PLAIN}, 'raw_only = 0\n[non_durable]\n"src/a.rs::Count" = "ints"\n')
        self.assertTrue(any("reason shorter" in p for p in serialize_guard.check(d)))

    def test_multi_line_derive_and_manual_impl_are_found(self):
        text = ("#[derive(\n    Debug,\n    Serialize,\n)]\npub struct Hash(pub u64);\n"
                "pub struct Manual;\nimpl Serialize for Manual {}\n")
        d = self.tree({"a.rs": text}, "raw_only = 0\n[non_durable]\n")
        problems = serialize_guard.check(d)
        self.assertTrue(any("src/a.rs::Hash" in p for p in problems), problems)
        self.assertTrue(any("src/a.rs::Manual" in p for p in problems), problems)

    def test_a_commented_derive_is_not_counted(self):
        text = "// #[derive(Serialize)]\npub struct Quiet;\n" + PLAIN
        d = self.tree({"a.rs": text}, f'raw_only = 0\n[non_durable]\n"src/a.rs::Count" = "{REASON}"\n')
        self.assertEqual(serialize_guard.check(d), [])

    def test_no_serde_type_fails(self):
        d = self.tree({"a.rs": "pub struct Plain;\n"}, "raw_only = 0\n[non_durable]\n")
        self.assertTrue(any("compared nothing" in p for p in serialize_guard.check(d)))

    def test_no_source_file_fails(self):
        d = self.tree({}, "raw_only = 0\n[non_durable]\n")
        self.assertTrue(any("compared nothing" in p for p in serialize_guard.check(d)))


if __name__ == "__main__":
    unittest.main()

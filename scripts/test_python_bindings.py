#!/usr/bin/env python3
"""Python-side tests for the `python` feature (src/python.rs).

Builds the crate as a Python extension for the interpreter running this script
(`cargo rustc --lib --crate-type cdylib --features python`, PYO3_PYTHON = this
interpreter), imports it as `alice_sync`, checks that every name the module registers is
exported, that version and semantics_id are the package version and the
SYNC_SEMANTICS_ID literal read from the source files, the input frame round
trip, the peer identifier check, a two-player lockstep step and the SoA world
hash. Needs numpy.

The extension is built in the debug profile on purpose: debug assertions make
pyo3 abort on a misaligned pyclass (contents aligned above CPython's 16 bytes),
which a release build would run as silent undefined behaviour (src/python.rs
also asserts the alignment of every pyclass at compile time).

Usage: python3 scripts/test_python_bindings.py
Exit 1 when the build fails, when a test fails or when no test ran.
"""

from __future__ import annotations

import importlib.util
import math
import os
import pathlib
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODULE = None  # set by load_module()


def build_extension() -> pathlib.Path:
    env = dict(os.environ)
    env["PYO3_PYTHON"] = sys.executable
    target_dir = pathlib.Path(env.get("CARGO_TARGET_DIR", ROOT / "target"))
    cmd = ["cargo", "rustc", "--lib", "--crate-type", "cdylib", "--features", "python"]
    profile = "debug"
    if sys.platform == "darwin":
        # the interpreter provides the Python symbols at import time; maturin passes the same
        # two arguments (a dependency's build script cannot add link arguments to this cdylib)
        cmd += ["--", "-C", "link-arg=-undefined", "-C", "link-arg=dynamic_lookup"]
    print("+", " ".join(cmd), f"(PYO3_PYTHON={sys.executable})", flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    out = target_dir / profile
    for name in ("libalice_sync.so", "libalice_sync.dylib", "alice_sync.dll"):
        if (out / name).is_file():
            return out / name
    raise SystemExit(f"no extension library found in {out}")


def load_module(lib: pathlib.Path, workdir: pathlib.Path):
    suffix = ".pyd" if sys.platform == "win32" else ".so"
    dest = workdir / f"alice_sync{suffix}"
    shutil.copyfile(lib, dest)
    spec = importlib.util.spec_from_file_location("alice_sync", dest)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {dest}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


import re

EXPORTED = ["InputFrame", "LockstepSession", "RollbackSession", "WorldSoA", "version", "semantics_id"]


def source_semantics_id() -> bytes:
    text = (ROOT / "src" / "semantics.rs").read_text()
    m = re.search(r'pub const SYNC_SEMANTICS_ID: \[u8; 32\] =\s*hex32\("([0-9a-f]{64})"\)', text)
    if m is None:
        raise AssertionError("SYNC_SEMANTICS_ID literal not found in src/semantics.rs")
    return bytes.fromhex(m.group(1))


class ExportTests(unittest.TestCase):
    def test_every_registered_name_is_exported(self):
        self.assertEqual([n for n in EXPORTED if not hasattr(MODULE, n)], [])

    def test_version_and_semantics_id(self):
        text = (ROOT / "Cargo.toml").read_text()
        pkg = text[text.index("[package]"):]
        self.assertEqual(MODULE.version(), pkg.split('version = "', 1)[1].split('"', 1)[0])
        self.assertEqual(bytes(MODULE.semantics_id()), source_semantics_id())


class InputFrameTests(unittest.TestCase):
    def test_round_trip_keeps_every_field(self):
        f = MODULE.InputFrame(7, 2, move_x=3, move_y=-4, move_z=5, actions=0b1011, aim_x=-1, aim_y=2, aim_z=-3)
        g = MODULE.InputFrame.from_bytes(bytes(f.to_bytes()))
        self.assertIsNotNone(g)
        for name in ("frame", "player_id", "movement", "actions", "aim"):
            self.assertEqual(getattr(g, name), getattr(f, name), name)
        self.assertEqual((g.frame, g.player_id, g.actions), (7, 2, 0b1011))

    def test_short_bytes_give_none(self):
        self.assertIsNone(MODULE.InputFrame.from_bytes(b"\x00\x01"))


class LockstepTests(unittest.TestCase):
    def test_peer_identifier_is_checked(self):
        with self.assertRaises(ValueError):
            MODULE.LockstepSession(2, bytes(32))
        with self.assertRaises(ValueError):
            MODULE.LockstepSession(2, source_semantics_id()[:31])

    def test_two_players_advance_one_frame(self):
        s = MODULE.LockstepSession(2, source_semantics_id())
        # the session starts confirmed at frame 0, so the next frame to advance to is 1
        self.assertEqual(s.confirmed_frame(), 0)
        s.add_local_input(MODULE.InputFrame(1, 0, move_x=1))
        self.assertFalse(s.ready_to_advance())
        self.assertIsNone(s.advance())
        s.add_remote_input(MODULE.InputFrame(1, 1, move_x=2))
        self.assertTrue(s.ready_to_advance())
        frames = s.advance()
        self.assertEqual([(f.player_id, f.frame, f.movement[0]) for f in frames], [(0, 1, 1), (1, 1, 2)])
        self.assertEqual(s.confirmed_frame(), 1)

    def test_checksums(self):
        s = MODULE.LockstepSession(2, source_semantics_id())
        s.record_checksum(5, 0xABCDEF)
        self.assertEqual(s.verify_checksum(5, 0xABCDEF), "ok")
        self.assertTrue(s.verify_checksum(5, 0x123).startswith("desync:frame=5,"))


class WorldTests(unittest.TestCase):
    def build(self):
        w = MODULE.WorldSoA(seed=1)
        a = w.spawn(10, 1, 100, 200, 300)
        b = w.spawn(11, 2, -5, 0, 7)
        return w, a, b

    def test_identical_worlds_hash_equal_and_motion_changes_the_hash(self):
        w1, _, _ = self.build()
        w2, _, _ = self.build()
        self.assertEqual(w1.entity_count(), 2)
        self.assertEqual(w1.hash(), w2.hash())
        before = w1.hash()
        w1.apply_motions([(10, 1, 0, 0)])
        self.assertNotEqual(w1.hash(), before)

    def test_positions_hold_the_spawned_coordinates(self):
        w, a, b = self.build()
        pos = w.positions()
        self.assertEqual(pos.shape, (w.capacity(), 3))
        self.assertEqual(pos[a].tolist(), [100, 200, 300])
        self.assertEqual(pos[b].tolist(), [-5, 0, 7])
        self.assertTrue(w.despawn(a))
        self.assertEqual(w.entity_count(), 1)


def main() -> int:
    global MODULE
    print(f"python {sys.version.split()[0]} ({sys.executable})", flush=True)
    lib = build_extension()
    with tempfile.TemporaryDirectory() as tmp:
        MODULE = load_module(lib, pathlib.Path(tmp))
        suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.testsRun == 0:
        print("no test ran", file=sys.stderr)
        return 1
    print(f"python bindings: {result.testsRun} tests, {len(result.failures)} failures, {len(result.errors)} errors")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())

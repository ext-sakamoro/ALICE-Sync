#!/usr/bin/env python3
"""Every serde-serializable type in src/ is either raw-only or declared not
durable (ALICE-Law G8).

The Law-bearing fixed-point state has one durable form, `alice_sync::persist`,
which stamps SYNC_SEMANTICS_ID into the bytes. A `Serialize` / `Deserialize`
implementation on such a type is a second, unstamped form, so it may exist only
behind the opt-in `serde-raw` feature:

    #[cfg_attr(feature = "serde-raw", derive(Serialize, Deserialize))]

Any other serde implementation (a plain `derive(Serialize ...)` or a manual
`impl Serialize for` / `impl Deserialize for`) must be listed in
scripts/serialize_types.toml under [non_durable] as `"<file>::<Type>" =
"<reason>"`: why its bytes carry no Law-bearing value, or why they are checked
at another boundary (G7).

The check fails when
  * a type has an ungated serde implementation and is not listed,
  * a listed type is gated, or no longer has a serde implementation (stale),
  * a reason is shorter than MIN_REASON characters,
  * the number of gated types differs from `raw_only` in the TOML
    (a type moved out from behind the gate, or a new one added, is a decision
    the file has to record), or
  * it found no serde implementation at all (it compared nothing).

usage: scripts/serialize_guard.py [--root DIR]
"""
from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

MIN_REASON = 12
GATE_RE = re.compile(r'#\[cfg_attr\(\s*feature\s*=\s*"serde-raw"\s*,\s*derive\(([^)]*)\)\s*\)\]')
DERIVE_RE = re.compile(r"#\[derive\(([^)]*)\)\]")
ITEM_RE = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum)\s+(\w+)")
IMPL_RE = re.compile(r"\bimpl\s*(?:<[^>]*>)?\s*(?:serde::)?(?:Serialize|Deserialize\s*<[^>]*>)\s+for\s+(\w+)")
SERDE = {"Serialize", "Deserialize"}


def strip_comments(text: str) -> str:
    """Comments removed and every multi-line attribute joined onto one line."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = "\n".join(line.split("//", 1)[0] for line in text.split("\n"))
    text = re.sub(r"#\[[^\]]*\]", lambda m: " ".join(m.group(0).split()), text, flags=re.S)
    # an item on the same line as its attribute (`#[derive(Deserialize)] pub struct S;`)
    return re.sub(r"(#\[[^\]\n]*\])[ \t]*(?=[^\s#])", "\\1\n", text)


def scan(root: Path) -> tuple[dict[str, str], int]:
    """`{"<file>::<Type>": "gated" | "ungated"}` and the number of files read."""
    found: dict[str, str] = {}
    files = sorted((root / "src").rglob("*.rs"))
    for path in files:
        rel = path.relative_to(root).as_posix()
        lines = strip_comments(path.read_text(encoding="utf-8")).split("\n")
        pending: str | None = None
        for line in lines:
            g = GATE_RE.search(line)
            d = DERIVE_RE.search(line)
            if g and SERDE & {x.strip() for x in g.group(1).split(",")}:
                pending = "gated" if pending in (None, "gated") else pending
            elif d and SERDE & {x.strip().split("::")[-1] for x in d.group(1).split(",")}:
                pending = "ungated"
            m = ITEM_RE.match(line)
            if m:
                if pending:
                    key = f"{rel}::{m.group(1)}"
                    found[key] = "ungated" if found.get(key) == "ungated" else pending
                pending = None
            elif line.strip() and not line.strip().startswith("#"):
                pending = None
            for im in IMPL_RE.finditer(line):
                found[f"{rel}::{im.group(1)}"] = "ungated"
    return found, len(files)


def check(root: Path) -> list[str]:
    cfg = tomllib.loads((root / "scripts" / "serialize_types.toml").read_text(encoding="utf-8"))
    listed: dict[str, str] = cfg.get("non_durable", {})
    raw_only = cfg.get("raw_only")
    found, n_files = scan(root)
    problems = []
    if n_files == 0 or not found:
        return [f"no serde implementation found in {n_files} files under src/ (compared nothing)"]
    gated = sorted(k for k, v in found.items() if v == "gated")
    for key, kind in sorted(found.items()):
        if kind == "ungated" and key not in listed:
            problems.append(f"{key}: Serialize / Deserialize without the serde-raw gate and not "
                            f"listed in scripts/serialize_types.toml [non_durable]")
    for key, reason in sorted(listed.items()):
        if key not in found:
            problems.append(f"{key}: listed but has no serde implementation (stale entry)")
        elif found[key] == "gated":
            problems.append(f"{key}: listed as non-durable but its serde derive is behind serde-raw")
        if len(reason.strip()) < MIN_REASON:
            problems.append(f"{key}: reason shorter than {MIN_REASON} characters")
    if raw_only != len(gated):
        problems.append(f"raw_only = {raw_only} in scripts/serialize_types.toml, found {len(gated)} "
                        f"gated types: {', '.join(gated)}")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    a = ap.parse_args(argv)
    root = Path(a.root)
    problems = check(root)
    for p in problems:
        print(f"serialize-guard: {p}", file=sys.stderr)
    if problems:
        return 1
    found, n = scan(root)
    gated = sum(1 for v in found.values() if v == "gated")
    print(f"serialize-guard: {n} files, {len(found)} serde types: {gated} behind serde-raw, "
          f"{len(found) - gated} listed non-durable, 0 problems")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env bash
# scripts/preflight.sh — local reproduction of the CI gates before `git push`.
# Every command is the one CI runs. A step this file does not cover is a step
# that can only fail remotely — when a workflow step is added, add it here in
# the same commit. Run `--quick` before every push.
#
# usage: scripts/preflight.sh [--quick]   (--quick skips the test / bench suites)
set -euo pipefail
cd "$(dirname "$0")/.."

quick=0
[[ "${1:-}" == "--quick" ]] && quick=1

step() { printf '\n\033[1;34m== %s\033[0m\n' "$*"; }
# `cargo clippy` reuses fresh `cargo check` artifacts and then lints nothing;
# touching the crate roots invalidates only this repo's fingerprints.
relint() { git ls-files | grep -E '(^|/)src/(lib|main)\.rs$' | xargs -r touch; }
need() { command -v "$1" >/dev/null 2>&1 || { echo "missing tool: $1 ($2)" >&2; exit 1; }; }
has_toolchain() { rustup toolchain list | grep -q "^$1"; }

# Steps CI runs that this file cannot reproduce locally (they can only fail remotely):
#   - security-audit.yml:audit:Install cargo-audit (needs network / runner-only)
#   - security-audit.yml:deny:Install cargo-deny (needs network / runner-only)
#   - security-audit.yml:coverage (job is continue-on-error: informational in CI)
#   - fuzz.yml:fuzz:Install cargo-fuzz (needs network / runner-only)
#   - fuzz.yml:fuzz:Build fuzz target (needs network / runner-only)
#   - fuzz.yml:fuzz:Set fuzz duration (no cargo / grep)
#   - fuzz.yml:fuzz:Run fuzz target (time-boxed) (continue-on-error)
#   - fuzz.yml:fuzz:Report crash (informational) (no cargo / grep)

need actionlint "brew install actionlint"
need cargo-audit "cargo install cargo-audit --locked"
need cargo-deny "cargo install cargo-deny --locked"
need cargo-hack "cargo install cargo-hack --locked"
need cargo-machete "cargo install cargo-machete --locked"
has_toolchain 1.88 || { echo "missing toolchain 1.88 (rustup toolchain install 1.88)" >&2; exit 1; }

step "ci.yml / fmt: run"
( export CARGO_TERM_COLOR="always" CARGO_INCREMENTAL="0" RUST_BACKTRACE="short" NATIVE_FEATURES="std,async,simd,cloud,physics,telemetry,cache,auth,codec,analytics,ffi,serde-raw"; cargo fmt --all -- --check )

step "ci.yml / clippy: Clippy (default)"
relint
( export CARGO_TERM_COLOR="always" CARGO_INCREMENTAL="0" RUST_BACKTRACE="short" NATIVE_FEATURES="std,async,simd,cloud,physics,telemetry,cache,auth,codec,analytics,ffi,serde-raw"; cargo clippy --all-targets -- -D warnings )

step "ci.yml / clippy: Clippy (all native features)"
relint
( export CARGO_TERM_COLOR="always" CARGO_INCREMENTAL="0" RUST_BACKTRACE="short" NATIVE_FEATURES="std,async,simd,cloud,physics,telemetry,cache,auth,codec,analytics,ffi,serde-raw"; cargo clippy --all-targets --features "$NATIVE_FEATURES" -- -D warnings )

step "ci.yml / clippy: Clippy (no default features)"
relint
( export CARGO_TERM_COLOR="always" CARGO_INCREMENTAL="0" RUST_BACKTRACE="short" NATIVE_FEATURES="std,async,simd,cloud,physics,telemetry,cache,auth,codec,analytics,ffi,serde-raw"; cargo clippy --lib --no-default-features -- -D warnings )

step "ci.yml / clippy: Clippy canary data (generator tests)"
python3 scripts/test_gen_clippy_canary_calls.py

step "ci.yml / clippy: Clippy canary data (generated from clippy.toml, no drift)"
python3 scripts/gen_clippy_canary_calls.py --check

step "ci.yml / clippy: Clippy canary (tests)"
python3 scripts/test_clippy_canary.py

step "ci.yml / clippy: Clippy canary (every planted banned call is flagged)"
python3 scripts/clippy_canary.py

step "ci.yml / test: Build (no default features)"
( export CARGO_TERM_COLOR="always" CARGO_INCREMENTAL="0" RUST_BACKTRACE="short" NATIVE_FEATURES="std,async,simd,cloud,physics,telemetry,cache,auth,codec,analytics,ffi,serde-raw"; cargo build --lib --no-default-features )

step "ci.yml / msrv: Check (default + all native features)"
(
  export CARGO_TERM_COLOR="always" CARGO_INCREMENTAL="0" RUST_BACKTRACE="short" NATIVE_FEATURES="std,async,simd,cloud,physics,telemetry,cache,auth,codec,analytics,ffi,serde-raw"
  cargo +1.88 check --lib
  cargo +1.88 check --lib --features "$NATIVE_FEATURES"
)

step "ci.yml / doc: run"
( export CARGO_TERM_COLOR="always" CARGO_INCREMENTAL="0" RUST_BACKTRACE="short" NATIVE_FEATURES="std,async,simd,cloud,physics,telemetry,cache,auth,codec,analytics,ffi,serde-raw" RUSTDOCFLAGS="-Dwarnings"; cargo doc --no-deps --features "$NATIVE_FEATURES" )

step "ci.yml / feature-powerset: Check every feature pair (python 除外)"
( export CARGO_TERM_COLOR="always" CARGO_INCREMENTAL="0" RUST_BACKTRACE="short" NATIVE_FEATURES="std,async,simd,cloud,physics,telemetry,cache,auth,codec,analytics,ffi,serde-raw"; cargo hack check --lib --feature-powerset --depth 2 --exclude-features python --no-dev-deps )

step "ci.yml / entropy-guard: Entropy guard (tests, then src/)"
python3 scripts/test_entropy_guard.py
python3 scripts/entropy_guard.py

step "ci.yml / entropy-guard: Serialize guard (G8, tests, then src/)"
python3 scripts/test_serialize_guard.py
python3 scripts/serialize_guard.py

step "ci.yml / entropy-guard: libm dependency guard (test, then check)"
python3 scripts/test_libm_dependency_guard.py
python3 scripts/libm_dependency_guard.py
python3 scripts/test_build_config_guard.py
python3 scripts/build_config_guard.py
python3 scripts/test_build_flags_check.py
python3 scripts/test_run_tests.py

step "ci.yml / actionlint: actionlint"
actionlint .github/workflows/*.yml

step "ci.yml / actionlint: workflow-timeouts (every job has timeout-minutes)"
python3 scripts/test_workflow_timeouts.py
python3 scripts/workflow_timeouts.py

step "ci.yml / test: changed_paths.py (tests)"
python3 scripts/test_changed_paths.py

step "security-audit.yml / deny: Run cargo deny check all"
( export CARGO_TERM_COLOR="always" NATIVE_FEATURES="std,async,simd,cloud,physics,telemetry,cache,auth,codec,analytics,ffi,serde-raw" CARGO_NET_RETRY="5" CARGO_HTTP_MULTIPLEXING="false"; cargo deny --features "$NATIVE_FEATURES" check all )

step "security-audit.yml / unused-deps: cargo machete"
cargo machete

step "security-audit.yml / stub-guard: Block panic!(STUB) in src/**"
(
  export CARGO_TERM_COLOR="always" NATIVE_FEATURES="std,async,simd,cloud,physics,telemetry,cache,auth,codec,analytics,ffi,serde-raw" CARGO_NET_RETRY="5" CARGO_HTTP_MULTIPLEXING="false"
  set -eo pipefail
  hits=$(grep -rnE 'panic!\([^)]*STUB' \
    src/ --include="*.rs" \
    --exclude-dir=bin \
    | grep -v ':[[:space:]]*//' \
    | grep -vE ':[[:space:]]*/\*' \
    || true)
  if [ -n "$hits" ]; then
    echo "❌ Explicit STUB panic detected in src/ (production path):"
    echo "$hits"
    exit 1
  fi
  echo "✓ No panic!(STUB) in src/"
)

step "security-audit.yml / stub-guard: Detect todo! / unimplemented! (informational, not blocking)"
(
  export CARGO_TERM_COLOR="always" NATIVE_FEATURES="std,async,simd,cloud,physics,telemetry,cache,auth,codec,analytics,ffi,serde-raw" CARGO_NET_RETRY="5" CARGO_HTTP_MULTIPLEXING="false"
  set -eo pipefail
  hits=$(grep -rnE 'todo!\(|unimplemented!\(' \
    src/ --include="*.rs" \
    --exclude-dir=bin \
    | grep -v ':[[:space:]]*//' \
    | grep -vE ':[[:space:]]*/\*' \
    || true)
  if [ -n "$hits" ]; then
    count=$(echo "$hits" | wc -l | tr -d ' ')
    echo "::warning::${count} todo!()/unimplemented!() marker(s) in src/ (informational, legitimate fail-fast idiom):"
    echo "$hits" | head -20
  else
    echo "✓ No todo!/unimplemented! markers in src/"
  fi
)

step "security-audit.yml / stub-guard: Detect dbg!() residual in src/**"
(
  export CARGO_TERM_COLOR="always" NATIVE_FEATURES="std,async,simd,cloud,physics,telemetry,cache,auth,codec,analytics,ffi,serde-raw" CARGO_NET_RETRY="5" CARGO_HTTP_MULTIPLEXING="false"
  set -eo pipefail
  hits=$(grep -rn 'dbg!(' src/ --include="*.rs" || true)
  if [ -n "$hits" ]; then
    echo "❌ dbg!() macro left in src/:"
    echo "$hits"
    exit 1
  fi
  echo "✓ No dbg!() in src/"
)

step "security-audit.yml / stub-guard: Detect TODO / FIXME / XXX / HACK (informational)"
(
  export CARGO_TERM_COLOR="always" NATIVE_FEATURES="std,async,simd,cloud,physics,telemetry,cache,auth,codec,analytics,ffi,serde-raw" CARGO_NET_RETRY="5" CARGO_HTTP_MULTIPLEXING="false"
  set -eo pipefail
  hits=$(grep -rnE 'TODO|FIXME|XXX|HACK' src/ --include="*.rs" || true)
  if [ -n "$hits" ]; then
    echo "::warning::TODO/FIXME/XXX/HACK found in src/ (informational, not blocking):"
    echo "$hits" | head -50
  else
    echo "✓ No TODO/FIXME/XXX/HACK in src/"
  fi
)

step "fuzz.yml / build every fuzz target (nightly; the replay needs the runner)"
if has_toolchain nightly && cargo +nightly fuzz --version >/dev/null 2>&1; then
  (cd fuzz && cargo +nightly fuzz build)
else
  echo "skip: nightly / cargo-fuzz not installed" >&2
fi

step "public-text.yml / public-text: Public text (ALICE-Law checker at the pin; tree, then the commits)"
python3 scripts/test_public_text.py
scripts/public_text.sh

step "ci.yml / test: No negated fma result (syntax tree, tests then check)"
cargo test --quiet --locked --manifest-path tools/no-outer-fneg-of-fma/Cargo.toml
cargo run --quiet --locked --release --manifest-path tools/no-outer-fneg-of-fma/Cargo.toml -- --root .

if [[ $quick -eq 1 ]]; then
  echo; echo "preflight --quick OK (test / bench suites skipped)"; exit 0
fi

step "ci.yml / test: Test (default = std)"
( export CARGO_TERM_COLOR="always" CARGO_INCREMENTAL="0" RUST_BACKTRACE="short" NATIVE_FEATURES="std,async,simd,cloud,physics,telemetry,cache,auth,codec,analytics,ffi,serde-raw"; cargo test )

step "ci.yml / test: Test (all native features)"
( export CARGO_TERM_COLOR="always" CARGO_INCREMENTAL="0" RUST_BACKTRACE="short" NATIVE_FEATURES="std,async,simd,cloud,physics,telemetry,cache,auth,codec,analytics,ffi,serde-raw"; cargo test --features "$NATIVE_FEATURES" )

step "ci.yml / test: Analytic oracle (async + physics + codec)"
( export CARGO_TERM_COLOR="always" CARGO_INCREMENTAL="0" RUST_BACKTRACE="short" NATIVE_FEATURES="std,async,simd,cloud,physics,telemetry,cache,auth,codec,analytics,ffi,serde-raw"; cargo test --test analytic_oracle --features "async,physics,codec" )

step "ci.yml / test: Float environment (debug)"
scripts/run_tests.sh float_environment=2 -- cargo test --test float_environment

step "ci.yml / test: Doc tests"
( export CARGO_TERM_COLOR="always" CARGO_INCREMENTAL="0" RUST_BACKTRACE="short" NATIVE_FEATURES="std,async,simd,cloud,physics,telemetry,cache,auth,codec,analytics,ffi,serde-raw"; cargo test --doc --features "$NATIVE_FEATURES" )

step "release-profile.yml / release-tests: Release-profile golden and oracle tests"
( export CARGO_TERM_COLOR="always" CARGO_INCREMENTAL="0" RUST_BACKTRACE="short" RELEASE_TEST_FEATURES="async,physics,codec,cloud,telemetry"; scripts/run_tests.sh float_edge_golden,analytic_oracle,float_environment=2 -- cargo test --release --features "$RELEASE_TEST_FEATURES" --test float_edge_golden --test analytic_oracle --test float_environment )

step "release-profile.yml / release-tests: Effective rustc flags (G14)"
python3 scripts/build_flags_check.py -- cargo check --release --lib
python3 scripts/build_flags_check.py -- cargo check --release --lib --all-features
python3 scripts/build_flags_check.py -- cargo check --release --lib --no-default-features

step "security-audit.yml / audit: Run cargo audit"
(
  export CARGO_TERM_COLOR="always" NATIVE_FEATURES="std,async,simd,cloud,physics,telemetry,cache,auth,codec,analytics,ffi,serde-raw" CARGO_NET_RETRY="5" CARGO_HTTP_MULTIPLEXING="false"
  cargo audit --deny yanked \
    --ignore RUSTSEC-2024-0436 \
    --ignore RUSTSEC-2025-0020 \
    --ignore RUSTSEC-2026-0176 \
    --ignore RUSTSEC-2026-0177 \
    --ignore RUSTSEC-2026-0204 \
    --ignore RUSTSEC-2026-0235
)

step "semver-checks.yml / semver-checks: semver gate (tests)"
python3 scripts/test_semver_gate.py

step "semver-checks.yml / semver-checks: semver gate (latest crates.io release, every consumer feature set)"
need cargo-semver-checks "cargo install cargo-semver-checks --version 0.50.0 --locked"
python3 scripts/semver_gate.py --config .github/semver-gate.toml --target-dir "${CARGO_TARGET_DIR:-target}/semver"

step "g3-ir.yml / ir: probe tests + no negated fma result in alice-sync's release IR (host build; the per-target floors are checked in CI)"
python3 scripts/test_fneg_fma_ir_check.py
python3 scripts/fneg_fma_ir_check.py --package alice-sync --all-features --findings-only

echo; echo "preflight OK"

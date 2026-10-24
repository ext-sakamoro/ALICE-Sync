# Changelog

All notable changes to ALICE-Sync will be documented in this file.

## [Unreleased]

### 破壊的変更

- 保存形式 2 (`persist::FORMAT_VERSION`): `WorldState` の本体を、生存する entity (id の昇順)、frame、seed で書く slot・generation・free list は書かない 形式 1 は `FormatStatus::Legacy(1)` として読み、`decode_state_strict` は `PersistError::LegacyFormat` で拒む `Decoded` に `format`、`PersistError` に `LegacyFormat` を足した `Handle` は保存と復元の後に保たれない
- `WorldState::id_map` を非公開にした 生存する entity の id と handle は `WorldState::ids` (id の昇順)、数は `WorldState::live_count`
- `SyncError` に `TooManyEntities` を足した `World::apply` と `WorldSoA::apply` は `world::MAX_LIVE_ENTITIES` (65536) を超える新しい id の Spawn を拒み、保存形式もその数を超える件数を拒む
- `WorldSoA` の位置を `Fixed` の Q16.16 (event の Q8.8 の値を `<< 8`) にした (以前は `<< 6`) slot の hash は `Entity::compute_hash` と同じ式 (active の mask を含む)
- 生存する id の Spawn は、`World` と `WorldSoA` の両方で前の entity を置き換える (以前は前の entity が storage に残った)
- `converge`: 記録は event が名指す id の小さい `MAX_ENTITY_RECORDS` 個、world は生存する id の小さい `MAX_LIVE_ENTITIES` 個 (届く順に依らない、`Converge::with_limits` で小さくできる)

### Added
- `WorldSoA::to_state` / `WorldSoA::from_state`、`WorldStorage::entity_at`、`Converge::with_limits`、`persist::decode_state_strict`、`persist::FormatStatus`、`persist::LEGACY_FORMAT_VERSIONS`
- 試験: `tests/persist_canonical.rs` (受信順の違う node の保存 bytes)、`tests/untrusted_entity_id_alloc_bound.rs` (上限 16 MiB の allocator で、他の node の entity id と保存された件数で確保する量)、`tests/g13_bounds.rs` (上限つきの world を模型と比べる)、`tests/keyed_lookup.rs` (key の違う表で同じ保存 bytes と hash)、`tests/backends_agree.rs` (`World` と `WorldSoA` の値と保存 bytes)、`tests/raw_serde_has_stamped_path.rs`、形式 1 の固定 file `tests/data/persist/world_state_v1.bin`
- feature `secure-transport`: `secure_transport::SecureTransport` は `SyncTransport` を ALICE-Channel の認証・暗号化した channel の上に実装する 相手は handshake で `PeerPolicy` が知る master key の証明書を持つ端末の鍵を示し、発信側は受け側の鍵の確認の後に channel を得る 接続は ALICE-Channel の Noise IK で、dial には相手の証明書 (ALICE-Identity の DeviceCert version 2) が要る `add_peer_cert` / `connect(addr, cert)` が address ごとの証明書を帳簿に置き (置き換えは `add_peer_cert` だけ)、dial のたびに TCP 接続の前に今の policy で検査する 帳簿に証明書の無い address への送信は `TransportError::NoPeerCert` で拒み、何も dial しない `send_to` は帳簿にあって接続の無い address に dial する message は `MAX_MESSAGE` (1 MB、`TcpTransport` と同じ上限) まで、channel の message 1 つに収まらない分は最初の message の種別 (1 byte: `KIND_MESSAGE` = 0 は Sync の message、`KIND_BYTES` = 1 は application の bytes) と長さ (4 byte、little-endian) に続けて順に送り、1 つの相手への送信は lock の下で交互に混ざらない 失敗した接続 (I/O、拒まれた / 読めない channel の message、0 / 1 以外の種別、Sync の message として読めない message の種別の本文 (`malformed_count` で数える)、上限を超える長さ、終わらなかった送信) は捨て、次の `send_to` が dial し直す `accept` は handshake を `DEFAULT_HANDSHAKE_TIMEOUT` (10 秒、`bind_with_timeout` で指定) で打ち切り、拒んだ数を `refused_count` で返す `serve` は拒否で止まらない accept の loop 試験 `tests/secure_transport.rs` (13 本、実際の TCP の上: 両方向、dial、channel の message 3 つを超える message、8 つの同時の大きな送信、上限を超える送信の拒否、知らない端末、黙る接続、枠を壊す相手 4 通り、手で書いた枠の往復、切れた相手の dial し直し) 変異 11 件 (拒否で serve が止まる、受信の上限・先頭の長さ・続きの長さの検査を外す、送信の lock を chunk ごとに、送信の上限を外す、dial しない、切れた相手を残す、accept の期限を外す、拒否を数えない、長さを big-endian に) はすべて失敗する `recv_verified` は受けた message を `VerifiedEnvelope` (`body`: `Body::Message` / `Body::Bytes`、`from`、その接続の handshake で検証した証明書 `peer`) で返す `send_bytes_to` は application の bytes を `KIND_BYTES` で送り、受け側は Sync の message として decode しない `SyncTransport::recv` は Sync の message だけを返し、間の bytes は飛ばして `skipped_bytes_count` で数える `close` は address の接続を閉じ (相手の reader も終わる)、`revalidate` は開いている接続を今の policy (`now`、master key、revocation list) で検査し直して拒むものを閉じる `dialer` は listener を開かない transport を作る (`accept` / `serve` / `local_addr` は `NotConnected`) 試験 `tests/secure_transport.rs` は 22 本 (Sync の message として読めない本文で接続を閉じる、帳簿に無い address への送信の拒否と帳簿の置き換え、bytes の往復と証明書、Sync の message の bytes を bytes のまま受ける、上限を超える bytes の送信の拒否と受信の切断、0 / 1 以外の種別 3 通りの切断、手で書いた両種別の枠) と `tests/secure_transport_control.rs` 5 本 (dial ごとの policy の検査、message ごとの証明書、両側で閉じる close、master key の撤回と期限切れで閉じる revalidate、listener の無い dialer への返信) 変異 11 件 (帳簿を置き換えない、dial の前の policy の検査を外す、証明書の無い address を Codec の error にする、connect が帳簿に置かない、種別の検査を外す、bytes を decode する、revalidate が閉じない、close が reader を止めない、期限を見ない、飛ばした bytes を数えない、bytes の上限を外す) はすべて失敗する
- `conformance/g11-own.toml`: ALICE-Law G11 の表のうち、この crate 固有の行 (G4 / G5 の golden、G6 の識別子、比較 0 件は `scripts/run_tests.sh` の試験、G10 の `scripts/entropy_guard.py` とその試験) 共有の検査器の行は ALICE-Law の `conformance/g11-shared-rows.toml` から生成する
- CI (ALICE-Law G11): test job の default と全 native feature の `cargo test` を `scripts/run_tests.sh determinism_golden,float_edge_golden,sync_semantics_id,g13_receive_order,g13_order_determinism` で走らせ、名指した golden・識別子・順序の試験の binary が 1 件も走らない時に step が落ちるようにした release-profile.yml の golden の段に `determinism_golden` を足した `tests/determinism_golden.rs` の 4 つの golden は 1 つの照合 (`golden_mismatch`) を通り、`checker_rejects_a_wrong_digest` が動いた bit の digest が記録と一致しないことを確かめる (照合を常に一致にする変異でこの試験だけが失敗する)
- CI (ALICE-Law G11 (c)): ほかの ALICE の crate と同じ bytes で持つ file を ALICE-Law の表と比べる `scripts/shared_files.sh` (試験 `scripts/test_shared_files.py`、ALICE-DetMath などと同じ中身) と `.github/shared-files-pin` を足し、test job (3 OS) と preflight で走らせる 表と検査器は pin の版で取得し SHA-256 を確かめる 差、表に無い共有の path、比較 0 件は失敗
- 受信順に依らない world (ALICE-Law G13): `Node::apply_event` は新しい `converge` module を通して world を作る world は受け取った別々の event の集合で決まり、届く順と重複した配送に依らない event は Lamport 時刻・origin・sequence 番号の順 (`converge::Key`) で並べ、同じ key の別々の event (同じ id で作った 2 つの node が刻んだもの) は中身で並べるので、その衝突も届く順に依らず 1 つの world になる (衝突は誤用で、error ではなく決定的に収束する、中身に意味を持たせるには node の id が node ごとに違うことが前提) 重複の検出に専用の構造は持たず、以下の最大値と集合が同じ event の 2 度目を吸収する 実体ごとに、key (同じ key では kind と位置) が最大の Spawn がその実体の世代で、それより key の大きい Despawn が世代を終える (遅れて届いた Spawn が蘇らせないよう Despawn の key を残す) 位置はその Spawn の位置にそれより key の大きい別々の Motion の和を足したもの (Spawn より先に届いた Motion も保持し、Spawn が届いた時に足す、key が Spawn 以下の Motion は前の世代のもので足さない) property は Spawn より後で (key、値) が最大の書き込みの値 (Spawn より先に届いた書き込みも保持する) frame は (key、frame) が最大の Tick の値 保持する量: 実体ごとの最大の Spawn と Despawn、最大の property、最大の Spawn より key の大きい Motion の集合 (より大きい Spawn が届くと、それより前の Motion を捨てる)、frame 1 つ 1 世代の中の Motion は上限を持たない (捨てるには全員が通過した時点、causal stability が要り、この module はそれを追わない) `EventStream` は Lamport 時刻を持ち、`EventStream::observe` で他の node の event の時刻より後に進め、`push` はその時刻以上を刻む (observe しない stream の時刻は従来どおり sequence 番号) `Node::apply_event` は受け取った event の時刻を observe する 試験 `tests/g13_receive_order.rs`: 3 つの origin の scenario (同時刻・同 sequence の同じ Motion、同時刻の property の書き込み 2 つ、Spawn の前に届く Motion と property、再 Spawn、Spawn より key の小さい Despawn、最後に Despawn される実体、各 origin の Tick) を、event を 1 度ずつ key の順に素の `World` に当てた模型 (何も捨てず重複も持たない) と比べる 7 event の全 5040 通りの順、重複を最大 1/3 混ぜた 2000 通りの seed 付き shuffle、カウンタでは負けるが因果では後の書き込みが勝つ場合、Spawn より先に届く Motion と property をそれぞれ確かめる 加えて seed 付きの乱数履歴 3000 回 (4 origin、各 25 event 以下、時刻 0 から 11 で多くの key が時刻で並ぶ、実体 4 つ、全種類、8 つに 1 つに同じ key の別の event、9 割を届け約 1/4 を重複させ 2 通りに shuffle、計 6000 回比較)、同じ key の 2 つの event を種類ごとに両方の順で当てる試験、scenario と 5040 通りの順にも衝突を 1 組入れ、刻印の無い event の拒否を確かめる 模型は同じ key では Spawn を最後に、ほかは中身の昇順で当てる 変異 14 件 (届いた順で property を上書き、property の値の比較を逆に、Despawn を残さない、Despawn の key 比較を外す、key から origin を外す、Motion を key だけで先着の 1 つにする、重複を検出しない、Spawn より前の Motion を残す、Spawn と同じ key の Motion を足す、Spawn より前の property を world に当てる、再構築で Spawn より前の property も当てる、Spawn を key だけで先着にする、Tick を key だけで先着にする、observe しない) はすべて失敗する `tests/determinism_golden.rs` に golden `receive` (3 つの peer の event を逆順に、3 つに 1 つを 2 度届けた node の world) を追加し、`SYNC_SEMANTICS_PINS` に加えた `SYNC_SEMANTICS_ID` は c844437e… から 705744d7… に移る (0.6.1 は識別子を持たない) 計測は 1 台の arm64 機で 1 回、同じ bench code を main と比べた (負荷により run ごとに変わる): 1 つの実体への Motion の emit 32.1 → 120.3 ns (3.7 倍、bench が同じ世代に Motion を足し続けるので集合が大きくなる、1 件あたり O(log n))、他の node が emit した 1100 event の適用 10.8 → 42.8 µs (4.0 倍)
- CI (ALICE-Law G13): library の code は `World` / `WorldSoA` / `WorldStorage` を直接変えない (`World::apply` / `get_entity_mut` / batch の適用 3 つ、`WorldSoA::apply` / Motion の適用 7 つ / `spawn_raw` / `despawn_slot`、`WorldStorage::spawn` / `despawn` / `set_position` / `add_position` / `set_property` / `store_pos_x_8` / `store_pos_y_8` / `store_pos_z_8` / `add_positions_8`) 禁止は library 用の clippy 設定 `.github/clippy-lib/clippy.toml` (root の `clippy.toml` に 24 entry を足したもの、`scripts/clippy_lib_gate.py` が生成) に置き、test・example・bench は root の設定のまま world を直接使える 例外は sink の `converge`、world 型自身の module の合成、単一の書き手の C-ABI と Python binding (Python の binding は transport も Node も持たず、他の node の event は届かない) で、各箇所に理由付きの allow を書いた library の clippy は default・全 native feature・`python` の 3 通りで走る `scripts/clippy_lib_gate.py` と試験 `scripts/test_clippy_lib_gate.py` (ALICE-LOL などと同じ中身) は canary crate `scripts/clippy-lib-canary` で 24 entry それぞれの診断を確かめる clippy job と preflight で走る
- CI: text の file の行末の検査 `scripts/check_eol.py` (試験 `scripts/test_check_eol.py`) と `.github/workflows/eol.yml` を足した `git ls-files --eol` で text の file が index か working tree で CRLF / 混在なら失敗 (`-text` と binary は除く、比べた数を出し 0 件は失敗)、workflow は ubuntu / macos / windows で走り、preflight にも同じ検査を足した (ALICE の crate で同じ中身の file)
- CI (ALICE-Law G9): `scripts/id_move_check.py` (ALICE の識別子を持つ crate で共有、ALICE-Zip と同じ file) を追加 最新の release tag の公開版と作業 tree で `SYNC_SEMANTICS_ID` の値を読み、動いたのに破壊的な版になっていない / 公開版の値が `PREVIOUS_SYNC_SEMANTICS_IDS` に無い / CHANGELOG が両方の値を書いていない時に fail する (ci.yml の clippy job と preflight)
- 握手 (ALICE-Law G7): `Message::Hello` が送り手の `SYNC_SEMANTICS_ID` を持ち、`SyncSession` は受け入れた Hello の前の message を拒否する (`TransportError::Handshake` / `SessionEvent::MessageRefused`、`HandshakeRefused::SemanticsMismatch` は両方の識別子を持つ、旧識別子は `peer_status` が `IdStatus::VerifiedPrevious` を返す) `PhysicsRollbackSession::new` は相手の alice-physics の識別子を受けて照合する C-ABI は `as_sync_semantics_id` を追加し、`as_sync_lockstep_new` / `as_sync_rollback_new` / `as_sync_node_add_peer` が相手の識別子 (32 byte) を受けて照合する (不一致は null か -2 と `as_sync_last_error`) Python は `alice_sync.semantics_id()` を追加し、`LockstepSession` / `RollbackSession` が `peer_semantics_id` を受ける (不一致は ValueError) CI に `scripts/api_boundary.py` (rustdoc JSON から公開 API の境界を列挙する: bytes・文字列・reader を取るか返す関数、C-ABI、serde / FromStr / Read の実装と bytes からの From / TryFrom、外部の再 export、197 件それぞれに照合とその試験か理由付きの local の判断があること) と、文字列で見る第 2 層の `scripts/boundary_guard.py` (C-ABI・Python・bytes の decoder の全入口に、照合とその不一致試験か理由付きの local の判断があること、118 件) と試験を追加 `alice-law-conformance.toml` の G7 を pass にした
- `persist` module (ALICE-Law G8): Law を担う固定小数点の状態 (`Fixed` / `Vec3Fixed` / `Entity` / `WorldState`、`async` のとき `LwwRegister<V>`) の保存形式 `encode_state` が magic `ASYN`・形式の版 1・書いた側の `SYNC_SEMANTICS_ID` (32 byte)・型の tag・本体を書き、`decode_state` が保存された識別子を `check_semantics` で比べる (自分の識別子は `IdStatus::Current`、`PREVIOUS_SYNC_SEMANTICS_IDS` にあるものは `IdStatus::VerifiedPrevious`、それ以外は両方の識別子を持つ `PersistError::Semantics`) 読んだ `WorldState` は free list と id map の不変条件を確かめ、壊れていれば `PersistError::Malformed` を返す (未検査の accessor が前提にする条件) `semantics` module に `check_semantics` / `IdStatus` / `SemanticsMismatch` を追加 試験 `tests/persist_roundtrip.rs` (全型の往復と再 encode の byte 一致、別の識別子の拒否、header と長さの誤り、不変条件ごとの拒否) CI に `scripts/serialize_guard.py` (src/ の serde 型はすべて `serde-raw` の後ろか、理由付きで非永続の一覧 `scripts/serialize_types.toml` にあること、gate の後ろの型数を固定、0 件で失敗) と試験を追加 `alice-law-conformance.toml` の G8 を pass にした
- CI (`clippy.toml`、ALICE-Law G2): `f32::max` / `f32::min` / `f64::max` / `f64::min` を disallowed-methods に追加 等しい大きさで符号の違う 0 の組で返す符号が target によって違うため 使う時は `alice_det_math::max` / `min` / `max64` / `min64` (maximumNumber / minimumNumber) を使う 現状 src・test・example・bench に浮動小数点の max / min は 0 件で、挙動の変更なし clippy の canary の表を再生成し (62 行)、4 行それぞれが診断されることを CI が確かめる SIMD は整数 lane (`wide::i32x4` / `i32x8`) のみ `alice-law-conformance.toml` の G2 を pass にし、G6 の note に Cargo.lock を持たないことによる上流の識別子の変化の見え方を書いた
- `SYNC_SEMANTICS_ID` (`semantics` module、crate root から `SYNC_SEMANTICS_PINS` と `PREVIOUS_SYNC_SEMANTICS_IDS` とともに export): この crate の決定論の状態を指す 32 byte の識別子 新しい `tests/determinism_golden.rs` の 3 つの golden (`Fixed` / `Vec3Fixed` の Q16.16 演算、決まった event 列を当てた後の world の状態と rolling hash、CRDT の merge、最後は `async` のとき) と、bridge 先の `alice-physics` の `PHYSICS_SEMANTICS_ID` と `alice-analytics` の `SEMANTICS_ID` を、名前順・長さ付きで SHA-256 に畳み込む 上流の識別子は hex で持つので feature に依らず同じ値で、feature を有効にした試験が解決した上流の定数と一致することを確かめる 試験 `tests/sync_semantics_id.rs` (畳み込みの再計算、golden の pin は試験の file から読む、5 つの pin のどれを変えても値が動く) 変異 4 件 (pin・上流の識別子・golden・`saturating_mul`) はすべて失敗する `alice-law-conformance.toml` の G6 を pass にした 出力の bit と既存の API は変わらない
- CI (ALICE-Law G3): fma の結果全体を negate する形の検査を追加 (ALICE-LOL と同一の構文木の前段検査 `tools/no-outer-fneg-of-fma` を `--locked` で、最適化後の LLVM IR の検査 `scripts/fneg_fma_ir_check.py` と試験 `scripts/test_fneg_fma_ir_check.py`、47 個の probe `tools/fneg-fma-ir-probes/`) この crate の source に fma の呼び出しは無いので、`scripts/no-outer-fneg-of-fma-allowlist.toml` に `expected-fma-calls = 0` / `expected-src-fma-calls = 0` と IR の下限 0 を宣言し、最初の呼び出しは件数で、negate する形は形で失敗する 構文木の検査は test job と preflight、IR の検査は新しい `.github/workflows/g3-ir.yml` が x86_64 (ubuntu-latest) と aarch64 (ubuntu-24.04-arm) のそれぞれ実機 runner で (`--all-features` は pyo3 を含み、pyo3-ffi の build script は cross build を拒否する) source・manifest・toolchain・検査自身の変更時と毎週実行 実行可否は on.push の paths filter でなく `scripts/changed_paths.py` (ALICE-DetMath と同一) と `.github/paths/g3-ir.txt` で merge-base(origin/main) から決め、走るべき job が走らなかった時は gate job が red になる `alice-law-conformance.toml` の G3 を unverified から pass にした
- CI (ALICE-Law G14): 浮動小数点の結果を変える build の flag を静的と動的の 2 段で検査する (ALICE-DB と同一の `scripts/build_config_guard.py` / `scripts/build_flags_check.py` と test) 静的は target/ 以外の設定・script・Rust を comment を除いて読み、`target-cpu=native`、素の literal でない target-cpu / target-feature / llvm-args の値、command や file から取った RUSTFLAGS、fast-math 系の flag、sse / sse2 を外す target-feature を検出する (意図的な出現は `scripts/build_config_allowlist.toml` に理由付きで書く) 動的は新しい target dir で `cargo check -v` を走らせ、cargo が rustc に実際に渡した flag を読む 静的は entropy-guard job、動的は release-profile job (ubuntu / macOS、default / all-features / no-default) と preflight で実行
- `tests/float_environment.rs` (ALICE-DB と同一): round-to-nearest-even と subnormal が flush されないことを実行時に確かめる test test job (ubuntu / macOS / Windows) と release profile で `scripts/run_tests.sh` を通して実行 `alice-law-conformance.toml` の G14 を pass にした
- CI (`clippy.toml`、G10): `std::time::Instant::elapsed` / `std::time::SystemTime::elapsed` を disallowed-methods に追加 どちらも `::now()` と同じく wall clock を読むが、どちらの層も検査していなかった (`.elapsed(` の字面検査は `elapsed` という名の値全般を拾い過ぎるため、型解決する clippy 側だけに持たせた) `duration_since` はすでに保持した値どうしの引き算で clock を読まないため対象外 非 test code の `.elapsed()` 呼び出しは 0 件で、挙動の変更なし canary: clippy.toml にこの 2 entry があることを確認する test
- `tests/float_edge_golden.rs` と release profile の CI (`release-profile.yml`、ubuntu / macOS) を追加 float の境界 (`Fixed::from_f32` / `to_f32` / `Vec3Fixed`、`BandwidthStats` と `loss_rate` の比、`average_rtt`、`SpatialRegion`) の符号付きゼロ・1 step の前後・大きな逆符号値の打ち消し・0 除算の既定値を `to_bits` / raw 値で固定し、`scripts/run_tests.sh` (ALICE-DB と byte 一致、指定した test file が 1 本も走らなければ失敗) 経由で `analytic_oracle` と合わせて `cargo test --release` で走らせる preflight の full にも同じ step を追加 挙動の変更なし
- `README_JP.md` を追加 (README.md と同じ節構成の日本語版、コード例は英語版と同一) README.md の冒頭に日本語版への link を追加
- README の benchmark 節の測定環境の表記を機種名から `arm64` に変更 (数値は変更なし)
- `clippy.toml` に ALICE-Law の G1 の禁止一覧 (f32 / f64 の超越関数 24 個と `sin_cos`、`powi`) と G10 の `Instant::now` / `SystemTime::now` / `RandomState::new` を `disallowed-methods` として追加 transport 層の単調時計の読み取りは `src/clock.rs` の 1 箇所に集め、その 1 箇所だけに理由つきの許可を置く `scripts/entropy_guard.py` (CI の `entropy-guard` job) が src/ の乱数・時計の使用を読み、`clock.rs` の 1 件以外を拒否する 挙動の変更なし
- CI: libm を直接依存に足した時に G1 の禁止 (clippy.toml の `libm::` path と `scripts/libm_ban_check.py`) が無ければ失敗する検査 `scripts/libm_dependency_guard.py` (試験 `scripts/test_libm_dependency_guard.py`) を entropy-guard job と preflight に追加 (ALICE-DetMath と byte 一致)

### Changed
- entity id の索引: id が `4 * MAX_LIVE_ENTITIES` より小さければ上限までの密な表、大きければ表ごとに key を引いた SipHash の表 使う memory は生存する entity の数に比例する 順序のある出力は id の昇順の写しを通す
- `scripts/api_boundary.toml`: features に `serde-raw` を足し、その `Serialize` / `Deserialize` 10 件を足した
- Licence: `AGPL-3.0-or-later OR LicenseRef-Commercial` → `LicenseRef-Proprietary`
- CI の msrv job: `cargo +1.88 --version` が 1.88 であることを確かめてから `cargo +1.88 check` を実行する (`rust-toolchain.toml` の toolchain で check していた) feature `secure-transport` の ALICE-Channel を Noise IK の rev に、ALICE-Identity を DeviceCert version 2 の rev に上げる
- CI: `scripts/run_tests.sh` と試験 `scripts/test_run_tests.py` を ALICE-LLM と同じ bytes にし、`.github/shared-files-pin` を ALICE-Law の新しい版に移す 名前 `lib` を library の単体試験として数えるようになったが、この repository は `lib` を名指さないので各段の試験の名前と件数は変わらない 試験 `scripts/test_public_text.py` を ALICE-SDF と同じ版にした (公開する text の検査を pin の checker で走らせ、置いた note link を tree と commit message で検出すること、pin の 3 行目 `config-sha256` が設定を承認することを確かめる、network を使う) `scripts/public_text.sh` と public-text の pin は変えないので検査の結果は変わらない
- **破壊的変更**: `Node::apply_event` は他の node から受け取った event の適用で、刻まれた key のまま当て、刻み直さない (これまでは sequence 番号 0 の event を自分の log に積み直していたが、stream の最初の event も番号 0 なので、peer の最初の event が node ごとに別の key になった) 自分が出す event は `Node::emit` を使う `Event::new` のままの event (刻印が無い) は `SyncError::Unstamped` で拒否する `World::iter_entities` は格納順でなく実体 id の順に返す (格納順は Spawn と Despawn の履歴に依る) README と crate の doc の例は emit した log を受け手に渡す形にした `EventStream` の sequence 番号は 1 から振る (0 は stream が刻んでいない event、`Event::is_stamped`) `Node::apply_event` は刻印の無い event を重複とみなさず `SyncError::Unstamped` で拒否する C-ABI: `as_sync_node_apply_event` は刻印の無い event に -2 を返し `as_sync_last_error` に理由を置く (それまでは 2 つ目以降を重複として黙って捨て 0 を返した) `as_sync_node_event_at` (node が emit した event の複製、受け手の apply に渡す) と `as_sync_event_origin` / `as_sync_event_seq` / `as_sync_event_timestamp` を追加 (79 関数、Unity と UE5 の宣言は 76) README に同期の手順を書いた (node の id は node ごとに違うこと: 同じ id の 2 つの node は同じ key で別の event を刻み、どの受け手でも中身の順で同じ world に収束するが、その中身は意図したものにならない) `WorldStorage` の field は crate の外から書けない (`pub(crate)`) crate の外からは capacity / ids / kinds / count / seed などの field も読めなくなった (accessor は無く、`len` / `get_position` / `iter_alive` などの method は残る)、`WorldSoA::storage` は読み取り専用の `WorldSoA::storage()` になった (hash と id の対応を保たない書き込みを防ぐ) Python の `WorldSoA.spawn` / `despawn` は新しい `WorldSoA::spawn_raw` / `despawn_slot` を通り、hash と id の対応を保つ (それまでは storage を直接変え、hash が古いままになった)
- CI: `.github/workflows/eol.yml` に top-level の concurrency group を足し (同じ branch の古い run は止め、main の run は commit ごとに残す)、action を commit SHA で固定して checkout に `persist-credentials: false` を付け、`workflow_dispatch` で手動でも走らせられるようにした (ALICE の crate で同じ中身の file、workflow_concurrency.py と workflow_pin_check.py を持つ crate の検査に合わせた)
- CI: `.gitattributes` に `* text=auto eol=lf` を置き、text の file を保存も checkout も全 OS で LF にした (共有 file は ALICE-Law の表と byte で比べるので、Windows の CRLF の checkout ではずれになる) binary と既存の行はそのまま、`git add --renormalize` で commit 済みの中身は変わらない
- CI: ALICE-SDF と共有する `scripts/api_boundary.py` / `scripts/test_api_boundary.py` を新しい版に揃えた (--document-private-items の 2 回目の rustdoc で private module の extern "C" も列挙、pyo3 の入口・wasm32 だけの fn の列挙、`pending:` の判定、`--skip-missing-targets`、1 つの OS でだけ compile される入口の `host` 表) この crate の一覧は変わらず、全件の判定はそのまま通る
- CI (ALICE-Law G3): `tools/no-outer-fneg-of-fma` と `scripts/fneg_fma_ir_check.py` (と試験) を ALICE-LOL と同じ新しい版にした `expected-src-fma-calls` は src/ の library code の fma 呼び出し数 (`#[test]` と test の時だけ成り立つ cfg の下の item と文、test-only の `mod y;` で読み込む file とその子は数えず、test-only の数を別に表示する) この repository の数は変わらない IR 検査は library の source に fma が無いのに IR に fma がある時を失敗にし、library の fma が 0 の時は共有の probe を target ごとに compile して判定を確かめる (0 件実行は失敗) tool は独自の workspace なので repository 全体の format 検査に入っていなかった CI と preflight に tool の format の検査を足した
- **Breaking:** `Discovery::set_metadata` は `Result<(), MetadataError>` を返す TXT の key でないもの (空、または表示可能な US-ASCII 0x20 から 0x7E 以外の文字か `=` を含む、RFC 6763 6.4) は `MetadataError::InvalidKey`、既に入っている key と大文字小文字だけが違う key は `MetadataError::CaseOnlyDuplicate` で拒否し、何も格納しない (同じ key は値を置き換える) `MetadataError` は crate root からも使える 返り値を捨てていた呼び出しは `unused_must_use` の警告になる (`-D warnings` では compile error)
- **Breaking:** `Discovery::register_peer` の metadata 引数を `HashMap<String, String>` から `impl IntoIterator<Item = (String, String)>` に変えた `HashMap` の値を渡す呼び出しはそのまま compile できるが、引数の型を推論に任せる呼び出し (`….collect()` / `Default::default()`) は型の注釈が要る
- **Behavior change:** 順序の決まらない容器を順序つきに (ALICE-Law G13): `PubSub` の topic の表を `HashMap` から `BTreeMap` にし、`PubSub::topics()` は topic 名の順で返す (以前は hash の順で、同じ topic を入れた instance ごとに順が違った 返り値の型 `Vec<Topic>` は同じ) mDNS で広告する TXT record の property は自分の metadata の key の順に並ぶ (以前は hash の順、受け手から見える wire 上の変化) TXT の key は大文字小文字を区別しない (RFC 6763 6.4) ので、大文字小文字だけが違う key は `set_metadata` が拒否する (以前は TXT record に両方を出していた) 再送の断片表 (`reliability`) と TCP の peer 表 (`transport`) を `rustc_hash::FxHashMap` から `BTreeMap` にし、依存 `rustc-hash` を外した
- CI (ALICE-Law G13): `clippy.toml` に `std::collections::HashMap` / `HashSet` とその iterator 型、`std::thread` の spawn / scope / Builder / JoinHandle / `available_parallelism` の禁止を追加 (ALICE-DB と同じ entry)、`scripts/order_guard.py` と試験 (ALICE-Analytics / ALICE-Edge / ALICE-DB と同じ file) を entropy-guard job と preflight に追加 src/ と Cargo.toml の依存に順序の決まらない容器・thread・rayon / crossbeam があれば、`scripts/order_allowlist.toml` に行の hash と理由付きで書いた箇所以外を拒否する (今は 0 箇所、読む file が 0 なら失敗) clippy の canary の生成器を generic な関数に turbofish を付けて植える版 (ALICE-DB と同じ file) に更新し、表を再生成した (78 行)
- CI: `scripts/api_boundary.py` が cdylib / staticlib を持つ crate の lib target も見つけるようにした (ALICE-Edge と同じ中身に揃えた、この crate の結果は変わらない) `alice-law-conformance.toml` の G2 に、同値時の試験の対象になる箇所が無いこと (浮動小数点の max / min が呼び出しにも比較の形にも 0 件) を書いた
- 版を 0.7.0 に上げた (G7 と G8 の破壊的変更をまとめる、未公開) C-ABI の `as_sync_version` は Cargo.toml の版から計算する (以前は 600 の固定値で 0.6.1 と食い違っていた)
- **破壊的変更**: `Message::Hello` に `semantics_id` を追加 (wire 形式が変わる)、`SyncSession::recv` と受信 loop は Hello の照合前の message を拒否する、`PhysicsRollbackSession::new` は相手の物理の識別子を受けて `Result` を返す、C-ABI の `as_sync_lockstep_new` / `as_sync_rollback_new` / `as_sync_node_add_peer` に相手の識別子の引数を追加 (`as_sync_node_add_peer` は `int32_t` を返す)、Python の `LockstepSession` / `RollbackSession` に `peer_semantics_id` を追加 移行: 自分の識別子 (`SYNC_SEMANTICS_ID` / `as_sync_semantics_id` / `alice_sync.semantics_id()`) を相手に送り、受け取った相手の識別子を各 constructor に渡す README の C-ABI の関数の数を実数 (75、C# と C++ の宣言は 72) に直した
- **破壊的変更**: `Fixed` / `Vec3Fixed` / `Entity` / `WorldState` / `WorldStorage` の `Serialize` / `Deserialize` を opt-in feature `serde-raw` の後ろに移した これらの bytes は識別子を持たず保存形式にならないため 保存には `persist::encode_state` を使う serde で保存した既存の bytes を読むには `serde-raw` を有効にする `LwwRegister` / `LwwMap` / `Arena` の serde derive は要素型が serializable な時だけ働くので、これらの型を入れた container も `serde-raw` なしでは serialize できない
- CI: 公開 API の互換性の検査を `.github/workflows/semver-checks.yml` (`scripts/semver_gate.py`、ALICE-Law / ALICE-DB / ALICE-Analytics / ALICE-Edge / ALICE-Zip と同一) に置き換えた crates.io の最新の alice-sync を baseline にし、既定・既定なし・全 feature の 3 組で cargo semver-checks を走らせ、互換でない変更を失敗させる 版が公開済みの 0.6.1 と同じ間は次の最小の版 0.6.2 として検査する 旧 job (security-audit.yml) は結果を問わない informational で、`rust-toolchain.toml` の 1.92.0 が cargo-semver-checks 0.50 の要求 (rustc 1.93 以上) に届かず毎回何も比較していなかった
- `rust-toolchain.toml` の pin を 1.92.0 から 1.98.1 に上げた (他の ALICE-* と同じ) `rust-version` (MSRV 1.88) は変えない
- CI: 公開する文面の検査の pin を ALICE-Law 5de698e (checker sha256 4b3d9542…) に進めた 新しい版で HEAD の全 file に指摘 0 件
- **`alice-physics` の要求を `2.0` から `3.0` に上げた (2026-10-15)** 公開済の 3.0.1 に追従するため caret は major を跨がず、`^2.0` のままでは 3.x に到達しない `physics_bridge` の変更は無し (`new` + `with_*` / `Default` 構築のため) 実測: `cargo test --features physics` が 161 + 4 + doc 2 pass (lock の解決は 3.0.1)、`cargo check --all-features --all-targets` が 0 error `alice-physics` 3.0.1 は `alice-det-math` を `0.5` で引くので、この crate の det-math 世代は 0.3 から 0.5 に動く
- CI: 上限の無かった 15 job に `timeout-minutes` を付けた (止まった step は共有 runner を GitHub の上限の 6 時間まで占有しうる) 値は直近の成功した run 10 本の所要時間の最大の 2 倍・中央値の 4 倍の大きい方を 5 分単位に切り上げ、cargo を使う job は最低 15 分 (cache が外れた build のため)、他は最低 10 分 fuzz は 30 分で、`workflow_dispatch` の `duration_seconds` は 1 から 1200 の整数 (先頭の 0 は不可、0 は libFuzzer で無制限になる) でなければ理由を出して止める (それより長い run は build と合わせて上限を超え途中で切れる) 入力は script に埋め込まず `env` で渡す `scripts/workflow_timeouts.py` (試験 `scripts/test_workflow_timeouts.py`、ALICE-LOL と byte 一致) が全 job の上限を検査し、job を 1 つも読めなければ失敗する CI (actionlint job) と preflight で走る
- CI: どの workflow も `ci/**` の branch への push でも走るようにした (`push.branches` に `ci/**` を足した) main に入れる前の commit をその branch で検証するため これまでは main への push と pull request でしか走らず、branch の commit には CI の結果が無かった どの workflow の concurrency も、main では commit ごとに別の group にして run を止めない形にした (これまでは main の run も次の commit で止まり、止まった run は検証されないまま残っていた) main 以外では従来どおり古い run を止める path の条件がある workflow (fuzz / security-audit) は従来どおり対象の path が変わった時だけ走る
- **`alice-physics` の要求を `1.4` から `2.0` に上げた (2026-10-09)** 2.0.0 は `#[non_exhaustive]` の一括付与と enum の variant 追加を含む major bump で、caret は major を跨がないため `^1.4` のままでは公開済の 2.0.0 に永久に到達しない `physics_bridge` は 2026-09-30 に `FrameInput` / `NetcodeConfig` の struct literal を `new` + `with_*` / `Default` へ変えてあるので、呼び出し側の変更は無し 実測: `cargo check --features physics --all-targets` が 0 error (lock の解決は 2.0.0)、`scripts/preflight.sh` (full) が exit 0 `alice-physics` は `alice-det-math` を `^0.3` で引くので、この crate の det-math 世代は 0.2 から 0.3 に動く (0.4 に揃うには alice-physics の 0.4 版の publish が必要)
- **`physics_bridge` の `FrameInput` / `NetcodeConfig` 構築を struct literal から `new` + `with_*` / `Default` に変更 (2026-09-30)** alice-physics 側でこの 2 型に `#[non_exhaustive]` が付いても壊れない形にしておく 挙動は変更なし (`sync_input_to_physics` は同じ 4 field を同じ値で埋める、test 側の `player_count: 2` / `max_snapshots: 10` は既定値と同値なので `assert_eq!` で固定)
- Licence: `AGPL-3.0-or-later` → `AGPL-3.0-or-later OR LicenseRef-Commercial`
- CI: `scripts/semver_gate.py` が cargo-semver-checks の出力を読む parser であることを `CARGO_OUTPUT_PARSER = True` で宣言し、`scripts/test_semver_gate.py` は module を `import semver_gate` で読むようにした (色の付いた出力を試験する検査器が parser とその試験を見つけられる形 実行時は従来どおり `CARGO_TERM_COLOR=never` と escape の除去) 共有 file の中身を揃える変更で、検査の挙動は変わらない
- CI: clippy の禁止一覧の canary (`scripts/clippy_canary.py`) を追加し、植える呼び出しの表 `scripts/clippy_canary_calls.tsv` を `scripts/gen_clippy_canary_calls.py` が `clippy.toml` から 1 entry 1 行で生成して CI が再生成した結果と byte で比べる (この repo は 58 件) `scripts/clippy_canary.py` / `scripts/test_clippy_canary.py` / `scripts/gen_clippy_canary_calls.py` / `scripts/test_gen_clippy_canary_calls.py` は ALICE-DB と byte 単位で同一 clippy job と preflight に 4 step を足した
- CI: `scripts/run_tests.sh` が名前ごとの最低の件数 (`float_environment=2`) を受け付けるようにし (ALICE-DB と同じ中身)、G14 の `tests/float_environment.rs` を名指す全 step を `float_environment=2` にした 2 つの試験の片方を `#[ignore]` にする・cfg で外す・名前を変えると、その step が失敗する `scripts/test_run_tests.py` も足し CI と preflight で走らせる
- CI: 公開する文面の検査の pin を ALICE-Law 779fa44 (checker sha256 e439fdd7…) に進めた 新しい版で HEAD の全 file に指摘 0 件

## [0.6.1] - 2026-09-17

### Fixed
- `codec_bridge`: `decompress_event_batch` が量子化 step を 1 に固定していたため step > 1 の batch が step 分の 1 に縮んで復号されていた → step を header (4 B、histogram の前) に載せて復号側で使う (header 1028 → 1032 B、0.6.0 の `data` とは非互換) / `to_symbols` の `|q| > 127` error を捨てて symbol が silent wrap していた → step を `⌈max|c| / 127⌉` まで広げる (8-bit 全域 data で ≤ 4)、`Result` は `expect` で fail fast
- `codec_bridge` fuzz 初回 (2 件): 破損 header の step (u32 最大) で alice-codec `dequantize` が乗算 overflow → 復号側 step を `MAX_STEP = 2¹⁶` に clamp / header 内 length と `original_len` の不一致で 4 GiB 割当 → 不一致は復号せず返す
- `alice-codec` 要求を 0.1.3 に (rANS の偏った histogram round-trip / 量子化 symbol wrap の修正版)、`codec` feature の oracle の `#[ignore]` を解除

### Added
- CI を canonical 構成に (ALICE-SDF 同等): `ci.yml` = fmt / clippy `-D warnings` 3 variant / test 3 OS (default + 全 native feature) / analytic oracle (async + physics + codec) / msrv `rust-version = 1.88` 実 compile / doc `-D warnings` / feature-powerset (cargo-hack depth 2、python 除外)、`security-audit.yml` (audit / deny / unused-deps / stub-guard / coverage / semver-checks)、`fuzz.yml` + `fuzz/` 3 target (packet decode + 再組立 / codec_bridge / CRDT merge の可換・冪等)、`deny.toml` (AGPL bridge 例外) `ci-unified.yml` (workflow_dispatch 専用、stub 生成) は削除、未使用の `crossbeam-channel` dep を撤去 (cargo-machete)、`bytes` 1.12.1 (RUSTSEC-2026-0007)
- `tests/analytic_oracle.rs` — 閉形式 oracle 8 本 + ignore 1 (2026-09-17): Q16.16 の dyadic 有理数 exact / 飽和 / i16 Q8.8 往復、SIMD add・sub が scalar と bit 一致 + batch、RFC 6298 の SRTT / RTTVAR / RTO を整数演算で逐語再現 + 定常 / clamp、header 往復 + 断片数 ⌈len/(MTU−4)⌉ + 逆順再組立、sequence / 重複、CRDT 代数 (可換 / 冪等 / LWW)、FNV-1a test vector + 最初の divergence frame、`physics` feature: i16 ↔ Fix128 の Q8.8 契約 (両方向・飽和) `codec` feature の oracle は alice-codec 0.1.3 publish 待ちで `#[ignore]` CI に async + physics の oracle step
- **i16 wire field (`InputFrame::movement` / `aim`、`EventKind::Motion` delta) の scale が 3 経路で食い違っていた** (oracle 先行 red): doc は Q8.8、`Fixed::from_i16` は `<< 6` (Q6.10、256 ↦ 0.25)、`physics_bridge` は整数 (`from_int`、256 ↦ 256.0) — 同じ入力が経路毎に 1024 倍違う変位になっていた → Q8.8 (256 ↦ 1.0) を唯一の法則に: `Fixed::from_i16` = `<< 8` / `to_i16` = `>> 8` + 飽和、`Vec3Simd::from_i16_array` は `Fixed::from_i16` 経由、`physics_bridge` は `Fix128::from_int(n).shr_bits(8)` と `⌊v·256⌋` 飽和 既存 unit test の `<< 6` / `.hi == n` pin を法則値に更新

### Changed
- README / lib.rs の全称 claim を実態に限定し、各 claim 行に `<!-- claim-test: fn -->` で検証 test を紐付け (strict-eval 検査 1、2026-09-17)
- bridge 系 sibling 6 crate (`alice-physics` 1.4 / `alice-db` 0.2.0-beta.2 / `alice-cache` 0.2 / `alice-auth` 0.5.1 (0.5.0 は release build 不能) / `alice-codec` 0.1 / `alice-analytics` 0.1) の依存を path から crates.io version に変更、CI の manifest-only stub (version 0.1.0 固定で `^1` / `^0.5` を満たせず 2026-09-14 から red) を撤去し `cargo check --lib --all-features` を追加 (bridge feature が公開版 API で compile することを CI が初めて確認)

### Fixed
- **FFI 71 関数の panic 隔離** (`src/ffi.rs`): 全 `extern "C"` の本体を `ffi_guard(sentinel, || ..)` で包み、panic は host を落とさず sentinel (null / 0 / −1 / NaN / ()) + `as_sync_last_error()` (新規、`as_sync_clear_last_error` / `as_sync_free_error_string` も) で通知 `[profile.release] panic = "abort"` を撤去 (abort では `catch_unwind` が機能しない、cdylib size が数 % 増) release profile で guard test 通過
- `analytics_bridge`: `alice_analytics::prelude` は存在しない module だった (bridge 追加以来 stub で一度も compile されていなかった) → `alice_analytics::sketch::{CountMinSketch1024x5, DDSketch256, HyperLogLog12}`
- `codec_bridge` test: 常に真の `u8 <= 255` assert (clippy `absurd_extreme_comparisons`) を長さ検証に置換、誤差上限 oracle `roundtrip_error_is_bounded` を追加 — 実測 MAE 24.6 / max 101 (0..63 鋸波) で定数予測より悪いため `#[ignore]` (bridge の量子化は要修正、未解決)

## [0.6.0] - 2026-02-23

### Added

- Input synchronization: `LockstepSession`, `RollbackSession`, `InputFrame`, `InputBuffer`
- Rollback netcode with prediction, confirmation, and automatic re-simulation
- `PhysicsRollbackSession` (feature `physics`): combined rollback + deterministic physics
- `codec_bridge` (feature `codec`): wavelet + rANS compression for event streams
- `analytics_bridge` (feature `analytics`): DDSketch/HLL/CMS probabilistic telemetry
- `auth_bridge` (feature `auth`): Ed25519 ZKP peer authentication via ALICE-Auth
- `cache_bridge` (feature `cache`): Markov oracle entity prefetching via ALICE-Cache
- `cloud_bridge` (feature `cloud`): star-topology multi-device spatial sync hub
- Python bindings: `WorldSoA`, `InputFrame`, `LockstepSession`, `RollbackSession`
- C-ABI FFI bindings: 71 `extern "C"` functions (feature `ffi`)
- Unity C# bindings: 71 `[DllImport]` wrappers + RAII handles (`bindings/unity/AliceSync.cs`)
- UE5 C++ bindings: 71 `extern "C"` declarations + 9 RAII handles (`bindings/ue5/AliceSync.h`)
- "Demon Mode" batch optimizations: sort by entity_id, coalesce same-entity updates
- Auto-SIMD detection for contiguous 8-slot ranges
- Comprehensive module documentation in `lib.rs`
- `CHANGELOG.md` and `CONTRIBUTING.md`
- 88 tests across core + bridge + FFI modules

### Changed

- Event batching re-sorts by slot index after entity-id sort for true memory locality
- `apply_motions_sorted` now 1.4x faster via prefetcher-friendly access patterns

### Fixed

- `codec_bridge` test used 256-byte input below 1028-byte header threshold
- Unused import warnings in `cloud_bridge` and unused `Result` warnings in `codec_bridge`
- Doc warnings from `dx:2`/`dz:2` parsed as intra-doc links

## [0.5.0] - 2026-02-20

### Added

- `WorldSoA`: complete Structure of Arrays world storage
- Vertical SIMD: process 8 entities' x/y/z coordinates in one instruction
- 8-wide parallel hash computation
- `Slot` abstraction for SoA indexing
- Python batch API: `apply_motions()`, `positions()` with GIL release

### Changed

- Motion reads 12 bytes/entity instead of 84 bytes (7x memory reduction)
- 100% cache efficiency: only touch data needed for each operation

## [0.4.0] - 2026-02-18

### Added

- SoA (Structure of Arrays) event storage: `MotionData`, `SpawnData`, `DespawnData`
- `SoAStats` for batch processing statistics
- Branchless property hash with arithmetic masking

### Changed

- Raw integer mixing replaces `Hasher` trait overhead
- Event enum tag eliminated in hot loops via SoA decomposition

## [0.3.0] - 2026-02-16

### Added

- `WorldHash`: XOR rolling hash (Zobrist-style) for O(1) incremental verification
- Vec direct indexing for O(1) entity lookup (no HashMap)
- `Entity` as `Copy` type (no heap allocation)
- SIMD `Vec3Simd` operations via `wide` crate
- `Vec3Fixed`: Q16.16 fixed-point 3D vector

### Changed

- World hash comparison reduced from O(N) full recalc to O(1) incremental

## [0.2.0] - 2026-02-14

### Added

- `Fixed`: Q16.16 fixed-point integer type for platform-independent determinism
- `Arena`: generational arena allocator with O(1) insert/remove/lookup
- bitcode serialization (61% smaller than bincode)
- `Protocol`: Handshake / Sync / Ack / HashCheck wire messages
- `EventStream`: ordered event history with sequence numbers

## [0.1.0] - 2026-02-12

### Added

- Core event system: `Event`, `EventKind`, `EventId`, `SeqNum`
- `Node`: P2P node with causal event ordering and world state
- `World`: AoS entity world with spawn/motion/despawn/property events
- `Entity`, `EntityProps`: fixed-size entity with 8 property slots
- bincode serialization
- 5 event types: Spawn, Motion, Despawn, Property, Tick

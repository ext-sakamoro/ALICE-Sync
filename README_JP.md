# ALICE-Sync

[English](README.md)

**汎用 P2P 同期基盤**

> 「データを送らない 差分を送る」

## ALICE-Sync とは

ALICE-Sync は高性能な P2P 状態同期エンジン ワールド全体の状態を送る代わりに、ノード間で**イベント** (spawn、move、despawn) だけを同期する `async` feature を有効にすると、トランスポート、ディスカバリ、pub/sub、CRDT、セッション管理を含む、それ自体で完結した P2P 基盤になる

### 主な特長

- **最小の帯域**: メガバイト級のワールドデータの代わりに 18 バイトのイベントを送る
- **ビット単位で一致する決定論**: 固定小数点演算により、すべてのプラットフォームで同一の結果になる
  <!-- claim-test: test_fixed_determinism -->
- **O(1) の同期検証**: 10,000 以上のエンティティでもハッシュ比較は即座に済む
- **ゼロアロケーション**: アリーナストレージ上の Copy 型エンティティ
- **自己完結した P2P**: UDP/TCP トランスポート、mDNS ディスカバリ、topic pub/sub、CRDT — 外部の基盤は不要
- **複数の同期モード**: Lockstep、Rollback、CRDT、Event Sourcing、Snapshot

## インストール

```toml
[dependencies]
alice-sync = "0.6"

# Full P2P infrastructure
alice-sync = { version = "0.6", features = ["async"] }
```

## クイックスタート

```rust
use alice_sync::{Node, NodeId, Event, EventKind};

fn main() {
    // Create two nodes with the same seed
    let mut node_a = Node::new(NodeId(1));
    let mut node_b = Node::new(NodeId(2));

    // Node A: Spawn an entity
    let spawn = Event::new(EventKind::Spawn {
        entity: 1,
        kind: 0,
        pos: [0, 0, 0],  // i16 fixed-point coordinates
    });
    node_a.apply_event(&spawn).unwrap();

    // Node A: Move the entity
    let motion = Event::new(EventKind::Motion {
        entity: 1,
        delta: [100, 0, 0],  // Move right
    });
    node_a.apply_event(&motion).unwrap();

    // Sync: Send events to Node B (not world data!)
    node_b.apply_event(&spawn).unwrap();
    node_b.apply_event(&motion).unwrap();

    // Verify: Both nodes have identical world state
    assert_eq!(node_a.world_hash(), node_b.world_hash());
    println!("Sync verified! Hash: {:016x}", node_a.world_hash().0);
}
```

## Async P2P 基盤

`--features async` で、自己完結した P2P スタック一式が有効になる

### モジュール

| モジュール | 説明 |
|--------|-------------|
| `reliability` | UDP の信頼性レイヤ: ACK 追跡、RTT 推定、再送、分割と再構築 |
| `transport` | `SyncTransport` trait と UDP、TCP、プロセス内 Channel の実装 |
| `crdt` | 競合のない複製データ型: `LwwRegister`、`GCounter`、`PnCounter`、`OrSet`、`LwwMap` |
| `channel` | topic 単位のフィルタを持つ topic ベースの pub/sub |
| `discovery` | mDNS による LAN 自動検出、手動のピア登録、GC |
| `session` | 高レベルのセッションドライバ: builder パターン、tick ループ、5 つの同期モード |

### 同期モード

| モード | 向く用途 | 特徴 |
|------|----------|-----------------|
| **Lockstep** | RTS、ターン制、4 人未満 | すべての入力が揃うまで進行を待つ |
| **Rollback** | 格闘ゲーム、FPS、アクション | 予測し、不一致があれば補正する |
| **CRDT** | 共同編集、IoT、データベース | 結果整合性、調停は不要 |
| **EventSourcing** | 監査ログ、金融、法務 | 追記のみのイベントログとリプレイ |
| **Snapshot** | 途中参加、復旧 | 状態全体の転送 |

### セッションの例

```rust
use alice_sync::{SessionBuilder, SyncMode, UdpTransport};

#[tokio::main]
async fn main() {
    let transport = UdpTransport::bind("0.0.0.0:9000").await.unwrap();

    let session = SessionBuilder::new()
        .mode(SyncMode::Crdt)
        .node_id(1)
        .tick_rate(60)
        .name("my-node")
        .build(transport);

    // Add peers manually or let mDNS discover them
    session.add_peer("203.0.113.10:9000".parse().unwrap()).await;

    // Subscribe to session events
    let mut events = session.subscribe_events();

    // Start the tick loop
    session.start().await;

    // Use CRDT for conflict-free state sync
    session.crdt_set("player/1/name".into(), b"Alice".to_vec()).await;

    // Use topic pub/sub for structured routing
    let mut rx = session.pubsub().subscribe("game/room42/state", addr).await;
    session.pubsub().publish("game/room42/state", payload, None).await;
}
```

### CRDT の例

```rust
use alice_sync::{LwwMap, OrSet, GCounter, CrdtMergeable};

// LWW-Map: distributed key-value store
let mut node_a = LwwMap::new(1); // replica ID 1
let mut node_b = LwwMap::new(2); // replica ID 2

node_a.insert("key".to_string(), "value_a".to_string());
node_b.insert("key".to_string(), "value_b".to_string());

// Merge — highest timestamp wins, no conflicts
node_a.merge(&node_b);

// OR-Set: add/remove with add-wins semantics
let mut set = OrSet::new();
set.add(1, "tag_a".to_string());
set.add(2, "tag_b".to_string());
assert!(set.contains(&"tag_a".to_string()));

// G-Counter: distributed monotonic counter
let mut counter = GCounter::new();
counter.increment(1); // replica 1 increments
counter.increment(2); // replica 2 increments
assert_eq!(counter.value(), 2);
```

### トランスポートの抽象化

```rust
use alice_sync::{SyncTransport, UdpTransport, TcpTransport, ChannelTransport};

// UDP: low-latency, with reliability layer (ACK, retransmit, fragmentation)
let udp = UdpTransport::bind("0.0.0.0:9000").await?;

// TCP: reliable ordered delivery, length-prefixed framing
let tcp = TcpTransport::bind("0.0.0.0:9001").await?;

// Channel: in-process, for testing
let (a, b) = ChannelTransport::pair();
```

### Topic Pub/Sub

```rust
use alice_sync::PubSub;

let pubsub = PubSub::new();

// Subscribe peers to topics
let mut rx = pubsub.subscribe("sensor/temp/room1", peer_addr).await;

// Publish with optional filtering
pubsub.publish("sensor/temp/room1", data, Some(origin)).await;

// Get all subscribers for transport-level routing
let peers = pubsub.subscribers("sensor/temp/room1").await;
```

### ピアディスカバリ

```rust
use alice_sync::Discovery;
use std::sync::Arc;

let discovery = Arc::new(Discovery::new(local_addr, "my-node"));

// Manual peer registration
discovery.register_peer(addr, "peer-1", metadata).await;

// Event callbacks
discovery.on_event(Arc::new(|event| {
    println!("Peer event: {:?}", event);
})).await;

// Start mDNS GC background task
discovery.start_mdns_gc();

// Get live peers
let peers = discovery.peer_addrs().await;
```

## バッチ処理 (高性能)

高スループットが必要な場面では、最適化されたバッチ API を使う

```rust
use alice_sync::{WorldSoA, Event, EventKind, MotionData};

let mut world = WorldSoA::new(42);

// Spawn entities
for i in 0..1000u32 {
    world.apply(&Event::new(EventKind::Spawn {
        entity: i,
        kind: 0,
        pos: [0, 0, 0],
    })).unwrap();
}

// Batch motion updates (optimized: sort + coalesce + SIMD)
let motions: Vec<MotionData> = (0..1000)
    .map(|i| MotionData {
        entity: i as u32,
        delta_x: 1,
        delta_y: 0,
        delta_z: 0,
    })
    .collect();

world.apply_motions_sorted(motions);  // 2-5x faster than naive apply
```

## イベント型

| イベント | 説明 | サイズ (bitcode) |
|-------|-------------|----------------|
| Spawn | 位置を指定してエンティティを作る | 約 20 バイト |
| Motion | エンティティを差分だけ動かす | 約 18 バイト |
| Despawn | エンティティを削除する | 約 10 バイト |
| Property | エンティティのプロパティを設定する | 約 18 バイト |
| Tick | フレームを進める | 約 14 バイト |

## ベンチマーク結果

すべて arm64 で `cargo bench --release` により測定した

```
World Hash (10000 entities):
  Incremental O(1):    543 ps   <- Constant time!
  Full Recalc O(N):    46.5 us  <- Traditional approach
  Speedup: ~85,000x

Entity Lookup:
  Vec[id] indexing:    1.15 ns  <- O(1) regardless of count

Batch Processing (1000 motions, 100 entities):
  Sorted + Coalesce:   17.3 us  <- Demon Mode
  No optimization:     22.3 us  <- Baseline
  Speedup: 1.3x

Coalesce Effect (same entity x1000):
  With coalesce:       6.5 us   <- Merges to 1 update
  Without coalesce:    35 us    <- 1000 separate updates
  Speedup: 5.5x

Serialization:
  bincode:             46 bytes
  bitcode:             18 bytes (61% smaller)

Single Event Apply:
  Motion event:        30 ns
```

## アーキテクチャ

```
                         ALICE-Sync P2P Stack

  Application Layer
  +-------------------------------------------------------------+
  |  SessionBuilder -> SyncSession                               |
  |    - 5 sync modes (Lockstep/Rollback/CRDT/EventSrc/Snapshot)|
  |    - Tick loop, event dispatch, stats                        |
  +-------------------------------------------------------------+
                            |
  +----------------+  +----------+  +-------------+
  | Topic Pub/Sub  |  |  CRDTs   |  |  Discovery  |
  | (channel.rs)   |  | LWW-Map  |  | mDNS + GC   |
  | Filter, Route  |  | OR-Set   |  | Manual reg  |
  +----------------+  | G/PN-Cnt |  +-------------+
                       +----------+
                            |
  Transport Layer
  +-------------------------------------------------------------+
  |  SyncTransport trait                                         |
  |    +----------+  +----------+  +-----------+                 |
  |    |   UDP    |  |   TCP    |  |  Channel  |                 |
  |    | + Reliab |  | Length-  |  | In-process|                 |
  |    | + Frag   |  | prefixed |  | (testing) |                 |
  |    +----------+  +----------+  +-----------+                 |
  +-------------------------------------------------------------+
                            |
  Core Engine
  +-------------------------------------------------------------+
  |  Node (causal ordering) <- Event Stream <- Protocol (bitcode)|
  |  World (AoS, O(1) hash)    WorldSoA (8-wide SIMD)           |
  |  Fixed-point (Q16.16)      Input Sync (Lockstep/Rollback)   |
  +-------------------------------------------------------------+
```

## 最適化の経緯

### v0.6 "Demon Mode" - ソートに基づくバッチ処理と P2P 基盤

- キャッシュに載りやすいアクセスのため、イベントをエンティティ ID でソートする
- 同一エンティティの更新をまとめる (バースト時に 5.5 倍高速)
- メモリの局所性のため、slot で再ソートする
- SIMD のため、連続した範囲を自動検出する
- **新規**: async P2P スタック一式 (トランスポート、ディスカバリ、pub/sub、CRDT、セッション)

### v0.5 "Zen Mode" - SoA への完全移行

- エンティティのデータをコンポーネントごとの配列に分解
- 垂直 SIMD: 1 命令で 8 エンティティを処理
- 移動データはエンティティあたり 12 バイト (従来は 84 バイト、7 分の 1)

### v0.4 "Ultra Mode" - 分岐なしと unsafe

- 整数のまま混ぜる (Hasher trait のオーバーヘッドなし)
- 算術マスクによる、分岐のないプロパティハッシュ
- キャッシュ効率のための SoA イベントストレージ

### v0.3 "God Mode" - すべてを O(1) に

- XOR ローリングハッシュ (Zobrist 方式)
- Vec の直接インデックス (HashMap を使わない)
- エンティティは Copy 型 (ヒープ割り当てなし)
- SIMD による Vec3 演算

### v0.2 - 決定論

- 固定小数点演算 (f32 のずれなし)
- アリーナアロケーション
- bitcode によるシリアライズ

## 主な技法

### O(1) の増分ハッシュ

```rust
// Each event updates hash in O(1) using XOR
pub fn apply_motion(&mut self, entity_id: u32, delta: [i16; 3]) {
    let old_hash = self.compute_entity_hash(entity_id);
    self.current_hash ^= old_hash;  // Remove old state

    self.update_position(entity_id, delta);

    let new_hash = self.compute_entity_hash(entity_id);
    self.current_hash ^= new_hash;  // Add new state
}
```

### 固定小数点による決定論

```rust
// Fixed-point sync state: the simulated world never touches a float, so there is
// no platform-dependent rounding (telemetry / analytics / cloud bridges report
// with f32, the world hash never sees one) <!-- claim-test: test_fixed_determinism -->
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Fixed(pub i32);  // Q16.16 format

// Guaranteed identical results on x86, ARM, WASM
let a = Fixed::from_f32(1.5);
let b = Fixed::from_f32(0.7);
let c = a + b;  // Always exactly the same bits
```

## 入力同期

マルチプレイヤーゲーム向けの決定論的な入力同期 — **プレイヤー 1 人・1 フレームあたり約 20 バイト**

### 使い方

```rust
use alice_sync::{InputFrame, LockstepSession, RollbackSession};

// --- Lockstep (RTS / turn-based) ---
let mut session = LockstepSession::new(2);  // 2 players

let input = InputFrame::new(0, 0)  // frame 0, player 0
    .with_movement(100, 0, 0)
    .with_actions(0x01);

session.add_local_input(input);
session.add_remote_input(remote_input);

if session.ready_to_advance() {
    let inputs = session.advance().unwrap();
    // Apply all inputs deterministically
}

// --- Rollback (fighting / FPS) ---
let mut rollback = RollbackSession::new(2, 0, 8);  // 2 players, local=0, max_rollback=8

let predicted_inputs = rollback.add_local_input(local_input);
// Apply predicted_inputs immediately (no waiting!)

match rollback.add_remote_input(confirmed_input) {
    RollbackAction::None => {},           // Prediction was correct
    RollbackAction::Rollback { to_frame } => {
        // Restore snapshot, re-simulate from to_frame
    },
    RollbackAction::Desync { frame } => {
        // Unrecoverable - reconnect
    },
}
```

### 帯域の比較

| 方式 | フレームあたりのデータ | 60 FPS、4 人 |
|----------|-----------|-------------------|
| 状態同期 (100 ボディ) | 約 16 KB | 3.84 MB/s |
| **入力同期 (ALICE)** | 約 80 B | **19.2 KB/s** |
| 削減率 | | **99.5%** |

## Cargo Feature

| Feature | 既定 | 説明 |
|---------|---------|-------------|
| `std` | あり | 標準ライブラリ対応 |
| `async` | なし | P2P 基盤一式: トランスポート、ディスカバリ、pub/sub、CRDT、セッションドライバ (tokio が必要) |
| `simd` | なし | SIMD 高速化のヒント |
| `python` | なし | Python バインディング (PyO3 0.28 と NumPy のゼロコピー) |
| `physics` | なし | ALICE-Physics ブリッジ (InputFrame <-> FrameInput、PhysicsRollbackSession) |
| `telemetry` | なし | ALICE-DB による同期テレメトリの記録 (RTT、rollback、desync の指標) |
| `cache` | なし | ALICE-Cache による Markov oracle でのエンティティ先読み |
| `auth` | なし | ALICE-Auth による Ed25519 ZKP のピア認証 |
| `codec` | なし | ALICE-Codec による wavelet と rANS のイベントストリーム圧縮 |
| `analytics` | なし | ALICE-Analytics による DDSketch / HLL / CMS の確率的テレメトリ |
| `ffi` | なし | Unity / UE5 向けの C-ABI FFI (71 関数) |
| `cloud` | なし | クラウド側の複数デバイス空間同期ハブ |
| `all-bridges` | なし | すべての連携ブリッジを有効にする (`physics` + `telemetry` + `cache` + `auth` + `codec` + `analytics`) |

## C-ABI FFI (Unity / UE5)

コア型をすべて扱う 71 個の `extern "C"` 関数 `--features ffi` で有効になる

### 対象範囲

| モジュール | 関数数 | 説明 |
|--------|-----------|-------------|
| World (AoS) | 8 | new, free, hash, entity_count, frame, apply_event, get_position, recalculate_hash |
| WorldSoA | 5 | new, free, hash, entity_count, apply_event |
| Node | 11 | new, with_seed, free, hash, state, emit, apply, add_peer, entity_count, events_count, events_bytes |
| Event | 8 | new_motion, new_spawn, new_despawn, new_property, new_input, new_tick, free, size_bytes |
| EventStream | 7 | new, free, push, len, is_empty, current_seq, total_bytes |
| InputFrame | 10 | new, free, set/get movement, set/get actions, set/get aim, get_frame, get_player_id |
| InputFrameArray | 3 | free, len, get |
| Lockstep | 5 | new, free, add_local/remote_input, ready, advance, confirmed_frame, record_checksum |
| Rollback | 7 | new, free, add_local/remote_input, confirmed/predicted_frame, frames_ahead |
| Utilities | 4 | fixed_from_f32, fixed_to_f32, vec3_hash, version |

### Unity C# (`bindings/unity/AliceSync.cs`)

71 個の `[DllImport]` 宣言と、RAII の `IDisposable` ハンドル

### UE5 C++ (`bindings/ue5/AliceSync.h`)

71 個の `extern "C"` 宣言と、9 個の RAII `std::unique_ptr` ハンドル

## Python バインディング (PyO3 と NumPy のゼロコピー)

```toml
[features]
python = ["pyo3", "numpy"]
```

### 最適化レイヤ

| レイヤ | 技法 | 効果 |
|-------|-----------|--------|
| L1 | NumPy のゼロコピー (`into_pyarray`) | 位置データの memcpy なし |
| L2 | バッチ API (SoA ワールド操作) | FFI 呼び出しコストの償却 |
| L3 | Rust バックエンド (8 幅 SIMD、Demon Mode) | ハードウェア速度での適用 |

### Python API

```python
import alice_sync

# --- WorldSoA (high-performance SoA world) ---
world = alice_sync.WorldSoA(seed=42)
world.spawn(entity_id=0, kind=0, x=0, y=0, z=0)
world.spawn(entity_id=1, kind=0, x=100, y=0, z=0)

# Batch motions (Demon Mode SIMD)
motions = [(0, 10, 0, 0), (1, -5, 3, 0)]  # (entity_id, dx, dy, dz)
world.apply_motions(motions)

positions = world.positions()  # NumPy (N, 3) int32 - zero-copy
print(f"Hash: {world.hash():016x}")

# --- Input sync ---
frame = alice_sync.InputFrame(frame=0, player_id=0, move_x=100, move_y=0, move_z=0)
data = frame.to_bytes()   # bitcode serialized (~18 bytes)
decoded = alice_sync.InputFrame.from_bytes(data)

# Lockstep session
session = alice_sync.LockstepSession(player_count=2)
session.add_local_input(frame)

# Rollback session
rollback = alice_sync.RollbackSession(player_count=2, local_player=0, max_rollback=8)
predicted = rollback.add_local_input(frame)
action = rollback.add_remote_input(remote_frame)  # "none" / "rollback:N" / "desync:N"
```

## ゲームエンジンパイプライン (ALICE-Physics ブリッジ)

`--features physics` で、ALICE-Sync の入力同期と [ALICE-Physics](../ALICE-Physics) の決定論的シミュレーションを橋渡しする

```toml
[dependencies]
alice-sync = { path = "../ALICE-Sync", features = ["physics"] }
```

### アーキテクチャ

```
Network --> InputFrame (i16, 24B) --> FrameInput (Fix128) --> PhysicsWorld
                 ALICE-Sync                ALICE-Physics

PhysicsWorld --> SimulationChecksum --> WorldHash --> Desync Verification
                   ALICE-Physics           ALICE-Sync
```

### PhysicsRollbackSession

rollback の入力同期と決定論的な物理を 1 つの struct にまとめたもの

```rust
use alice_sync::physics_bridge::PhysicsRollbackSession;
use alice_sync::InputFrame;
use alice_physics::{NetcodeConfig, RigidBody, Vec3Fix, Fix128};

let mut session = PhysicsRollbackSession::new(2, 0, 8, NetcodeConfig::default());

// Add player bodies to physics world
let body = RigidBody::new_dynamic(Vec3Fix::from_int(0, 10, 0), Fix128::ONE);
let idx = session.sim.add_body(body);
session.sim.assign_player_body(0, idx);

// Game loop: advance with local input
let input = InputFrame::new(1, 0).with_movement(1, 0, 0);
let checksum = session.advance_frame(input);

// When remote input arrives:
match session.add_remote_input(remote_input) {
    RollbackAction::None => {},
    RollbackAction::Rollback { to_frame } => {
        session.handle_rollback(to_frame);  // auto re-simulate
    },
    RollbackAction::Desync { .. } => { /* reconnect */ },
}
```

### ブリッジ関数

| 関数 | 説明 |
|----------|-------------|
| `sync_input_to_physics()` | InputFrame (i16 Q8.8) -> FrameInput (Fix128) |
| `physics_input_to_sync()` | FrameInput (Fix128) -> InputFrame (i16, truncate) |
| `sync_inputs_to_physics()` | 一括変換 |
| `physics_checksum_to_world_hash()` | SimulationChecksum(u64) -> WorldHash(u64) |
| `world_hash_to_physics_checksum()` | WorldHash(u64) -> SimulationChecksum(u64) |

## 同期テレメトリ (ALICE-DB 連携)

ネットワーク同期の指標を、時系列データとして [ALICE-DB](../ALICE-DB) に記録する `--features telemetry` で有効になる

### チャンネル

| チャンネル | 指標 | 範囲 |
|---------|--------|-------|
| 0 | Rollback 回数 | イベントあたりに巻き戻したフレーム数 |
| 1 | Desync の深刻度 | 0.0 = なし、1.0 = 復旧不能 |
| 2 | 予測精度 | 0.0..1.0 |
| 3 | RTT (ms) | 往復時間 |
| 4 | 入力遅延 | フレーム単位の遅延 |

### 使い方

```rust
use alice_sync::telemetry::SyncTelemetry;

let telemetry = SyncTelemetry::new("./telemetry_data")?;

// Record during gameplay
telemetry.record_rtt(frame, 15.5)?;
telemetry.record_prediction_accuracy(frame, 0.95)?;
telemetry.record_rollback(frame, 3)?;

telemetry.flush()?;

// Query for analysis
let rtt_data = telemetry.scan_rtt(0, 3600)?;
let avg_rtt = telemetry.average_rtt(0, 3600)?;
let max_rollback = telemetry.max_rollback(0, 3600)?;

telemetry.close()?;
```

## crate 間ブリッジ

### Cache ブリッジ (feature: `cache`)

[ALICE-Cache](../ALICE-Cache) による、CRDT ベースの分散キャッシュ無効化 同期イベントが共有状態を変更すると、cache ブリッジが無効化メッセージを伝播し、ノード間の分散キャッシュの一貫性を保つ

### Codec ブリッジ (feature: `codec`)

[ALICE-Codec](../ALICE-Codec) による、イベントストリームのバッチ向け wavelet と rANS の圧縮 シリアライズ済みのイベントデータに 1 次元 wavelet 変換とエントロピー符号化を適用し、イベントバッチの転送サイズを減らす

```rust
use alice_sync::codec_bridge::{compress_event_batch, decompress_event_batch, estimate_ratio};

let compressed = compress_event_batch(&event_bytes, quality)?;
let ratio = estimate_ratio(&event_bytes);
let original = decompress_event_batch(&compressed)?;
```

### Analytics ブリッジ (feature: `analytics`)

[ALICE-Analytics](../ALICE-Analytics) によるリアルタイムの同期テレメトリ イベントのスループット、往復レイテンシ、ユニークなピア数、ハッシュの乖離を、確率的 sketch (DDSketch、HyperLogLog、Count-Min Sketch) に流し込む

```rust
use alice_sync::analytics_bridge::SyncTelemetry;

let mut tel = SyncTelemetry::new();
tel.record_throughput(1500.0);
tel.record_latency(42.0);
tel.record_peer(b"peer-001");
println!("p99 latency: {:.1}us", tel.latency_p99());
println!("unique peers: {:.0}", tel.unique_peers());
```

### 受け側のブリッジ: ALICE-Streaming-Protocol

[ALICE-Streaming-Protocol](../ALICE-Streaming-Protocol) は `sync_bridge` module (feature `sync`) を通じて ALICE-Sync に接続し、P2P ノード間でメディアストリームの状態を同期する

## ユースケース

- **マルチプレイヤーゲーム**: RTS、格闘ゲーム、FPS、物理パズル (Lockstep / Rollback)
- **リアルタイム共同作業**: 文書の共同編集、3D モデルの同期、ホワイトボード (CRDT モード)
- **IoT / エッジコンピューティング**: センサーメッシュの同期、デバイス状態の収束 (CRDT と mDNS ディスカバリ)
- **分散データベース**: 変更データキャプチャによるレプリケーション、マルチマスター同期 (Event Sourcing と CRDT)
- **メッセージング基盤**: topic ベースの pub/sub、グループチャットの配信、通知のファンアウト
- **ゲームエンジンパイプライン**: ALICE-Physics と ALICE-Sync によるビット単位で一致する rollback ネットコード
- **分散シミュレーション**: 気象、交通、物理 (決定論的なイベント差分)
- **エッジキャッシュ**: 結果整合性による分散キャッシュの一貫性

## プロトコル

```
Handshake:
  A -> B: Hello { node_id, seq, world_hash }
  B -> A: Hello { node_id, seq, world_hash }

Sync:
  A -> B: Events { [event1, event2, ...] }  // bitcode encoded
  B -> A: Ack { seq }

Consistency Check:
  A -> B: HashCheck { seq, world_hash }
  B -> A: HashCheck { seq, world_hash }  // O(1) comparison
```

## テストスイート

コア module、async P2P 基盤、ブリッジ連携、FFI にまたがる 195 個のテスト

| 範囲 | コマンド | テスト数 |
|-------|---------|-------|
| コア | `cargo test` | 139 |
| + Async P2P | `cargo test --features async` | 195 |
| + Physics | `cargo test --features physics` | +6 |
| + 全ブリッジ | `cargo test --features all-bridges,cloud,ffi` | すべて |

```bash
cargo test                          # Core tests (139)
cargo test --features async         # Core + P2P infrastructure (195)
cargo bench                         # Benchmarks
```

### 品質

| 指標 | 値 |
|--------|-------|
| clippy (pedantic+nursery) | 警告 0 |
| fmt | 差分なし |
| テスト (既定) | 139 |
| テスト (async) | 195 |

## ライセンス

`AGPL-3.0-or-later OR LicenseRef-Commercial` — デュアルライセンス どちらかを選べる

| 選択肢 | 条件 | 向いている場合 |
|--------|-------|-------------|
| **AGPL-3.0-or-later** | [LICENSE-AGPL](LICENSE-AGPL) — 無償、報告義務なし | プロジェクト自体が AGPL 互換のオープンソースである、または社内でのみ利用する |
| **商用ライセンス** | [LICENSE-COMMERCIAL.md](LICENSE-COMMERCIAL.md) — 有償、コピーレフトを外す | クローズドソースの製品、独自仕様の SaaS、エッジやファームウェアでの配布、plugin の再配布、ソース開示を禁じるプラットフォームの NDA がある場合 |

AGPL は強いコピーレフト `alice-sync` をリンクし、ユーザーに配布または提供する製品、
ファームウェアイメージ、サービスは、同じく AGPL で公開しなければならない これはオープンな
エコシステムのための意図的な仕様で、商用ライセンスは、それができない場合のためにある

商用ライセンスの問い合わせ: <contact@extoria.co.jp>

## 作者

Moroya Sakamoto

---

*「最良の同期とは、データが移動しない同期である」*

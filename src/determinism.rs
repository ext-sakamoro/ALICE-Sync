//! 非決定性検出ツール（差分デバッガ）
//!
//! 複数ノード間の状態ハッシュを比較し、同期のずれ（非決定性）を検出する。
//! フレーム単位でどの時点で状態が分岐したかを特定し、
//! フィールド別チェックサムにより原因箇所を絞り込む。
//!
//! # 使い方
//!
//! ```rust
//! use alice_sync::determinism::{DeterminismChecker, StateSnapshot};
//!
//! let mut checker = DeterminismChecker::new(2); // 2ノード
//! checker.record(0, 1, StateSnapshot::new(0x1234, &[0xAA, 0xBB]));
//! checker.record(1, 1, StateSnapshot::new(0x1234, &[0xAA, 0xBB]));
//! assert!(checker.check_frame(1).is_none()); // 一致
//! ```

/// フレーム時点の状態スナップショット。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateSnapshot {
    /// ワールド全体のハッシュ。
    pub world_hash: u64,
    /// フィールド別チェックサム（位置, 速度, プロパティ等）。
    pub field_checksums: Vec<u64>,
}

impl StateSnapshot {
    /// 新しいスナップショットを作成。
    #[must_use]
    pub fn new(world_hash: u64, field_checksums: &[u64]) -> Self {
        Self {
            world_hash,
            field_checksums: field_checksums.to_vec(),
        }
    }

    /// 空のスナップショット。
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            world_hash: 0,
            field_checksums: Vec::new(),
        }
    }
}

/// 非決定性検出結果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Divergence {
    /// 分岐が検出されたフレーム番号。
    pub frame: u64,
    /// 分岐したノードペア (`node_a`, `node_b`)。
    pub node_a: usize,
    /// 分岐したノード B。
    pub node_b: usize,
    /// ワールドハッシュ (`node_a`)。
    pub hash_a: u64,
    /// ワールドハッシュ (`node_b`)。
    pub hash_b: u64,
    /// 分岐したフィールドのインデックス一覧。
    pub divergent_fields: Vec<usize>,
}

/// 非決定性検出器。
///
/// 各ノードのフレーム毎の状態スナップショットを記録し、
/// ノード間の差分を検出する。
#[derive(Debug, Clone)]
pub struct DeterminismChecker {
    /// ノード数。
    node_count: usize,
    /// 記録: `records[node_id]` = Vec<(frame, snapshot)>。
    records: Vec<Vec<(u64, StateSnapshot)>>,
    /// 検出済み分岐のフレーム番号。
    first_divergence_frame: Option<u64>,
}

impl DeterminismChecker {
    /// 指定ノード数で作成。
    #[must_use]
    pub fn new(node_count: usize) -> Self {
        Self {
            node_count,
            records: vec![Vec::new(); node_count],
            first_divergence_frame: None,
        }
    }

    /// ノードのフレーム状態を記録。
    ///
    /// # Panics
    ///
    /// `node_id >= node_count` の場合パニック。
    pub fn record(&mut self, node_id: usize, frame: u64, snapshot: StateSnapshot) {
        assert!(node_id < self.node_count, "node_id out of range");
        self.records[node_id].push((frame, snapshot));
    }

    /// 指定フレームで分岐があるかチェック。
    ///
    /// 分岐がある場合、最初のペアの `Divergence` を返す。
    #[must_use]
    pub fn check_frame(&self, frame: u64) -> Option<Divergence> {
        // 各ノードの該当フレームスナップショットを収集
        let snapshots: Vec<Option<&StateSnapshot>> = (0..self.node_count)
            .map(|n| {
                self.records[n]
                    .iter()
                    .find(|(f, _)| *f == frame)
                    .map(|(_, s)| s)
            })
            .collect();

        // ペアワイズ比較
        for i in 0..self.node_count {
            for j in (i + 1)..self.node_count {
                if let (Some(si), Some(sj)) = (snapshots[i], snapshots[j]) {
                    if si.world_hash != sj.world_hash {
                        let divergent_fields = find_divergent_fields(si, sj);
                        return Some(Divergence {
                            frame,
                            node_a: i,
                            node_b: j,
                            hash_a: si.world_hash,
                            hash_b: sj.world_hash,
                            divergent_fields,
                        });
                    }
                }
            }
        }
        None
    }

    /// 全記録フレームを走査し、最初の分岐フレームを検出。
    ///
    /// 分岐がない場合 `None`。
    pub fn find_first_divergence(&mut self) -> Option<Divergence> {
        // 全フレーム番号を収集（ソート済みユニーク）
        let mut frames: Vec<u64> = self
            .records
            .iter()
            .flat_map(|r| r.iter().map(|(f, _)| *f))
            .collect();
        frames.sort_unstable();
        frames.dedup();

        for frame in frames {
            if let Some(div) = self.check_frame(frame) {
                self.first_divergence_frame = Some(frame);
                return Some(div);
            }
        }
        None
    }

    /// 最初の分岐フレーム番号（`find_first_divergence` 呼び出し後）。
    #[must_use]
    pub const fn first_divergence_frame(&self) -> Option<u64> {
        self.first_divergence_frame
    }

    /// ノード数。
    #[must_use]
    pub const fn node_count(&self) -> usize {
        self.node_count
    }

    /// 指定ノードの記録フレーム数。
    #[must_use]
    pub fn frame_count(&self, node_id: usize) -> usize {
        self.records.get(node_id).map_or(0, Vec::len)
    }

    /// 全記録をクリア。
    pub fn clear(&mut self) {
        for r in &mut self.records {
            r.clear();
        }
        self.first_divergence_frame = None;
    }
}

/// フィールド別チェックサムの差分インデックスを返す。
fn find_divergent_fields(a: &StateSnapshot, b: &StateSnapshot) -> Vec<usize> {
    let len = a.field_checksums.len().min(b.field_checksums.len());
    let mut result = Vec::new();
    for i in 0..len {
        if a.field_checksums[i] != b.field_checksums[i] {
            result.push(i);
        }
    }
    // 長さが異なる場合、余剰フィールドも分岐扱い
    let max_len = a.field_checksums.len().max(b.field_checksums.len());
    for i in len..max_len {
        result.push(i);
    }
    result
}

/// 簡易 FNV-1a ハッシュ（フィールドチェックサム生成用）。
#[must_use]
pub fn fnv1a_hash(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_divergence_identical() {
        let mut checker = DeterminismChecker::new(2);
        checker.record(0, 1, StateSnapshot::new(0x1234, &[0xAA, 0xBB]));
        checker.record(1, 1, StateSnapshot::new(0x1234, &[0xAA, 0xBB]));
        assert!(checker.check_frame(1).is_none());
    }

    #[test]
    fn divergence_detected() {
        let mut checker = DeterminismChecker::new(2);
        checker.record(0, 1, StateSnapshot::new(0x1111, &[0xAA]));
        checker.record(1, 1, StateSnapshot::new(0x2222, &[0xBB]));
        let div = checker.check_frame(1).unwrap();
        assert_eq!(div.frame, 1);
        assert_eq!(div.hash_a, 0x1111);
        assert_eq!(div.hash_b, 0x2222);
        assert_eq!(div.divergent_fields, vec![0]);
    }

    #[test]
    fn find_first_divergence_multi_frames() {
        let mut checker = DeterminismChecker::new(2);
        // フレーム1: 一致
        checker.record(0, 1, StateSnapshot::new(0xAA, &[]));
        checker.record(1, 1, StateSnapshot::new(0xAA, &[]));
        // フレーム2: 分岐
        checker.record(0, 2, StateSnapshot::new(0xBB, &[]));
        checker.record(1, 2, StateSnapshot::new(0xCC, &[]));
        // フレーム3: 分岐
        checker.record(0, 3, StateSnapshot::new(0xDD, &[]));
        checker.record(1, 3, StateSnapshot::new(0xEE, &[]));

        let div = checker.find_first_divergence().unwrap();
        assert_eq!(div.frame, 2);
        assert_eq!(checker.first_divergence_frame(), Some(2));
    }

    #[test]
    fn no_divergence_returns_none() {
        let mut checker = DeterminismChecker::new(3);
        for node in 0..3 {
            checker.record(node, 1, StateSnapshot::new(0xFF, &[1, 2, 3]));
        }
        assert!(checker.check_frame(1).is_none());
    }

    #[test]
    fn three_nodes_divergence() {
        let mut checker = DeterminismChecker::new(3);
        checker.record(0, 1, StateSnapshot::new(0xAA, &[]));
        checker.record(1, 1, StateSnapshot::new(0xAA, &[]));
        checker.record(2, 1, StateSnapshot::new(0xBB, &[])); // node 2 diverges
        let div = checker.check_frame(1).unwrap();
        assert_eq!(div.node_a, 0);
        assert_eq!(div.node_b, 2);
    }

    #[test]
    fn divergent_fields_partial() {
        let mut checker = DeterminismChecker::new(2);
        checker.record(0, 1, StateSnapshot::new(0x11, &[1, 2, 3]));
        checker.record(1, 1, StateSnapshot::new(0x22, &[1, 9, 3])); // field[1] differs
        let div = checker.check_frame(1).unwrap();
        assert_eq!(div.divergent_fields, vec![1]);
    }

    #[test]
    fn divergent_fields_length_mismatch() {
        let a = StateSnapshot::new(0x11, &[1, 2]);
        let b = StateSnapshot::new(0x22, &[1, 2, 3]);
        let fields = find_divergent_fields(&a, &b);
        assert_eq!(fields, vec![2]); // field[2] only in b
    }

    #[test]
    fn frame_count() {
        let mut checker = DeterminismChecker::new(2);
        checker.record(0, 1, StateSnapshot::empty());
        checker.record(0, 2, StateSnapshot::empty());
        checker.record(1, 1, StateSnapshot::empty());
        assert_eq!(checker.frame_count(0), 2);
        assert_eq!(checker.frame_count(1), 1);
    }

    #[test]
    fn clear_resets() {
        let mut checker = DeterminismChecker::new(2);
        checker.record(0, 1, StateSnapshot::new(0xAA, &[]));
        checker.record(1, 1, StateSnapshot::new(0xBB, &[]));
        checker.find_first_divergence();
        assert!(checker.first_divergence_frame().is_some());

        checker.clear();
        assert!(checker.first_divergence_frame().is_none());
        assert_eq!(checker.frame_count(0), 0);
    }

    #[test]
    fn node_count() {
        let checker = DeterminismChecker::new(5);
        assert_eq!(checker.node_count(), 5);
    }

    #[test]
    fn missing_frame_no_divergence() {
        let mut checker = DeterminismChecker::new(2);
        checker.record(0, 1, StateSnapshot::new(0xAA, &[]));
        // node 1 has no frame 1 record
        assert!(checker.check_frame(1).is_none());
    }

    #[test]
    fn fnv1a_deterministic() {
        let h1 = fnv1a_hash(b"hello");
        let h2 = fnv1a_hash(b"hello");
        assert_eq!(h1, h2);
        assert_ne!(fnv1a_hash(b"hello"), fnv1a_hash(b"world"));
    }

    #[test]
    fn fnv1a_empty() {
        let h = fnv1a_hash(b"");
        assert_eq!(h, 0xcbf2_9ce4_8422_2325); // FNV offset basis
    }

    #[test]
    fn snapshot_empty() {
        let s = StateSnapshot::empty();
        assert_eq!(s.world_hash, 0);
        assert!(s.field_checksums.is_empty());
    }

    #[test]
    #[should_panic(expected = "node_id out of range")]
    fn record_out_of_range_panics() {
        let mut checker = DeterminismChecker::new(2);
        checker.record(5, 1, StateSnapshot::empty());
    }

    #[test]
    fn frame_count_out_of_range() {
        let checker = DeterminismChecker::new(2);
        assert_eq!(checker.frame_count(99), 0);
    }
}

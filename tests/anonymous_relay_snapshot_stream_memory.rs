//! The memory a snapshot takes, measured: the heap bytes live at once on this thread while a full room
//! (256 logged payloads of 65,438 bytes, a snapshot of about 16.7 MB) is written out
//!
//! - by `snapshot()`: the whole snapshot in one buffer;
//! - by `snapshot_writer()` in pieces of 120 KiB, each piece stored (here: dropped) before the next is
//!   made, with and without the core changing between the pieces.
//!
//! The bound asserted for the pieces: the piece buffer (`max`), plus the writer's own buffer
//! (`SnapshotWriter::BUFFER`), plus per room one reference, plus (when the core changes a room the
//! writer still holds) one copy of that room's fields without its payloads, plus what the core itself
//! allocates for the change (the published payload). No payload is copied: the peak stays far below
//! one payload per logged item.
//!
//! Measured with a counting allocator, so this file has one test (a second one running on another
//! thread is not counted, but the count is per thread anyway).

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::num::NonZeroUsize;

use alice_sync::anonymous_relay::{
    AnonymousRelayCore, ConnId, Limits, RoomAlias, RoomToken, SenderTag, SnapshotWriter,
};

struct Counting;

thread_local! {
    static LIVE: Cell<isize> = const { Cell::new(0) };
    static PEAK: Cell<isize> = const { Cell::new(0) };
}

fn add(n: isize) {
    // try_with: the thread-locals may be gone while the thread exits
    let _ = LIVE.try_with(|live| {
        let now = live.get() + n;
        live.set(now);
        let _ = PEAK.try_with(|peak| peak.set(peak.get().max(now)));
    });
}

// SAFETY: every call is forwarded to the system allocator unchanged; only sizes are counted
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        add(layout.size() as isize);
        // SAFETY: the caller's contract is the system allocator's
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        add(-(layout.size() as isize));
        // SAFETY: as above
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        add(new_size as isize - layout.size() as isize);
        // SAFETY: as above
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// The heap bytes live at the peak of `f`, above what was live when it started
fn peak_of<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let base = LIVE.with(Cell::get);
    PEAK.with(|p| p.set(base));
    let out = f();
    let peak = PEAK.with(Cell::get) - base;
    (out, usize::try_from(peak).unwrap_or(0))
}

const ROOM: RoomAlias = RoomAlias::from_raw([0x11; 32]);
const TOKEN: RoomToken = RoomToken::from_raw([0xa1; 32]);
const A: ConnId = ConnId::from_raw(1);
const B: ConnId = ConnId::from_raw(2);
const MAX_PAYLOAD: usize = 65_438;
const MAX_LOG: usize = 256;
const PIECE: usize = 120 * 1024;
/// The fields of one logged item as the core keeps it (index, tag, sequence number, a shared
/// payload reference), with room for the allocator's rounding
const PER_ITEM: usize = 64;

#[test]
fn snapshot_in_pieces_peak_memory_is_the_piece_not_the_state() {
    let limits = Limits {
        max_payload: MAX_PAYLOAD,
        max_log: MAX_LOG,
        max_pending: 4,
        max_rooms: 4,
        max_members: 8,
        max_publishes_per_window: u32::MAX,
        window: 1,
        max_rooms_per_conn: 4,
        max_tags_per_room: 8,
    };
    let mut core = AnonymousRelayCore::new(limits);
    core.bind(A, ROOM, TOKEN, SenderTag::from_raw([0x0a; 16]), 0);
    core.bind(B, ROOM, TOKEN, SenderTag::from_raw([0x0b; 16]), 0);
    for s in 0..MAX_LOG as u64 {
        core.publish(A, ROOM, s, vec![(s % 251) as u8; MAX_PAYLOAD], 1);
    }
    let logged = MAX_LOG * MAX_PAYLOAD;

    // one buffer
    let (whole, snapshot_peak) = peak_of(|| core.snapshot().expect("representable"));
    let total = whole.len();
    assert!(total > logged, "{total}");
    assert!(
        snapshot_peak >= total,
        "snapshot() peak {snapshot_peak} < {total}"
    );

    // in pieces, the core unchanged
    let max = NonZeroUsize::new(PIECE).expect("positive");
    let ((written, high_water), pieces_peak) = peak_of(|| {
        let mut w = core.snapshot_writer().expect("representable");
        let mut piece = Vec::with_capacity(PIECE);
        let mut written = 0usize;
        loop {
            piece.clear();
            let more = w.next_chunk(max, &mut piece);
            written += piece.len();
            if !more {
                break;
            }
        }
        (written, w.buffer_high_water())
    });
    assert_eq!(written, total, "the pieces wrote the whole snapshot");
    assert!(
        high_water > 0 && high_water <= SnapshotWriter::BUFFER,
        "{high_water}"
    );
    let bound = PIECE + SnapshotWriter::BUFFER + 4096;
    assert!(
        pieces_peak <= bound,
        "pieces peak {pieces_peak} > {bound} (snapshot() peak {snapshot_peak})"
    );

    // in pieces, the core publishing a full payload between every two pieces (the first change copies
    // the room's fields once, without its payloads; each publish allocates its own payload, which the
    // core keeps and the log's oldest is dropped)
    let mut seq = MAX_LOG as u64;
    let ((written, _), changing_peak) = peak_of(|| {
        let mut w = core.snapshot_writer().expect("representable");
        let mut piece = Vec::with_capacity(PIECE);
        let mut written = 0usize;
        loop {
            piece.clear();
            let more = w.next_chunk(max, &mut piece);
            written += piece.len();
            if !more {
                break;
            }
            core.publish(A, ROOM, seq, vec![(seq % 251) as u8; MAX_PAYLOAD], 2);
            seq += 1;
        }
        (written, w.buffer_high_water())
    });
    assert_eq!(written, total);
    // every payload published here outlives the closure in the core or in the writer's frozen
    // room, until the writer is dropped: the writer holds the room as it was, the core the new log
    let published = usize::try_from(seq).unwrap_or(0) - MAX_LOG;
    assert!(published > 100, "{published} publishes");
    // what stays of the publishes is the core's own data; the rest is the piece, the writer's buffer,
    // the copy of the room's fields (one per logged item, payloads shared), the core's log growing
    // past the copied one, and the transient copies a publish makes (the payload passed in and the
    // forward to the other member)
    let kept = published * MAX_PAYLOAD;
    let overhead = changing_peak.saturating_sub(kept);
    let bound = PIECE + SnapshotWriter::BUFFER + 2 * MAX_PAYLOAD + 4 * MAX_LOG * PER_ITEM + 4096;
    assert!(
        changing_peak >= kept && overhead <= bound,
        "changing peak {changing_peak}, overhead over the kept payloads {overhead} > {bound}"
    );

    println!(
        "full room: snapshot {total} B; snapshot() peak {snapshot_peak} B; \
         pieces of {PIECE} B peak {pieces_peak} B (writer buffer high water {high_water} B); \
         with a {MAX_PAYLOAD} B publish between pieces ({published} publishes) peak {changing_peak} B, \
         of which {overhead} B over the payloads the core keeps"
    );
}

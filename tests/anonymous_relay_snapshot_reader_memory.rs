//! The memory [`SnapshotReader`] takes, measured: the heap bytes live at once on this thread while a
//! full room (256 logged payloads of 65,438 bytes, a snapshot of about 16.7 MB) is read back in pieces
//! of 120 KiB, compared with the core it produces and with `restore` / `restore_from_pieces`.
//!
//! The bound asserted: above the core the read produces (live when it returns), the reader holds at
//! most one payload being completed across pieces (at most `max_payload`) plus a fixed overhead (the
//! decode's bookkeeping for one room). It does not keep the pieces it was fed: those are the caller's,
//! and here each one is a slice of a buffer allocated before the measurement.
//!
//! Measured with a counting allocator, so this file has one test (the count is per thread).

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use alice_sync::anonymous_relay::{
    AnonymousRelayCore, ConnId, Limits, RoomAlias, RoomToken, SenderTag, SnapshotError,
    SnapshotReader,
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

/// The heap bytes live at the peak of `f` above what was live when it started, and above what is
/// still live when it returns (the core it produced, kept)
fn peak_above_result<T>(f: impl FnOnce() -> T) -> (T, usize, usize) {
    let base = LIVE.with(Cell::get);
    PEAK.with(|p| p.set(base));
    let out = f();
    let peak = PEAK.with(Cell::get) - base;
    let kept = LIVE.with(Cell::get) - base;
    (
        out,
        usize::try_from(peak).unwrap_or(0),
        usize::try_from(kept).unwrap_or(0),
    )
}

const ROOM: RoomAlias = RoomAlias::from_raw([0x11; 32]);
const TOKEN: RoomToken = RoomToken::from_raw([0xa1; 32]);
const A: ConnId = ConnId::from_raw(1);
const B: ConnId = ConnId::from_raw(2);
const MAX_PAYLOAD: usize = 65_438;
const MAX_LOG: usize = 256;
const PIECE: usize = 120 * 1024;
/// The reader's own bookkeeping besides the payload it completes: its fixed fields, and for the room
/// being read the sets it checks the room against (a few tags and connections), with room for the
/// allocator's rounding
const FIXED: usize = 8 * 1024;

fn read_in_pieces(bytes: &[u8]) -> Result<AnonymousRelayCore, SnapshotError> {
    let mut r = SnapshotReader::new();
    for piece in bytes.chunks(PIECE) {
        r.feed(piece)?;
    }
    r.finish()
}

#[test]
fn reader_holds_one_payload_above_the_core() {
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
    // a payload held back too, so a held entry spans pieces as well as logged ones
    core.publish(B, ROOM, 1, vec![7; MAX_PAYLOAD], 2);
    let bytes = core.snapshot().expect("representable");
    drop(core);
    let logged = MAX_LOG * MAX_PAYLOAD;
    assert!(bytes.len() > logged, "{}", bytes.len());

    let (pushed, push_peak, push_kept) = peak_above_result(|| read_in_pieces(&bytes));
    let pushed = pushed.expect("the snapshot reads back");
    assert!(pushed.snapshot().expect("representable") == bytes);
    // the core holds every payload: what the read keeps is at least the logged bytes
    assert!(push_kept >= logged, "kept {push_kept} < {logged}");
    let push_above = push_peak - push_kept;
    drop(pushed);

    let (pieces, pieces_peak, pieces_kept) = peak_above_result(|| {
        AnonymousRelayCore::restore_from_pieces(bytes.chunks(PIECE)).expect("reads back")
    });
    let pieces_above = pieces_peak - pieces_kept;
    drop(pieces);

    let (whole, whole_peak, whole_kept) =
        peak_above_result(|| AnonymousRelayCore::restore(&bytes).expect("reads back"));
    let whole_above = whole_peak - whole_kept;
    drop(whole);

    println!(
        "full room: snapshot {} B; reader in pieces of {PIECE} B: peak {push_peak} B, core {push_kept} B, \
         above the core {push_above} B; restore_from_pieces: peak {pieces_peak} B, above {pieces_above} B; \
         restore: peak {whole_peak} B, above {whole_above} B",
        bytes.len()
    );
    // the same core from each read
    assert_eq!(push_kept, pieces_kept);
    assert_eq!(push_kept, whole_kept);
    let bound = MAX_PAYLOAD + FIXED;
    assert!(
        push_above <= bound,
        "reader: {push_above} B above the core > {bound}"
    );
    assert!(
        pieces_above <= bound,
        "restore_from_pieces: {pieces_above} B above the core > {bound}"
    );
}

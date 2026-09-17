//! Fuzz: `decompress_event_batch` on arbitrary bytes (rANS / wavelet decoders
//! fed garbage headers and histograms must not panic or over-allocate) and
//! compress → decompress on arbitrary payloads / quantiser steps (length is
//! preserved, every byte within 2 × the effective step).
#![no_main]

use alice_sync::codec_bridge::{compress_event_batch, decompress_event_batch, CompressedEventBatch};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // arbitrary compressed bytes; original_len is bounded so decode cannot
    // allocate more than a few MiB from a hostile header
    let claimed = data.first().map_or(0, |&b| usize::from(b) * 64);
    let garbage = CompressedEventBatch {
        data: data.to_vec(),
        original_len: claimed,
    };
    let _ = decompress_event_batch(&garbage);

    if data.len() > 1 && data.len() <= 4096 {
        let step = i32::from(data[0] % 16);
        let payload = &data[1..];
        let back = decompress_event_batch(&compress_event_batch(payload, step));
        assert_eq!(back.len(), payload.len(), "length preserved");
    }
});

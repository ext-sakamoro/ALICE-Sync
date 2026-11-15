//! The body of `GET /relay-revocations` (`alice_sync::relay_revocations_body`), with no feature: the layout a
//! relay serves and a client decodes.
//!
//! The expected bytes are built here from the parts and the layout the module states
//! (`version (1) = 1 || certificate length (u16 BE) || certificate || list length (u32 BE) || list`), not by
//! calling the encoder; the caps are the sums the constants state, written as literals. The golden hex pins
//! one body (fixed test-only seeds; Ed25519 signatures are deterministic) so a change of the layout or of the
//! parts' encodings shows. Every key is a fixed test-only seed.

use alice_identity::cert_revocation::RevocationList;
use alice_identity::delegated_revocation::issue_revocation_delegate;
use alice_identity::device_cert::{DeviceCert, DeviceId, MAX_DEVICE_CERT_LEN};
use alice_sign::KeyPair;
use alice_sync::relay_revocations_body::{
    BodyError, DelegatedListResponse, MAX_RELAY_DELEGATED_REVOCATIONS_LEN,
    MAX_RELAY_REVOCATIONS_LEN, MAX_RELAY_REVOCATION_IDS, RELAY_REVOCATIONS_BODY_VERSION,
};

const T0: u64 = 1_800_000_000;

fn master() -> KeyPair {
    KeyPair::from_seed([0x5a; 32])
}
fn delegate_key() -> KeyPair {
    KeyPair::from_seed([0x21; 32])
}
fn delegate() -> DeviceCert {
    issue_revocation_delegate(&master(), delegate_key().public().0, T0, T0 + 7 * 86_400).unwrap()
}
fn list(revoked: &[DeviceId]) -> RevocationList {
    RevocationList::issue(&delegate_key(), 3, T0 + 10, revoked)
}
fn body() -> DelegatedListResponse {
    DelegatedListResponse {
        list: list(&[DeviceId::from_raw([1; 16])]),
        delegate: delegate(),
    }
}

/// The layout, built from the parts
fn layout(version: u8, cert: &[u8], list: &[u8]) -> Vec<u8> {
    let mut b = vec![version];
    b.extend_from_slice(&(cert.len() as u16).to_be_bytes());
    b.extend_from_slice(cert);
    b.extend_from_slice(&(list.len() as u32).to_be_bytes());
    b.extend_from_slice(list);
    b
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
fn the_constants_are_the_stated_sums() {
    assert_eq!(RELAY_REVOCATIONS_BODY_VERSION, 1);
    assert_eq!(MAX_RELAY_REVOCATION_IDS, 4096);
    // version 1, issuer 32, sequence 8, issued-at 8, count 4, 16 per id, signature 64
    assert_eq!(MAX_RELAY_REVOCATIONS_LEN, 65_653);
    assert_eq!(
        MAX_RELAY_DELEGATED_REVOCATIONS_LEN,
        1 + 2 + MAX_DEVICE_CERT_LEN + 4 + 65_653
    );
}

#[test]
fn a_body_is_the_certificate_then_the_list_and_decodes_back() {
    let b = body();
    let bytes = b.to_bytes().unwrap();
    assert_eq!(bytes, layout(1, &b.delegate.to_bytes(), &b.list.to_bytes()));
    assert_eq!(DelegatedListResponse::from_bytes(&bytes), Ok(b));
    // an empty list too
    let empty = DelegatedListResponse {
        list: list(&[]),
        delegate: delegate(),
    };
    assert_eq!(
        DelegatedListResponse::from_bytes(&empty.to_bytes().unwrap()),
        Ok(empty)
    );
}

#[test]
fn the_golden_body() {
    let bytes = body().to_bytes().unwrap();
    assert_eq!(hex(&bytes), GOLDEN_BODY_HEX);
}

#[test]
fn decoding_is_strict() {
    let b = body();
    let cert = b.delegate.to_bytes();
    let lb = b.list.to_bytes();
    let good = layout(1, &cert, &lb);
    let mut trailing = good.clone();
    trailing.push(0);
    let short = &good[..good.len() - 1];
    let cert_only = &good[..1 + 2 + cert.len()];
    for bad in [
        layout(0, &cert, &lb).as_slice(),
        layout(2, &cert, &lb).as_slice(),
        trailing.as_slice(),
        short,
        cert_only,
        &[],
        &[1u8],
        &[1u8, 0],
    ] {
        assert_eq!(
            DelegatedListResponse::from_bytes(bad),
            Err(BodyError::Malformed),
            "{}",
            hex(bad)
        );
    }
    // a part that is not one
    assert!(matches!(
        DelegatedListResponse::from_bytes(&layout(1, &[0xff], &lb)),
        Err(BodyError::Cert(_))
    ));
    assert!(matches!(
        DelegatedListResponse::from_bytes(&layout(1, &cert, &[0xff])),
        Err(BodyError::Revocations(_))
    ));
}

#[test]
fn a_length_over_its_cap_is_refused_before_the_part_is_read() {
    // only the head of each part: a declared length over the cap is TooLarge, at the cap it is Malformed
    // (the part is short), so the bound is the cap itself
    let over_cert = (MAX_DEVICE_CERT_LEN as u16 + 1).to_be_bytes();
    assert_eq!(
        DelegatedListResponse::from_bytes(&[1, over_cert[0], over_cert[1]]),
        Err(BodyError::TooLarge)
    );
    let at_cert = (MAX_DEVICE_CERT_LEN as u16).to_be_bytes();
    assert_eq!(
        DelegatedListResponse::from_bytes(&[1, at_cert[0], at_cert[1]]),
        Err(BodyError::Malformed)
    );
    let cert = delegate().to_bytes();
    let mut head = vec![1u8];
    head.extend_from_slice(&(cert.len() as u16).to_be_bytes());
    head.extend_from_slice(&cert);
    let mut over_list = head.clone();
    over_list.extend_from_slice(&(MAX_RELAY_REVOCATIONS_LEN as u32 + 1).to_be_bytes());
    assert_eq!(
        DelegatedListResponse::from_bytes(&over_list),
        Err(BodyError::TooLarge)
    );
    let mut at_list = head;
    at_list.extend_from_slice(&(MAX_RELAY_REVOCATIONS_LEN as u32).to_be_bytes());
    assert_eq!(
        DelegatedListResponse::from_bytes(&at_list),
        Err(BodyError::Malformed)
    );
}

#[test]
fn the_longest_list_fits_its_cap_and_one_more_id_does_not() {
    let ids: Vec<_> = (0..MAX_RELAY_REVOCATION_IDS)
        .map(|i| {
            let mut raw = [0u8; 16];
            raw[..8].copy_from_slice(&(i as u64).to_be_bytes());
            DeviceId::from_raw(raw)
        })
        .collect();
    let longest = DelegatedListResponse {
        list: list(&ids),
        delegate: delegate(),
    };
    assert_eq!(longest.list.to_bytes().len(), MAX_RELAY_REVOCATIONS_LEN);
    let bytes = longest.to_bytes().unwrap();
    assert!(bytes.len() <= MAX_RELAY_DELEGATED_REVOCATIONS_LEN);
    assert_eq!(DelegatedListResponse::from_bytes(&bytes), Ok(longest));

    let mut more = ids;
    more.push(DeviceId::from_raw([0xff; 16]));
    let over = DelegatedListResponse {
        list: list(&more),
        delegate: delegate(),
    };
    // the relay refuses to serve it, and a body carrying it is refused by the client
    assert_eq!(over.to_bytes(), Err(BodyError::TooLarge));
    let over_body = layout(1, &over.delegate.to_bytes(), &over.list.to_bytes());
    assert_eq!(
        DelegatedListResponse::from_bytes(&over_body),
        Err(BodyError::TooLarge)
    );
}

#[test]
fn the_body_errors_read_as_text() {
    assert_eq!(BodyError::Malformed.to_string(), "not a revocation body");
    assert_eq!(
        BodyError::TooLarge.to_string(),
        "revocation body part over its cap"
    );
}

/// One body: delegate key 0x21 under master 0x5a (epoch T0, 7 days), list sequence 3 at T0 + 10 revoking
/// one id (16 bytes of 1)
const GOLDEN_BODY_HEX: &str = concat!(
    // version 1, certificate length 230
    "0100e6",
    // the delegate certificate
    "0300236469643a616c6963652d736572766963653a72656c61792d7265766f636174696f6e73884b8857f4eaa1613c61",
    "504db34d4beaf346517a0e31de3cddd4d9b4201d9d0b0000000000000000000000000000000000000000000000000000",
    "000000000000000000006b49d200000000006b530c8011f28911c39163f57010a50d7a7503b80d7550754e0800a5d237",
    "eef5826035766b9b3e5a15868a940ab289958788e3b060523fb9f9f0078137a3dc682f955222f11aeb8da110c2e2d16e",
    "94f96c7906acfb21ce228226e42d4bb37e752236d398f8ef5d1551167d544eb2970d95d44c0b",
    // list length 133
    "00000085",
    // the list (one id)
    "01884b8857f4eaa1613c61504db34d4beaf346517a0e31de3cddd4d9b4201d9d0b0000000000000003000000006b49d2",
    "0a0000000101010101010101010101010101010101f3d462957187a352b1ba3ddb9b4a63e0231c11fca1da146f5d36ab",
    "1ef226cdb9fec22fd48aa9c5320db255f8b431000fdc629539cfeb1c995dbcfa631c0de700",
);

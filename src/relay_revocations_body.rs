//! The body of `GET /relay-revocations`: the operator's revocation list with the certificate of the delegate
//! key that signed it, its layout version and its length caps.
//!
//! No feature: the layout needs only `alice-identity` (a required dependency), so a relay built with this
//! crate serves the body with [`DelegatedListResponse::to_bytes`] without the blocking client
//! (`anonymous-relay-client`), and a client reads it with [`DelegatedListResponse::from_bytes`] (which
//! `anonymous_relay_client::fetch_relay_revocations` calls). Both sides use the one codec, so the layout
//! is written once.
//!
//! The operator's master key stays offline and certifies a delegate key that signs the lists
//! (`alice_identity::delegated_revocation`). A list is checked against the delegate certificate that vouches
//! for its signer, so the relay serves both in one body:
//!
//! ```text
//! body = version (1) = 1 || certificate length (u16 BE) || delegate certificate
//!        || list length (u32 BE) || list
//! ```
//!
//! One body, not two paths: the certificate and the list are read from one response, so a delegate rotated
//! between two requests cannot pair a new certificate with an old list (which would read as a wrong signer).
//!
//! Decoding does not verify: trust comes from checking the parts under the operator's master key
//! (`alice_identity::delegated_revocation::DelegatedRevocations::accept`).

use alice_identity::cert_revocation::{RevocationList, RevocationListError};
use alice_identity::device_cert::{DeviceCert, DeviceCertError, MAX_DEVICE_CERT_LEN};

/// The most device ids a revocation list in a body may hold: the list format's own cap
/// ([`alice_identity::cert_revocation::MAX_REVOKED`], 4096), which `RevocationList::issue` and
/// `RevocationList::from_bytes` enforce.
///
/// Within a delegate's epoch a newer list keeps every id of the older one, so the list grows by one id per relay
/// key the operator revokes (a new epoch may drop ids); 4096 ids is over three centuries of one revocation a month.
/// A body whose list is longer than [`MAX_RELAY_REVOCATIONS_LEN`] is refused as [`BodyError::TooLarge`] before the
/// list is read.
pub const MAX_RELAY_REVOCATION_IDS: usize = alice_identity::cert_revocation::MAX_REVOKED;

/// The longest list a body carries: [`MAX_RELAY_REVOCATION_IDS`] ids in the list's canonical layout
/// (version 1, issuer key 32, sequence 8, issued-at 8, count 4, 16 per id, signature 64), the list format's
/// [`alice_identity::cert_revocation::MAX_REVOCATION_LIST_LEN`].
pub const MAX_RELAY_REVOCATIONS_LEN: usize =
    alice_identity::cert_revocation::MAX_REVOCATION_LIST_LEN;

/// The layout version of the revocation body ([`DelegatedListResponse::to_bytes`]).
pub const RELAY_REVOCATIONS_BODY_VERSION: u8 = 1;

/// The longest body: the version, the longest certificate there is ([`MAX_DEVICE_CERT_LEN`]) and the longest
/// list ([`MAX_RELAY_REVOCATIONS_LEN`]) with their lengths.
pub const MAX_RELAY_DELEGATED_REVOCATIONS_LEN: usize =
    1 + 2 + MAX_DEVICE_CERT_LEN + 4 + MAX_RELAY_REVOCATIONS_LEN;

/// What `GET /relay-revocations` serves: the operator's revocation list and the certificate of the delegate key
/// that signed it (see the module documentation for the layout).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DelegatedListResponse {
    /// The list, as decoded (not verified).
    pub list: RevocationList,
    /// The delegate's certificate, as decoded (not verified).
    pub delegate: DeviceCert,
}

impl DelegatedListResponse {
    /// The body a relay serves (the layout of the module documentation).
    ///
    /// # Errors
    ///
    /// [`BodyError::TooLarge`] for a certificate over [`MAX_DEVICE_CERT_LEN`] or a list over
    /// [`MAX_RELAY_REVOCATIONS_LEN`]: the caps [`Self::from_bytes`] applies, so a relay never serves a body
    /// a client refuses. Both caps are under the u16 / u32 length fields.
    pub fn to_bytes(&self) -> Result<Vec<u8>, BodyError> {
        let cert = self.delegate.to_bytes();
        let list = self.list.to_bytes();
        let cert_len = capped_len::<u16>(cert.len(), MAX_DEVICE_CERT_LEN)?;
        let list_len = capped_len::<u32>(list.len(), MAX_RELAY_REVOCATIONS_LEN)?;
        let mut b = Vec::with_capacity(1 + 2 + cert.len() + 4 + list.len());
        b.push(RELAY_REVOCATIONS_BODY_VERSION);
        b.extend_from_slice(&cert_len.to_be_bytes());
        b.extend_from_slice(&cert);
        b.extend_from_slice(&list_len.to_be_bytes());
        b.extend_from_slice(&list);
        Ok(b)
    }

    /// Decode a body: exactly the layout, nothing after the list.
    ///
    /// # Errors
    ///
    /// [`BodyError::Malformed`] for another version, a length past the end or bytes after the list;
    /// [`BodyError::TooLarge`] for a certificate over [`MAX_DEVICE_CERT_LEN`] or a list over
    /// [`MAX_RELAY_REVOCATIONS_LEN`]; [`BodyError::Cert`] / [`BodyError::Revocations`] when a part does not
    /// decode.
    pub fn from_bytes(body: &[u8]) -> Result<Self, BodyError> {
        let (&version, rest) = body.split_first().ok_or(BodyError::Malformed)?;
        if version != RELAY_REVOCATIONS_BODY_VERSION {
            return Err(BodyError::Malformed);
        }
        let (cert_len, rest) = split_len::<2>(rest)?;
        if cert_len > MAX_DEVICE_CERT_LEN {
            return Err(BodyError::TooLarge);
        }
        let (cert, rest) = split_part(rest, cert_len)?;
        let (list_len, rest) = split_len::<4>(rest)?;
        if list_len > MAX_RELAY_REVOCATIONS_LEN {
            return Err(BodyError::TooLarge);
        }
        let (list, rest) = split_part(rest, list_len)?;
        if !rest.is_empty() {
            return Err(BodyError::Malformed);
        }
        Ok(Self {
            delegate: DeviceCert::from_bytes(cert).map_err(BodyError::Cert)?,
            list: RevocationList::from_bytes(list).map_err(BodyError::Revocations)?,
        })
    }
}

/// `len` as the length field `T`, or [`BodyError::TooLarge`] when it is over `cap` or does not fit `T`
fn capped_len<T: TryFrom<usize>>(len: usize, cap: usize) -> Result<T, BodyError> {
    if len > cap {
        return Err(BodyError::TooLarge);
    }
    T::try_from(len).map_err(|_| BodyError::TooLarge)
}

/// A big-endian length of `N` bytes (2 or 4) at the start of `bytes`
fn split_len<const N: usize>(bytes: &[u8]) -> Result<(usize, &[u8]), BodyError> {
    let (head, rest) = split_part(bytes, N)?;
    let len = head.iter().fold(0usize, |n, b| (n << 8) | usize::from(*b));
    Ok((len, rest))
}

/// The first `n` bytes and the rest, or [`BodyError::Malformed`] when fewer are left
fn split_part(bytes: &[u8], n: usize) -> Result<(&[u8], &[u8]), BodyError> {
    if bytes.len() < n {
        return Err(BodyError::Malformed);
    }
    Ok(bytes.split_at(n))
}

/// Why [`DelegatedListResponse::from_bytes`] refused a body, or [`DelegatedListResponse::to_bytes`] a part.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BodyError {
    /// Another layout version, a length past the end, or bytes after the list.
    Malformed,
    /// The certificate is longer than [`MAX_DEVICE_CERT_LEN`] or the list longer than
    /// [`MAX_RELAY_REVOCATIONS_LEN`].
    TooLarge,
    /// The certificate part is not a certificate.
    Cert(DeviceCertError),
    /// The list part is not a revocation list.
    Revocations(RevocationListError),
}

impl core::fmt::Display for BodyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Malformed => f.write_str("not a revocation body"),
            Self::TooLarge => f.write_str("revocation body part over its cap"),
            Self::Cert(e) => write!(f, "certificate: {e:?}"),
            Self::Revocations(e) => write!(f, "revocation list: {e}"),
        }
    }
}

impl std::error::Error for BodyError {}

//! `GET /relay-cert` and `GET /relay-revocations`: the relay's certificate, and the operator's revocation list
//! with the certificate of the delegate key that signed it, over HTTP/1.1.
//!
//! A reader of exactly these responses, not a general HTTP client: the request is fixed, the response head is
//! capped at [`MAX_RELAY_RESPONSE_HEAD`] bytes and the body at the resource's own cap (the longest certificate
//! there is, [`MAX_DEVICE_CERT_LEN`]; the longest delegate certificate and list this reader takes,
//! [`MAX_RELAY_DELEGATED_REVOCATIONS_LEN`]), both before they are read; the body is framed by `Content-Length`
//! or, without one, by the end of the stream (the request asks the server to close); a transfer coding
//! (`chunked`) is refused rather than decoded. Both resources go through the one bounded reader ([`get`]);
//! only the path, the cap and the decoder differ.
//!
//! # The revocation body
//!
//! The body of `GET /relay-revocations` (the delegate certificate and the list, [`DelegatedListResponse`]),
//! its version and its caps live in [`crate::relay_revocations_body`], which needs no feature, so a relay
//! serves the same layout without this client; `anonymous_relay_client` re-exports them.

use std::io::{Read, Write};

use alice_identity::cert_revocation::RevocationListError;
use alice_identity::device_cert::{DeviceCert, DeviceCertError, MAX_DEVICE_CERT_LEN};

use crate::relay_revocations_body::{
    BodyError, DelegatedListResponse, MAX_RELAY_DELEGATED_REVOCATIONS_LEN,
};

/// The longest response head [`fetch_relay_cert`] and [`fetch_relay_revocations`] read.
pub const MAX_RELAY_RESPONSE_HEAD: usize = 8 * 1024;

/// Why [`fetch_relay_cert`] or [`fetch_relay_revocations`] returned nothing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FetchError {
    /// The host name is empty or holds a character that would break the request.
    BadHost,
    /// The stream failed or ended early.
    Io(std::io::ErrorKind),
    /// The response is not well-formed HTTP/1.1, or uses a transfer coding this reader does not take.
    Malformed,
    /// The status is not 200 (for the revocation list, 404 means the relay serves none).
    Status(u16),
    /// The response head or body is longer than its cap.
    TooLarge,
    /// The body is not a certificate.
    Cert(DeviceCertError),
    /// The body is not a revocation list.
    Revocations(RevocationListError),
}

impl core::fmt::Display for FetchError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::BadHost => f.write_str("host name not usable in a request"),
            Self::Io(kind) => write!(f, "stream: {kind:?}"),
            Self::Malformed => f.write_str("not a response this reader takes"),
            Self::Status(code) => write!(f, "HTTP status {code}"),
            Self::TooLarge => f.write_str("response over its cap"),
            Self::Cert(e) => write!(f, "certificate: {e:?}"),
            Self::Revocations(e) => write!(f, "revocation list: {e}"),
        }
    }
}

impl std::error::Error for FetchError {}

/// A body that is not one is the same refusal as from the reader: [`BodyError::Malformed`] is
/// [`FetchError::Malformed`], [`BodyError::TooLarge`] is [`FetchError::TooLarge`], and a part that does not
/// decode is [`FetchError::Cert`] / [`FetchError::Revocations`].
impl From<BodyError> for FetchError {
    fn from(e: BodyError) -> Self {
        match e {
            BodyError::Malformed => Self::Malformed,
            BodyError::TooLarge => Self::TooLarge,
            BodyError::Cert(e) => Self::Cert(e),
            BodyError::Revocations(e) => Self::Revocations(e),
        }
    }
}

impl From<std::io::Error> for FetchError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e.kind())
    }
}

/// One `read`, retrying `Interrupted`
fn read_some(stream: &mut impl Read, buf: &mut [u8]) -> Result<usize, FetchError> {
    loop {
        match stream.read(buf) {
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            other => return Ok(other?),
        }
    }
}

/// The status code of an `HTTP/1.x <3 digits> <reason>` status line
fn status_code(line: &str) -> Option<u16> {
    let rest = line
        .strip_prefix("HTTP/1.1 ")
        .or_else(|| line.strip_prefix("HTTP/1.0 "))?;
    let code = rest.get(..3)?;
    if !code.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    // after the code: nothing, or a space and the reason phrase
    if !(rest.len() == 3 || rest.as_bytes().get(3) == Some(&b' ')) {
        return None;
    }
    code.parse().ok()
}

/// Fetch the relay's certificate with `GET /relay-cert` on `stream`.
///
/// `stream` is the connection to the relay's host as the caller made it (plain TCP, or a TLS stream: the
/// choice is the caller's). The certificate is returned as parsed ([`DeviceCert::from_bytes`]), not
/// verified: check it against the operator's key (for example `alice_channel::PinnedPeer::service`, which
/// [`super::RelayConnection::open`] does before it sends anything).
///
/// # Errors
///
/// [`FetchError::BadHost`] before anything is written; [`FetchError::Io`] when the stream fails or ends inside
/// the response; [`FetchError::Malformed`] for a response this reader does not take (status line, header line,
/// `Content-Length`, a `Transfer-Encoding`); [`FetchError::Status`] for a status other than 200;
/// [`FetchError::TooLarge`] when the head is over [`MAX_RELAY_RESPONSE_HEAD`] or the body over
/// [`MAX_DEVICE_CERT_LEN`] (refused from the declared length, before the body is read); [`FetchError::Cert`]
/// when the body is not a certificate.
pub fn fetch_relay_cert<S: Read + Write>(stream: S, host: &str) -> Result<DeviceCert, FetchError> {
    let body = get(stream, host, "/relay-cert", MAX_DEVICE_CERT_LEN)?;
    DeviceCert::from_bytes(&body).map_err(FetchError::Cert)
}

/// Fetch the operator's revocation list and its delegate's certificate with `GET /relay-revocations` on
/// `stream`.
///
/// `stream` is a connection to the relay's host of its own, as for [`fetch_relay_cert`]. Both parts are
/// returned as decoded ([`DelegatedListResponse::from_bytes`]), **not verified**: the caller checks them under
/// the operator's master key, for example with [`super::KeptRevocations::accept`] (the delegate certificate,
/// the list's signature and freshness, and the (epoch, sequence) order, kept on disk so a restart does not
/// lower it), or through [`super::RevocationRule::apply_fetch`], which does that and decides about the relay.
///
/// A relay that serves no list answers 404: that is [`FetchError::Status`]`(404)`.
///
/// # Errors
///
/// As [`fetch_relay_cert`], with the body capped at [`MAX_RELAY_DELEGATED_REVOCATIONS_LEN`] and the errors of
/// [`DelegatedListResponse::from_bytes`] (as [`FetchError`], `From<BodyError>`) when the body is not one.
pub fn fetch_relay_revocations<S: Read + Write>(
    stream: S,
    host: &str,
) -> Result<DelegatedListResponse, FetchError> {
    let body = get(
        stream,
        host,
        "/relay-revocations",
        MAX_RELAY_DELEGATED_REVOCATIONS_LEN,
    )?;
    Ok(DelegatedListResponse::from_bytes(&body)?)
}

/// `GET path` on `stream` and the body of a 200 response, at most `cap` bytes (see the module documentation)
fn get<S: Read + Write>(
    mut stream: S,
    host: &str,
    path: &str,
    cap: usize,
) -> Result<Vec<u8>, FetchError> {
    if !super::connection::usable_host(host) {
        return Err(FetchError::BadHost);
    }
    let request = format!(
        "GET {path} HTTP/1.1\r\nHost: {host}\r\nAccept: application/octet-stream\r\nConnection: close\r\n\r\n"
    );
    stream.write_all(request.as_bytes())?;
    stream.flush()?;

    // the head, up to its cap; bytes read past it are the start of the body
    let mut buf = Vec::new();
    let head_end = loop {
        if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break i + 4;
        }
        if buf.len() >= MAX_RELAY_RESPONSE_HEAD {
            return Err(FetchError::TooLarge);
        }
        let mut chunk = [0u8; 1024];
        let room = (MAX_RELAY_RESPONSE_HEAD - buf.len()).min(chunk.len());
        let n = read_some(&mut stream, &mut chunk[..room])?;
        if n == 0 {
            return Err(FetchError::Io(std::io::ErrorKind::UnexpectedEof));
        }
        buf.extend_from_slice(&chunk[..n]);
    };
    let head = std::str::from_utf8(&buf[..head_end - 4]).map_err(|_| FetchError::Malformed)?;
    let mut lines = head.split("\r\n");
    let code = lines
        .next()
        .and_then(status_code)
        .ok_or(FetchError::Malformed)?;
    let mut length: Option<usize> = None;
    for line in lines {
        let (name, value) = line.split_once(':').ok_or(FetchError::Malformed)?;
        let value = value.trim();
        if name.eq_ignore_ascii_case("content-length") {
            if length.is_some() || value.is_empty() || !value.bytes().all(|b| b.is_ascii_digit()) {
                return Err(FetchError::Malformed);
            }
            // a length past usize is over the cap as well
            length = Some(value.parse().unwrap_or(usize::MAX));
        } else if name.eq_ignore_ascii_case("transfer-encoding") {
            return Err(FetchError::Malformed);
        }
    }
    if code != 200 {
        return Err(FetchError::Status(code));
    }
    if length.is_some_and(|n| n > cap) {
        return Err(FetchError::TooLarge);
    }
    let mut body = buf.split_off(head_end);
    // read up to the declared length, or to the end of the stream, never past the cap + 1 byte
    let want = length.unwrap_or(cap + 1);
    while body.len() < want {
        let mut chunk = [0u8; 1024];
        let room = (want - body.len()).min(chunk.len());
        let n = read_some(&mut stream, &mut chunk[..room])?;
        if n == 0 {
            if length.is_some() {
                return Err(FetchError::Io(std::io::ErrorKind::UnexpectedEof));
            }
            break;
        }
        body.extend_from_slice(&chunk[..n]);
    }
    if length.is_some_and(|n| body.len() != n) {
        // bytes past the declared length came with the head
        return Err(FetchError::Malformed);
    }
    if body.len() > cap {
        // no length, and the stream went on past the cap
        return Err(FetchError::TooLarge);
    }
    Ok(body)
}

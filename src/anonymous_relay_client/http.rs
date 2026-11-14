//! `GET /relay-cert`: the relay's certificate over HTTP/1.1.
//!
//! A reader of exactly this one response, not a general HTTP client: the request is fixed, the response
//! head is capped at [`MAX_RELAY_CERT_RESPONSE_HEAD`] bytes and the body at
//! [`MAX_DEVICE_CERT_LEN`] (the longest certificate there is), both before they are read; the body is
//! framed by `Content-Length` or, without one, by the end of the stream (the request asks the server to
//! close); a transfer coding (`chunked`) is refused rather than decoded.

use std::io::{Read, Write};

use alice_identity::device_cert::{DeviceCert, DeviceCertError, MAX_DEVICE_CERT_LEN};

/// The longest response head [`fetch_relay_cert`] reads.
pub const MAX_RELAY_CERT_RESPONSE_HEAD: usize = 8 * 1024;

/// Why [`fetch_relay_cert`] returned no certificate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FetchCertError {
    /// The host name is empty or holds a character that would break the request.
    BadHost,
    /// The stream failed or ended early.
    Io(std::io::ErrorKind),
    /// The response is not well-formed HTTP/1.1, or uses a transfer coding this reader does not take.
    Malformed,
    /// The status is not 200.
    Status(u16),
    /// The response head or body is longer than its cap.
    TooLarge,
    /// The body is not a certificate.
    Cert(DeviceCertError),
}

impl core::fmt::Display for FetchCertError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::BadHost => f.write_str("host name not usable in a request"),
            Self::Io(kind) => write!(f, "stream: {kind:?}"),
            Self::Malformed => f.write_str("not a response this reader takes"),
            Self::Status(code) => write!(f, "HTTP status {code}"),
            Self::TooLarge => f.write_str("response over its cap"),
            Self::Cert(e) => write!(f, "certificate: {e:?}"),
        }
    }
}

impl std::error::Error for FetchCertError {}

impl From<std::io::Error> for FetchCertError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e.kind())
    }
}

/// One `read`, retrying `Interrupted`
fn read_some(stream: &mut impl Read, buf: &mut [u8]) -> Result<usize, FetchCertError> {
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
/// [`FetchCertError::BadHost`] before anything is written; [`FetchCertError::Io`] when the stream fails or
/// ends inside the response; [`FetchCertError::Malformed`] for a response this reader does not take
/// (status line, header line, `Content-Length`, a `Transfer-Encoding`); [`FetchCertError::Status`] for a
/// status other than 200; [`FetchCertError::TooLarge`] when the head is over
/// [`MAX_RELAY_CERT_RESPONSE_HEAD`] or the body over [`MAX_DEVICE_CERT_LEN`] (refused from the declared
/// length, before the body is read); [`FetchCertError::Cert`] when the body is not a certificate.
pub fn fetch_relay_cert<S: Read + Write>(
    mut stream: S,
    host: &str,
) -> Result<DeviceCert, FetchCertError> {
    if !super::connection::usable_host(host) {
        return Err(FetchCertError::BadHost);
    }
    let request = format!(
        "GET /relay-cert HTTP/1.1\r\nHost: {host}\r\nAccept: application/octet-stream\r\nConnection: close\r\n\r\n"
    );
    stream.write_all(request.as_bytes())?;
    stream.flush()?;

    // the head, up to its cap; bytes read past it are the start of the body
    let mut buf = Vec::new();
    let head_end = loop {
        if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break i + 4;
        }
        if buf.len() >= MAX_RELAY_CERT_RESPONSE_HEAD {
            return Err(FetchCertError::TooLarge);
        }
        let mut chunk = [0u8; 1024];
        let room = (MAX_RELAY_CERT_RESPONSE_HEAD - buf.len()).min(chunk.len());
        let n = read_some(&mut stream, &mut chunk[..room])?;
        if n == 0 {
            return Err(FetchCertError::Io(std::io::ErrorKind::UnexpectedEof));
        }
        buf.extend_from_slice(&chunk[..n]);
    };
    let head = std::str::from_utf8(&buf[..head_end - 4]).map_err(|_| FetchCertError::Malformed)?;
    let mut lines = head.split("\r\n");
    let code = lines
        .next()
        .and_then(status_code)
        .ok_or(FetchCertError::Malformed)?;
    let mut length: Option<usize> = None;
    for line in lines {
        let (name, value) = line.split_once(':').ok_or(FetchCertError::Malformed)?;
        let value = value.trim();
        if name.eq_ignore_ascii_case("content-length") {
            if length.is_some() || value.is_empty() || !value.bytes().all(|b| b.is_ascii_digit()) {
                return Err(FetchCertError::Malformed);
            }
            // a length past usize is over the cap as well
            length = Some(value.parse().unwrap_or(usize::MAX));
        } else if name.eq_ignore_ascii_case("transfer-encoding") {
            return Err(FetchCertError::Malformed);
        }
    }
    if code != 200 {
        return Err(FetchCertError::Status(code));
    }
    if length.is_some_and(|n| n > MAX_DEVICE_CERT_LEN) {
        return Err(FetchCertError::TooLarge);
    }
    let mut body = buf.split_off(head_end);
    // read up to the declared length, or to the end of the stream, never past the cap + 1 byte
    let want = length.unwrap_or(MAX_DEVICE_CERT_LEN + 1);
    while body.len() < want {
        let mut chunk = [0u8; 1024];
        let room = (want - body.len()).min(chunk.len());
        let n = read_some(&mut stream, &mut chunk[..room])?;
        if n == 0 {
            if length.is_some() {
                return Err(FetchCertError::Io(std::io::ErrorKind::UnexpectedEof));
            }
            break;
        }
        body.extend_from_slice(&chunk[..n]);
    }
    if length.is_some_and(|n| body.len() != n) {
        // bytes past the declared length came with the head
        return Err(FetchCertError::Malformed);
    }
    if body.len() > MAX_DEVICE_CERT_LEN {
        // no length, and the stream went on past the cap
        return Err(FetchCertError::TooLarge);
    }
    DeviceCert::from_bytes(&body).map_err(FetchCertError::Cert)
}

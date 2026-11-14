//! The blocking connection to the relay: a WebSocket to `/r/<alias>`, the anonymous handshake to the
//! relay's certificate inside it, and the wire frames inside that.

use std::io::{Read, Write};

use alice_channel::noise::{Transport, MAX_MESSAGE};
use alice_channel::{check_cert, connect_messages, ChannelError, MessageIo, PeerPolicy};
use alice_identity::device_cert::DeviceCert;
use alice_websocket::client::{Connection, Received};
use alice_websocket::{MaskSource, WsError};

use crate::anonymous_relay::RoomAlias;
use crate::anonymous_relay_wire::{Frame, WireError};

/// The longest encoded frame a [`RelayConnection`] sends or accepts: one encrypted message of the channel
/// carries one frame ([`alice_channel::MAX_PAYLOAD`] bytes).
pub const MAX_FRAME_BYTES: usize = alice_channel::MAX_PAYLOAD;

/// Why a [`RelayConnection`] could not open, send or receive.
#[derive(Debug)]
pub enum RelayConnectionError {
    /// The host name is empty or holds a character that would break the request line.
    BadHost,
    /// The WebSocket failed (the upgrade, the stream, the framing).
    WebSocket(WsError),
    /// The channel refused the relay's certificate (nothing was sent) or the handshake failed.
    Channel(ChannelError),
    /// The relay sent bytes that are not one frame.
    Wire(WireError),
    /// The encoded frame is longer than [`MAX_FRAME_BYTES`]; nothing was sent.
    TooLarge {
        /// The encoded length.
        len: usize,
    },
    /// The relay sent a text message; the relay hop carries binary messages only.
    TextMessage,
    /// The relay closed the WebSocket.
    Closed,
    /// An earlier send or receive did not finish; the connection is not used again.
    Poisoned,
}

impl core::fmt::Display for RelayConnectionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::BadHost => f.write_str("host name not usable in a request"),
            Self::WebSocket(e) => write!(f, "websocket: {e}"),
            Self::Channel(e) => write!(f, "channel: {e:?}"),
            Self::Wire(e) => write!(f, "frame: {e:?}"),
            Self::TooLarge { len } => write!(f, "frame of {len} bytes over {MAX_FRAME_BYTES}"),
            Self::TextMessage => f.write_str("a text message on the relay hop"),
            Self::Closed => f.write_str("the relay closed the connection"),
            Self::Poisoned => f.write_str("an earlier operation did not finish"),
        }
    }
}

impl std::error::Error for RelayConnectionError {}

/// Whether `host` can stand in a request's `Host` header: not empty, printable ASCII, no space
pub(super) fn usable_host(host: &str) -> bool {
    !host.is_empty() && host.bytes().all(|b| b.is_ascii_graphic())
}

/// The WebSocket as the channel's message transport ([`alice_channel::MessageIo`]): one binary message per
/// handshake message. The channel's error type has no room for a WebSocket error, so the first one is
/// kept here and reported in its place
struct WsIo<'a, S, M> {
    ws: &'a mut Connection<S, M>,
    failed: Option<RelayConnectionError>,
}

impl<S: Read + Write, M: MaskSource> MessageIo for WsIo<'_, S, M> {
    fn send(&mut self, message: &[u8]) -> Result<(), ChannelError> {
        self.ws.send_binary(message).map_err(|e| self.fail(e))
    }

    fn recv(&mut self, max: usize) -> Result<Vec<u8>, ChannelError> {
        match self.ws.recv_message(max) {
            Ok(Received::Binary(bytes)) => Ok(bytes),
            Ok(Received::Text(_)) => Err(self.keep(RelayConnectionError::TextMessage)),
            Ok(_) => Err(self.keep(RelayConnectionError::Closed)),
            Err(e) => Err(self.fail(e)),
        }
    }
}

impl<S, M> WsIo<'_, S, M> {
    fn fail(&mut self, e: WsError) -> ChannelError {
        let kind = match &e {
            WsError::Io { kind, .. } => *kind,
            _ => std::io::ErrorKind::InvalidData,
        };
        self.keep(RelayConnectionError::WebSocket(e));
        ChannelError::Io(kind)
    }

    fn keep(&mut self, e: RelayConnectionError) -> ChannelError {
        let kind = match &e {
            RelayConnectionError::Closed => std::io::ErrorKind::UnexpectedEof,
            _ => std::io::ErrorKind::InvalidData,
        };
        if self.failed.is_none() {
            self.failed = Some(e);
        }
        ChannelError::Io(kind)
    }
}

/// A read that ran out of time before a whole message arrived (the stream's read timeout): retryable,
/// since the WebSocket keeps the bytes it has and the channel has not advanced
fn is_timeout(e: &WsError) -> bool {
    matches!(
        e,
        WsError::Io { kind, .. }
            if matches!(kind, std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut)
    )
}

/// A connection to the anonymous relay over a blocking stream `S`, with the WebSocket mask keys from `M`.
///
/// Three layers: a WebSocket (RFC 6455) to `/r/<room alias in lowercase hex>` on the relay's host, the
/// anonymous handshake of [`alice_channel::anonymous`] inside it (the relay is authenticated by its
/// certificate, this side is not named), and one [`Frame`] per encrypted message inside that. Messages
/// are in order and each one advances the channel's nonce, so a send or receive that fails partway leaves
/// the two sides out of step: the connection is poisoned and every later call returns
/// [`RelayConnectionError::Poisoned`] (open a new one).
///
/// The stream's read timeout is the caller's (set it on `S` before [`Self::open`]); a timeout surfaces as
/// [`RelayConnectionError::WebSocket`] with the stream's error kind and does not poison the connection:
/// [`Self::recv`] can be called again and continues the message it was reading.
pub struct RelayConnection<S, M> {
    ws: Connection<S, M>,
    transport: Transport,
    poisoned: bool,
}

impl<S: Read + Write, M: MaskSource> RelayConnection<S, M> {
    /// Open a connection to the relay for `room` on `stream`: check `cert` against `policy`, upgrade to a
    /// WebSocket at `/r/<room>` on `host`, and run the anonymous handshake to the Noise static key `cert`
    /// names.
    ///
    /// `cert` is the relay's certificate (for example from [`super::fetch_relay_cert`]) and `policy` what
    /// this side trusts (for the relay, `alice_channel::PinnedPeer::service` with the operator's master
    /// key). The certificate is checked before anything is written: a refused one sends no byte, not even
    /// the WebSocket upgrade.
    ///
    /// # Errors
    ///
    /// [`RelayConnectionError::BadHost`] and [`RelayConnectionError::Channel`] (`UnknownPeer` /
    /// `Rejected`) before anything is written; [`RelayConnectionError::WebSocket`] when the upgrade or the
    /// stream fails; [`RelayConnectionError::Channel`] (`Noise`) when the relay does not finish the
    /// handshake for that key; [`RelayConnectionError::TextMessage`] / [`RelayConnectionError::Closed`]
    /// for a text message or a close during the handshake.
    pub fn open(
        stream: S,
        host: &str,
        room: RoomAlias,
        cert: &DeviceCert,
        policy: &impl PeerPolicy,
        mask: M,
    ) -> Result<Self, RelayConnectionError> {
        if !usable_host(host) {
            return Err(RelayConnectionError::BadHost);
        }
        // the certificate first: a relay this side does not trust gets no byte at all
        check_cert(cert, policy).map_err(RelayConnectionError::Channel)?;
        let mut ws = Connection::connect(stream, host, &format!("/r/{room}"), mask)
            .map_err(RelayConnectionError::WebSocket)?;
        let mut io = WsIo {
            ws: &mut ws,
            failed: None,
        };
        let transport = match connect_messages(&mut io, cert, policy) {
            Ok(t) => t,
            Err(e) => return Err(io.failed.take().unwrap_or(RelayConnectionError::Channel(e))),
        };
        Ok(Self {
            ws,
            transport,
            poisoned: false,
        })
    }

    /// Send one frame: encode it, encrypt it as the channel's next message, send that as one binary
    /// WebSocket message.
    ///
    /// # Errors
    ///
    /// [`RelayConnectionError::TooLarge`] when the encoded frame is over [`MAX_FRAME_BYTES`] (nothing is
    /// sent, the connection stays usable); [`RelayConnectionError::WebSocket`] /
    /// [`RelayConnectionError::Channel`] when the write or the encryption fails (the connection is
    /// poisoned); [`RelayConnectionError::Poisoned`].
    pub fn send(&mut self, frame: &Frame) -> Result<(), RelayConnectionError> {
        if self.poisoned {
            return Err(RelayConnectionError::Poisoned);
        }
        let bytes = frame.encode();
        if bytes.len() > MAX_FRAME_BYTES {
            return Err(RelayConnectionError::TooLarge { len: bytes.len() });
        }
        // the nonce advances with the encryption: a message not fully written leaves the relay out of step
        self.poisoned = true;
        let sealed = self
            .transport
            .send(&bytes)
            .map_err(|e| RelayConnectionError::Channel(ChannelError::Noise(e)))?;
        self.ws
            .send_binary(&sealed)
            .map_err(RelayConnectionError::WebSocket)?;
        self.poisoned = false;
        Ok(())
    }

    /// Receive the next frame: one binary WebSocket message, decrypted as the channel's next message,
    /// decoded strictly.
    ///
    /// # Errors
    ///
    /// [`RelayConnectionError::Wire`] when the decrypted bytes are not exactly one frame (the connection
    /// stays usable: the channel is in step); [`RelayConnectionError::Closed`] when the relay closed the
    /// WebSocket; [`RelayConnectionError::WebSocket`] / [`RelayConnectionError::Channel`] /
    /// [`RelayConnectionError::TextMessage`] (the connection is poisoned, except for a read timeout, which
    /// leaves it usable); [`RelayConnectionError::Poisoned`].
    pub fn recv(&mut self) -> Result<Frame, RelayConnectionError> {
        if self.poisoned {
            return Err(RelayConnectionError::Poisoned);
        }
        // nothing has advanced the channel until a whole message is decrypted: a read timeout before that
        // leaves the WebSocket's buffered bytes and the nonce as they were, so the call can be retried
        let sealed = match self.ws.recv_message(MAX_MESSAGE) {
            Ok(Received::Binary(bytes)) => bytes,
            Ok(Received::Text(_)) => {
                self.poisoned = true;
                return Err(RelayConnectionError::TextMessage);
            }
            Ok(_) => {
                self.poisoned = true;
                return Err(RelayConnectionError::Closed);
            }
            Err(e) => {
                if !is_timeout(&e) {
                    self.poisoned = true;
                }
                return Err(RelayConnectionError::WebSocket(e));
            }
        };
        let bytes = match self.transport.recv(&sealed) {
            Ok(bytes) => bytes,
            Err(e) => {
                self.poisoned = true;
                return Err(RelayConnectionError::Channel(ChannelError::Noise(e)));
            }
        };
        Frame::decode(&bytes).map_err(RelayConnectionError::Wire)
    }
}

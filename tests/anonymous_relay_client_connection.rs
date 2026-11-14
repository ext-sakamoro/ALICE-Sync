//! The anonymous relay's blocking client (feature `anonymous-relay-client`) over loopback TCP, against a
//! relay built in the test from the same pieces a relay host uses: the WebSocket server side of
//! `alice-websocket` (the upgrade answer and the frame parser), the responder side of the anonymous
//! handshake (`alice_channel::accept_messages`) and `AnonymousRelayCore`.
//!
//! The relay's certificate is a service certificate (`did:alice-service:relay`) signed by a fixed
//! test-only operator key, and the client trusts it through `PinnedPeer::service`, as a member would.
//!
//! `fetch_relay_cert` is run against canned HTTP responses.
#![cfg(feature = "anonymous-relay-client")]
// a test server runs on its own thread; no result of the library is combined across threads here
#![allow(clippy::disallowed_methods)]

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::Duration;

use alice_channel::noise::{Keypair, MAX_MESSAGE};
use alice_channel::{accept_messages, ChannelError, MessageIo, PinnedPeer};
use alice_identity::artifact::Revocation;
use alice_identity::device_cert::{DeviceCert, MAX_DEVICE_CERT_LEN};
use alice_identity::service_cert::issue_service_cert;
use alice_sign::KeyPair;
use alice_sync::anonymous_relay::{
    AnonymousRelayCore, ConnId, Limits, RoomAlias, RoomToken, SenderTag,
};
use alice_sync::anonymous_relay_client::{
    fetch_relay_cert, Event, FetchCertError, RelayClient, RelayConnection, RelayConnectionError,
};
use alice_sync::anonymous_relay_wire::Frame;
use alice_websocket::{Frame as WsFrame, HandshakeRequest, HandshakeResponse, MaskSource, Opcode};

const ROOM: RoomAlias = RoomAlias::from_raw([0x41; 32]);
const TOKEN: RoomToken = RoomToken::from_raw([0xc1; 32]);
const TAG_A: SenderTag = SenderTag::from_raw([0x0a; 16]);
const TAG_B: SenderTag = SenderTag::from_raw([0x0b; 16]);
const CLIENT: ConnId = ConnId::from_raw(1);
const OTHER: ConnId = ConnId::from_raw(2);
const ISSUED: u64 = 1_800_000_000;
const NOW: u64 = ISSUED + 3_600;
const TIMEOUT: Duration = Duration::from_secs(10);

const LIMITS: Limits = Limits {
    max_payload: 1_024,
    max_log: 16,
    max_pending: 4,
    max_rooms: 2,
    max_members: 4,
    max_publishes_per_window: 100,
    window: 60,
    max_rooms_per_conn: 1,
    max_tags_per_room: 4,
};

/// Test-only operator master key: a fixed seed, so the test never touches a real operator key
fn operator() -> KeyPair {
    KeyPair::from_seed([0x5a; 32])
}

/// The relay's Noise static key (test-only, fixed)
fn relay_static() -> Keypair {
    Keypair::from_secret([0x07; 32])
}

/// The relay's service certificate, signed by the test operator
fn relay_cert() -> DeviceCert {
    let signing = KeyPair::from_seed([0x33; 32]);
    issue_service_cert(
        &operator(),
        "relay",
        signing.public().0,
        relay_static().public(),
        ISSUED,
        ISSUED + 86_400,
    )
    .unwrap()
}

/// A mask source for the tests only (RFC 6455 masks must be unpredictable in production: a counter is
/// enough to exercise the framing here)
struct CountingMask(u8);

impl MaskSource for CountingMask {
    fn fill(&mut self, buf: &mut [u8]) {
        for b in buf {
            self.0 = self.0.wrapping_mul(31).wrapping_add(17);
            *b = self.0;
        }
    }
}

/// The server side of the WebSocket: unmasked frames out, masked frames in
struct ServerWs {
    stream: TcpStream,
    buf: Vec<u8>,
}

impl ServerWs {
    fn read_message(&mut self, max: usize) -> Option<Vec<u8>> {
        loop {
            match WsFrame::parse_limited(&self.buf, max) {
                Ok((frame, used)) => {
                    self.buf.drain(..used);
                    assert!(frame.masked, "a client frame is masked");
                    match frame.opcode {
                        Opcode::Binary => return Some(frame.payload),
                        Opcode::Close => return None,
                        other => panic!("unexpected opcode {other:?}"),
                    }
                }
                Err(alice_websocket::WsError::Incomplete) => {
                    let mut chunk = [0u8; 4096];
                    let n = self.stream.read(&mut chunk).ok()?;
                    if n == 0 {
                        return None;
                    }
                    self.buf.extend_from_slice(&chunk[..n]);
                }
                Err(e) => panic!("{e:?}"),
            }
        }
    }

    fn write_message(&mut self, bytes: &[u8]) {
        self.stream
            .write_all(&WsFrame::binary(bytes.to_vec()).serialize())
            .unwrap();
    }
}

impl MessageIo for ServerWs {
    fn send(&mut self, message: &[u8]) -> Result<(), ChannelError> {
        self.write_message(message);
        Ok(())
    }
    fn recv(&mut self, max: usize) -> Result<Vec<u8>, ChannelError> {
        self.read_message(max).ok_or(ChannelError::Closed)
    }
}

/// Read an HTTP request head (up to the blank line)
fn read_head(stream: &mut TcpStream) -> Vec<u8> {
    let mut head = Vec::new();
    let mut byte = [0u8; 1];
    while !head.ends_with(b"\r\n\r\n") {
        if stream.read(&mut byte).unwrap() == 0 {
            break;
        }
        head.push(byte[0]);
    }
    head
}

/// The relay: one connection, upgraded at `/r/<ROOM>`, the anonymous handshake, then the core. A second
/// member (`OTHER`, tag B) is bound in the core from the start and publishes once when the client binds.
/// Returns the frames the core received from the client, in order
fn relay(listener: &TcpListener) -> Vec<Frame> {
    let (mut stream, _) = listener.accept().unwrap();
    stream.set_read_timeout(Some(TIMEOUT)).unwrap();
    let head = read_head(&mut stream);
    let request = HandshakeRequest::parse(&head).unwrap();
    assert_eq!(request.path, format!("/r/{ROOM}"));
    assert_eq!(request.host, "relay.test");
    stream
        .write_all(
            HandshakeResponse::from_key(&request.key)
                .to_http()
                .as_bytes(),
        )
        .unwrap();
    let mut ws = ServerWs {
        stream,
        buf: Vec::new(),
    };
    let mut transport = accept_messages(&mut ws, &relay_static()).unwrap();
    let mut core = AnonymousRelayCore::new(LIMITS);
    assert!(matches!(
        core.bind(OTHER, ROOM, TOKEN, TAG_B, 0)[0],
        alice_sync::anonymous_relay::Out::Bound { .. }
    ));
    let mut got = Vec::new();
    while let Some(sealed) = ws.read_message(MAX_MESSAGE) {
        let frame = Frame::decode(&transport.recv(&sealed).unwrap()).unwrap();
        got.push(frame.clone());
        let mut answers: Vec<Frame> = Vec::new();
        match frame {
            Frame::Bind { room, token, tag } => {
                answers.extend(
                    core.bind(CLIENT, room, token, tag, 0)
                        .into_iter()
                        .map(|o| o.into_wire().1),
                );
                answers.extend(
                    core.publish(OTHER, ROOM, 0, b"from the other member".to_vec(), 0)
                        .into_iter()
                        .filter_map(|o| {
                            let (to, f) = o.into_wire();
                            (to == CLIENT).then_some(f)
                        }),
                );
            }
            Frame::Publish { room, seq, payload } => {
                answers.extend(
                    core.publish(CLIENT, room, seq, payload, 0)
                        .into_iter()
                        .filter_map(|o| {
                            let (to, f) = o.into_wire();
                            (to == CLIENT).then_some(f)
                        }),
                );
            }
            Frame::LogRange {
                room,
                from_index,
                to_index,
            } => answers.push(
                match core.log_range_within(CLIENT, room, from_index, to_index, 4_096) {
                    Ok(range) => range.into_wire(room),
                    Err(reason) => Frame::Refused { room, reason },
                },
            ),
            Frame::Close { room } => core.leave(CLIENT, room),
            other => panic!("not a client frame {other:?}"),
        }
        for a in answers {
            ws.write_message(&transport.send(&a.encode()).unwrap());
        }
    }
    got
}

fn policy(master: &KeyPair) -> PinnedPeer<'static> {
    PinnedPeer::service("relay", master.public(), NOW, Revocation::NoListAvailable).unwrap()
}

fn connect(listener: &TcpListener) -> TcpStream {
    let stream = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    stream.set_read_timeout(Some(TIMEOUT)).unwrap();
    stream
}

#[test]
fn a_member_binds_receives_publishes_and_reads_the_log_through_the_relay() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let stream = connect(&listener);
    let server = std::thread::spawn(move || relay(&listener));

    let mut conn = RelayConnection::open(
        stream,
        "relay.test",
        ROOM,
        &relay_cert(),
        &policy(&operator()),
        CountingMask(1),
    )
    .unwrap();
    let mut client = RelayClient::new(ROOM, TOKEN, TAG_A, 0, LIMITS.max_payload);

    conn.send(&client.bind()).unwrap();
    assert_eq!(
        client.on_frame(conn.recv().unwrap()).unwrap(),
        Event::Bound { replay: Vec::new() }
    );
    assert_eq!(
        client.on_frame(conn.recv().unwrap()).unwrap(),
        Event::Forward {
            from: TAG_B,
            seq: 0,
            payload: b"from the other member".to_vec(),
        }
    );
    let publish = client.publish(b"from the client".to_vec()).unwrap();
    conn.send(&publish).unwrap();
    conn.send(&client.read_log(0)).unwrap();
    let Event::LogPage { items, next, .. } = client.on_frame(conn.recv().unwrap()).unwrap() else {
        panic!("not a log page");
    };
    let logged: Vec<(SenderTag, u64, Vec<u8>)> = items
        .into_iter()
        .map(|i| (i.from, i.seq, i.payload))
        .collect();
    assert_eq!(
        logged,
        vec![
            (TAG_B, 0, b"from the other member".to_vec()),
            (TAG_A, 0, b"from the client".to_vec()),
        ]
    );
    assert!(next.is_some());
    assert_eq!(
        client.unacked().count(),
        0,
        "the log holds the client's payload"
    );
    conn.send(&client.leave()).unwrap();
    drop(conn);

    let got = server.join().unwrap();
    assert_eq!(
        got,
        vec![client.bind(), publish, client.read_log(0), client.leave()],
        "the relay received the client's frames byte for byte, in order"
    );
}

#[test]
fn a_relay_under_another_operator_key_is_refused_before_any_byte_is_sent() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let stream = connect(&listener);
    let server = std::thread::spawn(move || {
        let (mut s, _) = listener.accept().unwrap();
        s.set_read_timeout(Some(TIMEOUT)).unwrap();
        let mut all = Vec::new();
        s.read_to_end(&mut all).unwrap();
        all.len()
    });
    let other_operator = KeyPair::from_seed([0x5b; 32]);
    let result = RelayConnection::open(
        stream,
        "relay.test",
        ROOM,
        &relay_cert(),
        &policy(&other_operator),
        CountingMask(1),
    );
    assert!(
        matches!(
            result,
            Err(RelayConnectionError::Channel(ChannelError::Rejected(_)))
        ),
        "{:?}",
        result.err()
    );
    // the stream was dropped with the refused open: the relay saw the end of the stream and no byte
    assert_eq!(
        server.join().unwrap(),
        0,
        "not even the WebSocket upgrade was written"
    );
}

#[test]
fn a_host_name_that_would_break_the_request_is_refused() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let result = RelayConnection::open(
        connect(&listener),
        "relay.test\r\nX: y",
        ROOM,
        &relay_cert(),
        &policy(&operator()),
        CountingMask(1),
    );
    assert!(matches!(result, Err(RelayConnectionError::BadHost)));
    assert_eq!(
        fetch_relay_cert(connect(&listener), "").err(),
        Some(FetchCertError::BadHost)
    );
}

/// Serve one canned response to `GET /relay-cert`, then hold the connection open until the client closes
/// it (so a reader that does not stop on its own waits instead of seeing the end of the stream); returns
/// the request head the server read
fn serve(listener: TcpListener, response: Vec<u8>) -> impl FnOnce() -> String {
    let server = std::thread::spawn(move || {
        let (mut s, _) = listener.accept().unwrap();
        s.set_read_timeout(Some(TIMEOUT)).unwrap();
        let head = read_head(&mut s);
        // a reader that stops early may close before the whole response is written
        let _ = s.write_all(&response);
        let mut rest = Vec::new();
        let _ = s.read_to_end(&mut rest);
        String::from_utf8(head).unwrap()
    });
    move || server.join().unwrap()
}

fn response(status: &str, length: Option<usize>, body: &[u8]) -> Vec<u8> {
    let mut r = format!("HTTP/1.1 {status}\r\nContent-Type: application/octet-stream\r\n");
    if let Some(n) = length {
        r.push_str(&format!("Content-Length: {n}\r\n"));
    }
    r.push_str("\r\n");
    let mut bytes = r.into_bytes();
    bytes.extend_from_slice(body);
    bytes
}

#[test]
fn fetch_relay_cert_reads_a_200_body_as_the_certificate() {
    let cert = relay_cert();
    let body = cert.to_bytes();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let stream = connect(&listener);
    let request = serve(listener, response("200 OK", Some(body.len()), &body));
    let got = fetch_relay_cert(stream, "relay.test").unwrap();
    assert_eq!(got.to_bytes(), body);
    let head = request();
    assert!(head.starts_with("GET /relay-cert HTTP/1.1\r\n"), "{head}");
    assert!(head.contains("\r\nHost: relay.test\r\n"), "{head}");

    // without a length: the body runs to the end of the stream
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let stream = connect(&listener);
    let server = std::thread::spawn(move || {
        let (mut s, _) = listener.accept().unwrap();
        read_head(&mut s);
        s.write_all(&response("200 OK", None, &body)).unwrap();
        // the end of the stream ends the body
    });
    assert_eq!(
        fetch_relay_cert(stream, "relay.test").unwrap().to_bytes(),
        cert.to_bytes()
    );
    server.join().unwrap();
}

#[test]
fn fetch_relay_cert_reports_a_status_other_than_200() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let stream = connect(&listener);
    let request = serve(listener, response("404 Not Found", Some(9), b"not found"));
    assert_eq!(
        fetch_relay_cert(stream, "relay.test").err(),
        Some(FetchCertError::Status(404))
    );
    request();
}

/// A client stream whose reads give up after a short time: a reader that does not stop at the cap waits
/// for bytes that never come and fails with the stream's timeout instead of `TooLarge`
fn connect_short(listener: &TcpListener) -> TcpStream {
    let stream = connect(listener);
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    stream
}

#[test]
fn fetch_relay_cert_refuses_a_body_over_the_cap() {
    let over = MAX_DEVICE_CERT_LEN + 1;
    // a declared length over the cap is refused from the head, before any byte of the body: the server
    // declares far more than it sends and keeps the connection open
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let stream = connect_short(&listener);
    let request = serve(
        listener,
        response("200 OK", Some(10 * 1024 * 1024), &[0u8; 16]),
    );
    assert_eq!(
        fetch_relay_cert(stream, "relay.test").err(),
        Some(FetchCertError::TooLarge)
    );
    request();
    // without a length, reading stops one byte past the cap (the server sends more and keeps the
    // connection open, so a reader that does not stop waits)
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let stream = connect_short(&listener);
    let request = serve(listener, response("200 OK", None, &vec![0u8; over + 4_096]));
    assert_eq!(
        fetch_relay_cert(stream, "relay.test").err(),
        Some(FetchCertError::TooLarge)
    );
    request();
}

#[test]
fn fetch_relay_cert_refuses_a_transfer_coding_and_a_malformed_head() {
    for bad in [
        b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n".to_vec(),
        b"HTTP/1.1 2000 OK\r\n\r\n".to_vec(),
        b"SPDY 200 OK\r\n\r\n".to_vec(),
        b"HTTP/1.1 200 OK\r\nContent-Length: 1\r\nContent-Length: 1\r\n\r\nx".to_vec(),
        b"HTTP/1.1 200 OK\r\nno colon\r\n\r\n".to_vec(),
    ] {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let stream = connect(&listener);
        let request = serve(listener, bad.clone());
        assert_eq!(
            fetch_relay_cert(stream, "relay.test").err(),
            Some(FetchCertError::Malformed),
            "{}",
            String::from_utf8_lossy(&bad)
        );
        request();
    }
}

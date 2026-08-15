"""Minimal, secret-safe Asterisk Manager Interface client.

The current implementation intentionally supports only connection,
authentication, Ping, and Logoff. Outbound call origination is added only
after this transport layer has been independently validated.

VoiceProbe AMI is restricted to IPv4 localhost by design.
"""

from __future__ import annotations

import re
import socket
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, Self
from uuid import uuid4

AMI_HOST = "127.0.0.1"
AMI_DEFAULT_PORT = 5038

_HEADER_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


class AMIClientError(RuntimeError):
    """Base error for local Asterisk Manager communication."""


class AMIProtocolError(AMIClientError):
    """Raised when Asterisk sends an invalid or unexpected AMI message."""


class AMIAuthenticationError(AMIClientError):
    """Raised when AMI rejects VoiceProbe credentials."""


class AMIConnectionStateError(AMIClientError):
    """Raised when an AMI operation is invalid for the current state."""


class _SocketLike(Protocol):
    def recv(
        self,
        size: int,
    ) -> bytes: ...

    def sendall(
        self,
        data: bytes,
    ) -> None: ...

    def settimeout(
        self,
        value: float | None,
    ) -> None: ...

    def close(self) -> None: ...


_ConnectionFactory = Callable[
    [tuple[str, int], float],
    _SocketLike,
]


def _open_connection(
    address: tuple[str, int],
    timeout_seconds: float,
) -> _SocketLike:
    return socket.create_connection(
        address,
        timeout=timeout_seconds,
    )


@dataclass(frozen=True, slots=True)
class AsteriskAMIConfig:
    """Connection credentials for the localhost-only VoiceProbe AMI user."""

    username: str
    secret: str = field(
        repr=False,
    )
    host: str = AMI_HOST
    port: int = AMI_DEFAULT_PORT
    timeout_seconds: float = 5.0

    def __post_init__(self) -> None:
        if self.host != AMI_HOST:
            raise ValueError("VoiceProbe AMI must use 127.0.0.1.")

        if (
            isinstance(self.port, bool)
            or not isinstance(self.port, int)
            or not 1 <= self.port <= 65535
        ):
            raise ValueError("AMI port must be between 1 and 65535.")

        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(
                self.timeout_seconds,
                (int, float),
            )
            or self.timeout_seconds <= 0
        ):
            raise ValueError("AMI timeout must be greater than zero.")

        _validate_header_value(
            self.username,
            name="AMI username",
        )

        _validate_header_value(
            self.secret,
            name="AMI secret",
        )


@dataclass(frozen=True, slots=True)
class AMIMessage:
    """One parsed AMI response or event."""

    headers: tuple[
        tuple[str, str],
        ...,
    ]

    def get(
        self,
        name: str,
    ) -> str | None:
        """Return the last matching header using case-insensitive lookup."""
        normalized = name.casefold()

        for key, value in reversed(self.headers):
            if key.casefold() == normalized:
                return value

        return None

    @property
    def response(self) -> str | None:
        return self.get("Response")

    @property
    def action_id(self) -> str | None:
        return self.get("ActionID")


def parse_ami_message(
    payload: bytes,
) -> AMIMessage:
    """Parse one AMI header block without a trailing blank line."""
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise AMIProtocolError("AMI message is not valid UTF-8.") from error

    headers: list[tuple[str, str]] = []

    for raw_line in text.split("\r\n"):
        if not raw_line:
            continue

        name, separator, value = raw_line.partition(":")

        if not separator:
            raise AMIProtocolError("AMI message contains a malformed header.")

        header_name = name.strip()
        header_value = value.strip()

        if not header_name:
            raise AMIProtocolError("AMI message contains an empty header name.")

        headers.append(
            (
                header_name,
                header_value,
            )
        )

    if not headers:
        raise AMIProtocolError("AMI message contains no headers.")

    return AMIMessage(headers=tuple(headers))


def _validate_header_value(
    value: str,
    *,
    name: str,
) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be text.")

    if not value.strip():
        raise ValueError(f"{name} cannot be blank.")

    if "\r" in value or "\n" in value:
        raise ValueError(f"{name} cannot contain line breaks.")


def _encode_action(
    headers: tuple[
        tuple[str, str],
        ...,
    ],
) -> bytes:
    lines: list[str] = []

    for name, value in headers:
        if not _HEADER_NAME_PATTERN.fullmatch(name):
            raise ValueError(f"Invalid AMI header name {name!r}.")

        _validate_header_value(
            value,
            name=f"AMI header {name}",
        )

        lines.append(f"{name}: {value}\r\n")

    return ("".join(lines) + "\r\n").encode("utf-8")


class _BufferedAMIStream:
    """Buffered AMI framing over an already-connected socket."""

    def __init__(
        self,
        connection: _SocketLike,
    ) -> None:
        self._connection = connection
        self._buffer = bytearray()

    def read_banner(self) -> str:
        """Read the single-line Asterisk Manager banner."""
        raw = self._read_until(b"\r\n")

        try:
            banner = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise AMIProtocolError("AMI banner is not valid UTF-8.") from error

        if not banner.startswith("Asterisk Call Manager/"):
            raise AMIProtocolError("Unexpected Asterisk Manager banner.")

        return banner

    def read_message(self) -> AMIMessage:
        """Read one CRLF-delimited AMI message."""
        raw = self._read_until(b"\r\n\r\n")

        return parse_ami_message(raw)

    def _read_until(
        self,
        delimiter: bytes,
    ) -> bytes:
        while True:
            index = self._buffer.find(delimiter)

            if index >= 0:
                result = bytes(self._buffer[:index])

                del self._buffer[: index + len(delimiter)]

                return result

            try:
                chunk = self._connection.recv(4096)
            except TimeoutError as error:
                raise AMIProtocolError("AMI connection timed out.") from error
            except OSError as error:
                raise AMIProtocolError("AMI socket read failed.") from error

            if not chunk:
                raise AMIProtocolError("AMI disconnected before completing a message.")

            self._buffer.extend(chunk)


class AsteriskAMIClient:
    """Small serial AMI client for localhost Asterisk control."""

    def __init__(
        self,
        config: AsteriskAMIConfig,
        *,
        connection_factory: _ConnectionFactory = _open_connection,
    ) -> None:
        self.config = config
        self._connection_factory = connection_factory
        self._connection: _SocketLike | None = None
        self._stream: _BufferedAMIStream | None = None
        self._authenticated = False
        self._banner: str | None = None

    @property
    def connected(self) -> bool:
        return self._connection is not None

    @property
    def authenticated(self) -> bool:
        return self._authenticated

    @property
    def banner(self) -> str | None:
        return self._banner

    def connect(self) -> str:
        """Connect to localhost AMI and verify the manager banner."""
        if self.connected:
            raise AMIConnectionStateError("AMI client is already connected.")

        try:
            connection = self._connection_factory(
                (
                    self.config.host,
                    self.config.port,
                ),
                float(self.config.timeout_seconds),
            )
        except OSError as error:
            raise AMIClientError("Unable to connect to local Asterisk AMI.") from error

        connection.settimeout(float(self.config.timeout_seconds))

        stream = _BufferedAMIStream(connection)

        try:
            banner = stream.read_banner()
        except Exception:
            connection.close()
            raise

        self._connection = connection
        self._stream = stream
        self._banner = banner

        return banner

    def login(self) -> None:
        """Authenticate as the restricted VoiceProbe AMI user."""
        if not self.connected:
            raise AMIConnectionStateError("AMI client must connect before login.")

        if self.authenticated:
            raise AMIConnectionStateError("AMI client is already authenticated.")

        response = self._request(
            (
                (
                    "Action",
                    "Login",
                ),
                (
                    "Username",
                    self.config.username,
                ),
                (
                    "Secret",
                    self.config.secret,
                ),
                (
                    "Events",
                    "off",
                ),
            )
        )

        if response.response != "Success":
            raise AMIAuthenticationError("Asterisk AMI authentication failed.")

        self._authenticated = True

    def ping(self) -> AMIMessage:
        """Send a no-side-effect AMI Ping."""
        self._require_authenticated()

        response = self._request(
            (
                (
                    "Action",
                    "Ping",
                ),
            )
        )

        if response.response != "Success" or response.get("Ping") != "Pong":
            raise AMIProtocolError("Asterisk AMI returned an invalid Ping response.")

        return response

    def logoff(self) -> None:
        """Log off when authenticated, then close the socket."""
        if not self.connected:
            return

        try:
            if self.authenticated:
                response = self._request(
                    (
                        (
                            "Action",
                            "Logoff",
                        ),
                    )
                )

                if response.response not in (
                    "Goodbye",
                    "Success",
                ):
                    raise AMIProtocolError(
                        "Asterisk AMI returned an invalid Logoff response."
                    )
        finally:
            self.close()

    def close(self) -> None:
        """Close the underlying connection without exposing credentials."""
        connection = self._connection

        self._connection = None
        self._stream = None
        self._authenticated = False
        self._banner = None

        if connection is not None:
            connection.close()

    def _request(
        self,
        headers: tuple[
            tuple[str, str],
            ...,
        ],
    ) -> AMIMessage:
        connection = self._require_connection()
        stream = self._require_stream()

        action_id = f"voiceprobe-{uuid4().hex}"

        payload = _encode_action(
            headers
            + (
                (
                    "ActionID",
                    action_id,
                ),
            )
        )

        try:
            connection.sendall(payload)
        except OSError as error:
            raise AMIProtocolError("AMI socket write failed.") from error

        # VoiceProbe currently sends serial requests. Ignore unrelated
        # asynchronous messages but require the matching response before
        # proceeding.
        for _ in range(100):
            message = stream.read_message()

            if message.response is None:
                continue

            message_action_id = message.action_id

            if message_action_id not in (
                None,
                action_id,
            ):
                continue

            return message

        raise AMIProtocolError("AMI response correlation limit exceeded.")

    def _require_connection(
        self,
    ) -> _SocketLike:
        if self._connection is None:
            raise AMIConnectionStateError("AMI client is not connected.")

        return self._connection

    def _require_stream(
        self,
    ) -> _BufferedAMIStream:
        if self._stream is None:
            raise AMIConnectionStateError("AMI client is not connected.")

        return self._stream

    def _require_authenticated(
        self,
    ) -> None:
        if not self.authenticated:
            raise AMIConnectionStateError("AMI client must authenticate first.")

    def __enter__(
        self,
    ) -> Self:
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        if self.authenticated:
            try:
                self.logoff()
            except AMIClientError:
                self.close()
        else:
            self.close()

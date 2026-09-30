"""Reply audio: independent bounded token tap, never an agent cancellation owner.

Wire v1: HTTP/1.1 chunked audio/pcm, mono signed 16-bit little endian at
24,000 Hz. No WAV headers, sentence markers, or application framing. HTTP
chunks may split samples; clients carry odd bytes forward. A zero chunk means
success; failures abort the connection (never a successful truncated reply).
The configured relay's PCM output must follow the OpenAI 24 kHz PCM contract.
"""
from __future__ import annotations

import json
import queue
import re
import select
import socket
import threading
import time
from collections import OrderedDict
from urllib.parse import parse_qs
from urllib.request import Request, ProxyHandler, HTTPHandler, HTTPSHandler, build_opener
from http.client import HTTPConnection

from api import tts_sanitize, voice

_LOCK = threading.Lock()
_REPLIES = OrderedDict()
_MAX_REPLIES = 128
_MAX_TEXT = 128_000
_TTL = 120
# A fenced block is held whole so sanitize() sees it once; past this size it
# is cut like prose (each piece then announces the code it drops).
_MAX_FENCE_HOLD = 20_000
_FENCE_RE = re.compile(r"[ \t>]*(`{3,}|~{3,})")


class Reply:
    def __init__(self):
        self.tokens = queue.Queue()
        self.chars = 0
        self.created = time.monotonic()
        self.claimed = False
        self.ended = False
        self.cancel = threading.Event()


def _reply_locked(stream_id):
    now = time.monotonic()
    for key, rec in list(_REPLIES.items()):
        if now - rec.created > _TTL:
            rec.cancel.set()
            del _REPLIES[key]
    if stream_id not in _REPLIES:
        while len(_REPLIES) >= _MAX_REPLIES:
            _, rec = _REPLIES.popitem(last=False)
            rec.cancel.set()
        _REPLIES[stream_id] = Reply()
    return _REPLIES[stream_id]


def note_text(stream_id, text):
    if not text:
        return
    with _LOCK:
        rec = _reply_locked(stream_id)
        if rec.ended or rec.cancel.is_set():
            return
        rec.chars += len(text)
        if rec.chars > _MAX_TEXT:
            rec.cancel.set()  # audio fails closed; agent and SSE continue
            return
        rec.tokens.put_nowait(str(text))


def note_end(stream_id):
    with _LOCK:
        rec = _reply_locked(stream_id)
        if not rec.ended:
            rec.ended = True
            rec.tokens.put_nowait(None)


class SentenceChunker:
    """Reuse voice's sentence boundaries, with Hark's first-clause hold rule."""
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.pending = ""
        self.held = None
        self.first = True
        self.deadline = None
        self.reopen = ""  # fence carried past a block too long to hold

    def _emit(self, part):
        if self.held is not None:
            first, self.held = self.held, None
            if len(part) >= 60 and len(first) + 1 + len(part) <= 5000:
                return [first + " " + part]
            return [first, part]
        if self.first:
            self.first = False
            if len(part) < 60:
                self.held = part
                return []
        return [part]

    def _fences(self):
        if "```" not in self.pending and "~~~" not in self.pending:
            return []
        return tts_sanitize.fenced_spans(self.pending)

    def _sentence_end(self, fences):
        # A sentence end inside a fenced block (every newline of one) is not
        # a cut: sanitize() must see the whole block to speak or announce it.
        # A closed block ends in a backtick or tilde, never a sentence end, so
        # a cut at a span's end is inside a block still streaming.
        for match in voice._SENTENCE_END_RE.finditer(self.pending):
            if not any(start < match.end() <= end for start, end in fences):
                return match.end()
        return 0

    def _outside_fences(self, cut, fences):
        """Move a forced cut off a fenced block: before it, else after it. A
        block still streaming is held (it is dropped from speech anyway) up to
        a bound, past which it is dispatched whole and reopened for the rest,
        so no later piece reads code aloud as prose."""
        for start, end in fences:
            if start < cut <= end:
                if start:
                    return start
                if end < len(self.pending):
                    return end
                if len(self.pending) < _MAX_FENCE_HOLD:
                    return 0
                self.reopen = _FENCE_RE.match(self.pending).group(1) + "\n"
                return len(self.pending)
        return cut

    def append(self, token):
        if self.deadline is None:
            self.deadline = self.clock() + 2.5
        self.pending += token
        out = []
        while True:
            fences = self._fences()
            # Bound each relay request and dispatch long clauses before a full
            # sentence arrives. Split at whitespace where possible.
            cut = self._sentence_end(fences)
            if len(self.pending) >= 300 and (not cut or cut > 300):
                cut = self.pending.rfind(" ", 20, 301)
                if cut < 20:
                    cut = 300
                cut = self._outside_fences(cut, fences)
            if not cut:
                break
            part, self.pending = self.pending[:cut].strip(), self.pending[cut:]
            self.pending, self.reopen = self.reopen + self.pending, ""
            if part:
                out.extend(self._emit(part))
            # Never reset a held first clause's deadline for incoming tokens.
            if self.held is None:
                self.deadline = self.clock() + 2.5 if self.pending.strip() else None
        return out

    def flush(self):
        parts = [part for part in (self.held, self.pending.strip()) if part]
        self.held, self.pending, self.deadline = None, "", None
        if parts:
            self.first = False
        return [" ".join(parts)] if parts else []

    def due(self):
        if self.deadline is None or self.clock() < self.deadline:
            return []
        fences = self._fences()
        if not fences or fences[-1][1] < len(self.pending):
            return self.flush()
        # A block may still be streaming: speak what precedes it, keep it.
        start = fences[-1][0]
        parts = [part for part in (self.held, self.pending[:start].strip()) if part]
        self.held, self.pending, self.deadline = None, self.pending[start:], None
        if parts:
            self.first = False
        return [" ".join(parts)] if parts else []


def _relay_settings():
    # Exactly the batch OpenAI path's key precedence and profile fallback.
    import os
    from api import routes
    from api.config import get_config
    key = os.getenv("VOICE_TOOLS_OPENAI_KEY", "").strip() or os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        from api.onboarding import _load_env_file
        from api.profiles import get_active_hermes_home
        env = _load_env_file(get_active_hermes_home() / ".env")
        key = env.get("VOICE_TOOLS_OPENAI_KEY", "") or env.get("OPENAI_API_KEY", "")
    if not key:
        raise ValueError("TTS key is not configured")
    cfg = ((get_config() or {}).get("tts") or {}).get("openai") or {}
    base = routes._normalized_openai_tts_base_url(cfg.get("base_url") or "https://api.openai.com/v1")
    return base, key, cfg.get("model") or "gpt-4o-mini-tts", cfg.get("voice") or "alloy"


def _abort_response(resp):
    # HTTPResponse.close alone can wait on a blocked read's BufferedReader
    # lock. Shut down its socket first to wake that read immediately.
    try:
        sock = getattr(resp, "sock", None)
        if sock is None:
            sock = resp.fp.raw._sock
        sock.shutdown(socket.SHUT_RDWR)
    except (AttributeError, OSError):
        pass
    try:
        resp.close()
    except OSError:
        pass


def _disconnected(handler):
    connection = getattr(handler, "connection", None)
    if connection is None:
        return False
    readable, _, _ = select.select([connection], [], [], 0)
    if not readable:
        return False
    # Any readability on this dedicated GET means EOF or unexpected input.
    # SSL sockets cannot MSG_PEEK; no request pipelining on the audio stream.
    return True


def handle(handler, parsed):
    from api import routes
    from api.config import peek_stream
    from api.helpers import bad
    if handler.command != "GET":
        return bad(handler, "GET required", 405)
    stream_id = parse_qs(parsed.query).get("stream_id", [""])[0]
    if not stream_id or len(stream_id) > 128:
        return bad(handler, "stream_id required", 400)
    # Cookie auth already ran in the GET router, exactly like chat SSE.
    if not routes._stream_id_visible_to_request_profile(handler, stream_id):
        return True
    with _LOCK:
        if stream_id not in _REPLIES and peek_stream(stream_id) is None:
            missing = True
        else:
            missing = False
            rec = _reply_locked(stream_id)
            claimed = rec.claimed
            if not claimed:
                rec.claimed = True
    if missing:
        return bad(handler, "stream not found", 404)
    if claimed:
        return bad(handler, "audio stream already opened", 409)
    try:
        base, key, model, relay_voice = _relay_settings()
    except (ValueError, TypeError, AttributeError):
        rec.cancel.set()
        return bad(handler, "streaming TTS is not configured", 503)

    audio = queue.Queue(maxsize=16)  # <= 128 KiB ahead of network consumer
    active = [None]
    active_lock = threading.Lock()

    def send(value):
        while not rec.cancel.is_set():
            try:
                audio.put(value, timeout=0.1)
                return
            except queue.Full:
                pass

    def opener():
        # Register the connection before urlopen waits for response headers.
        # The response thread can then abort synthesis even before a
        # HTTPResponse exists. Once returned, active is replaced by that body.
        class CancelableConnection:
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                with active_lock:
                    active[0] = self
                if rec.cancel.is_set():
                    raise OSError("audio cancelled")

            def connect(self):
                super().connect()
                if rec.cancel.is_set():
                    _abort_response(self)
                    raise OSError("audio cancelled")

        class HTTPSConnection(CancelableConnection, routes._PinnedHTTPSConnection):
            pass

        class LocalHTTPConnection(CancelableConnection, HTTPConnection):
            pass

        class SecureHandler(HTTPSHandler):
            def https_open(self, req):
                return self.do_open(HTTPSConnection, req, context=self._context)

        class LocalHandler(HTTPHandler):
            def http_open(self, req):
                return self.do_open(LocalHTTPConnection, req)

        return build_opener(ProxyHandler({}), routes._NoRedirectTtsHandler(), SecureHandler(), LocalHandler())

    def synthesize(text):
        text = tts_sanitize.sanitize(text).spoken()
        if not text or rec.cancel.is_set():
            return
        req = Request(base + "/audio/speech", data=json.dumps({
            "model": model, "voice": relay_voice, "input": text,
            "stream": True, "response_format": "pcm",
        }).encode(), headers={"Authorization": "Bearer " + key,
            "Content-Type": "application/json", "Accept": "audio/pcm",
            "User-Agent": "HermesWebUI/1.0 (TTS proxy)"})
        resp = routes._tts_open(req, timeout=30, opener_factory=opener)
        with active_lock:
            active[0] = resp
        try:
            if resp.headers.get("Content-Type", "").split(";")[0].lower() != "audio/pcm":
                raise ValueError("expected PCM")
            if rec.cancel.is_set():
                return
            reader = getattr(resp, "read1", resp.read)
            total = 0
            while not rec.cancel.is_set():
                chunk = reader(8192)  # read1 never waits to fill a large buffer
                if not chunk:
                    break
                total += len(chunk)
                if total > routes._TTS_PROXY_MAX_BYTES:
                    raise ValueError("audio exceeds limit")
                send(chunk)
            if not total and not rec.cancel.is_set():
                raise ValueError("empty audio")
        finally:
            with active_lock:
                active[0] = None
            resp.close()

    def pump():
        chunker = SentenceChunker()
        try:
            while not rec.cancel.is_set():
                try:
                    token = rec.tokens.get(timeout=0.1)
                except queue.Empty:
                    for text in chunker.due():
                        synthesize(text)
                    continue
                if token is None:
                    for text in chunker.flush():
                        synthesize(text)
                    send(None)
                    return
                for text in chunker.append(token):
                    synthesize(text)
                for text in chunker.due():
                    synthesize(text)
        except Exception:
            send(RuntimeError("streaming TTS failed"))  # never log upstream secrets

    worker = threading.Thread(target=pump, name="reply-audio", daemon=True)
    started = time.monotonic()
    first_byte = None
    worker.start()
    handler.close_connection = True
    try:
        handler.send_response(200)
        handler.send_header("Content-Type", "audio/pcm")
        handler.send_header("X-Audio-Sample-Rate", "24000")
        handler.send_header("X-Audio-Channels", "1")
        handler.send_header("X-Audio-Format", "s16le")
        handler.send_header("Transfer-Encoding", "chunked")
        handler.send_header("Cache-Control", "no-store")
        handler.send_header("X-Accel-Buffering", "no")
        handler.send_header("Connection", "close")
        handler.end_headers()
        handler.wfile.flush()
        connection = getattr(handler, "connection", None)
        if connection is not None:
            connection.settimeout(10)
        while not rec.cancel.is_set():
            if _disconnected(handler):
                break
            try:
                chunk = audio.get(timeout=0.1)
            except queue.Empty:
                continue
            if chunk is None:
                handler.wfile.write(b"0\r\n\r\n")
                handler.wfile.flush()
                break
            if isinstance(chunk, Exception):
                break  # incomplete chunked response surfaces an error in URLSession
            handler.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
            handler.wfile.flush()
            if first_byte is None:
                first_byte = int((time.monotonic() - started) * 1000)
                with voice._LOCK:
                    turn_id = next((k for k, v in voice._TURNS.items() if v["stream_id"] == stream_id), None)
                voice.mark_stage(turn_id, "tts_first_byte")
    except (OSError, ValueError):
        pass
    finally:
        rec.cancel.set()
        with active_lock:
            resp = active[0]
        if resp is not None:
            _abort_response(resp)
        worker.join(timeout=0.2)  # connect/header waits remain bounded by upstream timeout
        from api.request_logging import emit_request_log
        emit_request_log("[webui] " + json.dumps({"event": "voice_turn_audio", "stream_id": stream_id,
                         "tts_first_byte": first_byte}))
    return True

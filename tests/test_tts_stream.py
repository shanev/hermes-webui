import io
import json
import threading
from urllib.parse import urlsplit

import pytest
from api import tts_stream as tts, voice, routes


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    with tts._LOCK:
        tts._REPLIES.clear()
    monkeypatch.setattr(routes, '_stream_id_visible_to_request_profile', lambda *a, **k: True)
    monkeypatch.setattr(tts, '_relay_settings', lambda: ('https://relay.invalid/v1', 'fake-key', 'fake-model', 'Kore'))
    yield
    with tts._LOCK:
        for rec in tts._REPLIES.values():
            rec.cancel.set()
        tts._REPLIES.clear()


class Handler:
    command = 'GET'
    close_connection = False
    def __init__(self):
        self.wfile = io.BytesIO()
        self.headers = {}
        self.sent_headers = {}
        self.status = None
    def send_response(self, status): self.status = status
    def send_header(self, k, v): self.sent_headers[k] = v
    def end_headers(self): pass


class PCM(io.BytesIO):
    headers = {'Content-Type': 'audio/pcm'}


def decoded(data):
    parts = []
    while data:
        size, data = data.split(b'\r\n', 1)
        n = int(size, 16)
        if not n:
            assert data == b'\r\n'
            break
        parts.append(data[:n])
        assert data[n:n+2] == b'\r\n'
        data = data[n+2:]
    return parts


def test_mocked_llm_stream_chunked_pcm_and_batch_unchanged(monkeypatch):
    requests = []
    def open_audio(req, **kwargs):
        requests.append(json.loads(req.data))
        return PCM(bytes([len(requests)]) * 20_000)
    monkeypatch.setattr(routes, '_tts_open', open_audio)
    voice.note_reply_text('s', 'A complete first sentence that is comfortably longer than sixty characters. ')
    voice.note_reply_text('s', 'This is the next complete sentence. ')
    voice.note_reply_end('s')
    batch = routes._handle_tts
    h = Handler()
    assert tts.handle(h, urlsplit('/api/tts/stream?stream_id=s'))
    assert h.status == 200
    assert h.sent_headers['Transfer-Encoding'] == 'chunked'
    assert 'Content-Length' not in h.sent_headers
    assert h.sent_headers['X-Audio-Sample-Rate'] == '24000'
    chunks = decoded(h.wfile.getvalue())
    assert len(chunks) >= 4
    assert b''.join(chunks) == b'\x01' * 20_000 + b'\x02' * 20_000
    assert all(r['stream'] and r['response_format'] == 'pcm' for r in requests)
    assert routes._handle_tts is batch


def test_tiny_first_merges_with_long_next():
    c = tts.SentenceChunker()
    assert c.append('Hi. ') == []
    second = 'The following sentence has enough characters to merit merging into one speech request. '
    assert c.append(second) == ['Hi. ' + second.strip()]
    assert c.flush() == []


def test_tiny_hold_deadline_not_extended_by_tokens():
    now = [0.0]
    c = tts.SentenceChunker(clock=lambda: now[0])
    assert c.append('Hi. ') == []
    now[0] = 2.0
    assert c.append('more') == []
    assert c.due() == []
    now[0] = 2.5
    assert c.due() == ['Hi. more']
    assert c.due() == []


def test_cjk_and_unpunctuated_clauses():
    c = tts.SentenceChunker()
    assert c.append('你好。') == []
    assert c.flush() == ['你好。']
    c = tts.SentenceChunker()
    result = c.append('word ' * 100)
    assert result and max(map(len, result)) <= 300
    assert c.flush()


def test_audio_disconnect_closes_upstream_not_agent(monkeypatch):
    reading = threading.Event()
    closed = threading.Event()
    continued = threading.Event()
    class Blocking:
        headers = {'Content-Type': 'audio/pcm'}
        def read(self, n):
            reading.set()
            assert closed.wait(2)
            return b''
        def close(self): closed.set()
    monkeypatch.setattr(routes, '_tts_open', lambda *a, **k: Blocking())
    monkeypatch.setattr(tts, '_disconnected', lambda h: reading.wait(0.1))
    voice.note_reply_text('s', 'This complete sentence is more than sixty characters long so synthesis begins immediately. ')
    h = Handler()
    tts.handle(h, urlsplit('/api/tts/stream?stream_id=s'))
    assert closed.is_set()
    # The same agent producer and independent SSE channel continue accepting
    # tokens after audio teardown. No cancel_stream/interrupt call is made.
    from api.config import StreamChannel
    channel = StreamChannel()
    subscriber = channel.subscribe()
    def agent():
        voice.note_reply_text('s', 'The agent continues. ')
        channel.put_nowait(('token', {'text': 'The agent continues. '}))
        continued.set()
    producer = threading.Thread(target=agent)
    producer.start(); producer.join(1)
    assert continued.is_set()
    assert subscriber.get_nowait()[1]['text'] == 'The agent continues. '
    assert b'0\r\n\r\n' not in h.wfile.getvalue()


def test_tokens_before_audio_attach_are_retained_independently():
    voice.note_reply_text('s', 'First delta. ')
    voice.note_reply_text('s', 'Second delta. ')
    voice.note_reply_end('s')
    rec = tts._REPLIES['s']
    assert [rec.tokens.get_nowait() for _ in range(3)] == ['First delta. ', 'Second delta. ', None]


def test_upstream_error_aborts_chunked_response(monkeypatch):
    monkeypatch.setattr(routes, '_tts_open', lambda *a, **k: PCM(b''))
    voice.note_reply_text('s', 'This sentence is long enough to send directly to the speech provider without any hold. ')
    voice.note_reply_end('s')
    h = Handler()
    tts.handle(h, urlsplit('/api/tts/stream?stream_id=s'))
    assert h.close_connection
    assert h.wfile.getvalue() == b''


def test_unknown_and_duplicate_stream(monkeypatch):
    monkeypatch.setattr('api.config.peek_stream', lambda s: None)
    h = Handler()
    tts.handle(h, urlsplit('/api/tts/stream?stream_id=unknown'))
    assert h.status == 404
    voice.note_reply_end('s')
    tts.handle(Handler(), urlsplit('/api/tts/stream?stream_id=s'))
    h = Handler()
    tts.handle(h, urlsplit('/api/tts/stream?stream_id=s'))
    assert h.status == 409


def test_token_cap_cancels_only_audio():
    voice.note_reply_text('s', 'x' * (tts._MAX_TEXT + 1))
    assert tts._REPLIES['s'].cancel.is_set()


def test_large_single_delta_never_exceeds_relay_input_cap():
    c = tts.SentenceChunker()
    chunks = c.append('word ' * 2000 + '. ')
    chunks += c.flush()
    assert chunks and max(map(len, chunks)) <= 5000
    assert ' '.join(chunks).replace(' ', '') == ('word ' * 2000 + '.').replace(' ', '')


def test_profile_guard_runs_before_claim_or_synthesis(monkeypatch):
    voice.note_reply_text('s', 'Only the owning profile may request this speech. ')
    calls = []
    monkeypatch.setattr(routes, '_stream_id_visible_to_request_profile', lambda h, s: calls.append(s) or False)
    assert tts.handle(Handler(), urlsplit('/api/tts/stream?stream_id=s'))
    assert calls == ['s']
    assert not tts._REPLIES['s'].claimed


def test_audio_tap_failure_never_terminates_agent_hook(monkeypatch):
    def broken(*args):
        raise RuntimeError('tap failure')
    monkeypatch.setattr(tts, 'note_text', broken)
    monkeypatch.setattr(tts, 'note_end', broken)
    voice.note_reply_text('s', 'The agent may keep producing text.')
    voice.note_reply_end('s')


def test_cancel_closes_connection_before_response_headers_exist():
    calls = []
    class Sock:
        def shutdown(self, mode): calls.append(('shutdown', mode))
    class Connection:
        sock = Sock()
        def close(self): calls.append(('close', None))
    tts._abort_response(Connection())
    assert [name for name, _ in calls] == ['shutdown', 'close']

"""Voice Phase 1: stage instrumentation and barge-in cancel.

Covers the voice-turn registry (``api/voice.py``), turn_id correlation from
``/api/transcribe`` through the reply stream, TTS and client metrics, the
``POST /api/voice/metrics`` validation rules, and ``POST /api/voice/interrupt``
with and without an active run. Cancellation itself is mocked: the interrupt
endpoint must delegate to the existing Stop path, never reimplement it.
"""
import io
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

import api.config as config
import api.routes as routes
import api.upload as upload
import api.voice as voice


REPO = Path(__file__).resolve().parents[1]
SID = "voicesession1"
OTHER_SID = "voicesession2"
STREAM = "voice-stream-1"
NOW_MS = 1_790_000_000_000  # a valid epoch-milliseconds client timestamp


class _FakeHandler:
    def __init__(self, body: bytes = b"", content_type: str = "application/json"):
        self.command = "POST"
        self.rfile = io.BytesIO(body)
        self.wfile = io.BytesIO()
        self.headers = {"Content-Type": content_type, "Content-Length": str(len(body))}
        self.client_address = ("203.0.113.20", 12345)
        self.close_connection = False
        self.status = None
        self.sent_headers = {}

    def send_response(self, status):
        self.status = status

    def send_header(self, key, value):
        self.sent_headers[key] = value

    def end_headers(self):
        pass

    def payload(self):
        return json.loads(self.wfile.getvalue().decode("utf-8"))


def _json_handler(payload) -> _FakeHandler:
    return _FakeHandler(json.dumps(payload).encode("utf-8"))


@pytest.fixture(autouse=True)
def _isolate_voice_state(monkeypatch):
    """Reset the voice registry and the run registries the tests populate."""

    def _clear():
        voice.reset_for_tests()
        with config.STREAMS_LOCK:
            config.STREAMS.pop(STREAM, None)
            config.STREAM_PARTIAL_TEXT.pop(STREAM, None)
        config.unregister_stream_owner(STREAM)
        with config.ACTIVE_RUNS_LOCK:
            config.ACTIVE_RUNS.pop(STREAM, None)

    _clear()
    monkeypatch.delenv("HERMES_WEBUI_RUNTIME_ADAPTER", raising=False)
    yield
    _clear()


@pytest.fixture(autouse=True)
def log_lines(monkeypatch):
    """Capture voice log lines instead of writing them to the service log."""
    lines = []
    monkeypatch.setattr(voice, "emit_request_log", lines.append)
    return lines


def _events(lines, event):
    parsed = [json.loads(line.split(" ", 1)[1]) for line in lines]
    assert all(line.startswith("[webui] ") for line in lines)
    return [entry for entry in parsed if entry["event"] == event]


def _register_active_run(session_id=SID, stream_id=STREAM, partial="Hello there, this is"):
    with config.STREAMS_LOCK:
        config.STREAMS[stream_id] = object()
        config.STREAM_PARTIAL_TEXT[stream_id] = partial
    config.register_stream_owner(stream_id, session_id)
    config.register_active_run(stream_id, session_id=session_id)


# ── turn_id correlation ─────────────────────────────────────────────────────


def _multipart_body(fields, files, boundary=b"voicephase1"):
    body = b""
    for name, value in fields.items():
        body += b"--" + boundary + b"\r\n"
        body += f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        body += str(value).encode() + b"\r\n"
    for name, (filename, data, content_type) in files.items():
        body += b"--" + boundary + b"\r\n"
        body += (
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode()
        body += data + b"\r\n"
    body += b"--" + boundary + b"--\r\n"
    return body, f"multipart/form-data; boundary={boundary.decode()}"


def _transcribe(monkeypatch, fields=None) -> dict:
    fake_mod = types.ModuleType("tools.transcription_tools")
    fake_mod.transcribe_audio = lambda path: {"success": True, "transcript": "what is the weather"}
    monkeypatch.setitem(sys.modules, "tools.transcription_tools", fake_mod)
    tools_pkg = sys.modules.get("tools")
    if tools_pkg is not None:
        monkeypatch.setattr(tools_pkg, "transcription_tools", fake_mod, raising=False)
    body, content_type = _multipart_body(
        fields or {},
        {"file": ("voice.webm", b"RIFFfakeaudio", "audio/webm")},
    )
    handler = _FakeHandler(body, content_type)
    upload.handle_transcribe(handler)
    assert handler.status == 200
    return handler.payload()


def test_transcribe_returns_a_fresh_turn_id_per_request(monkeypatch, log_lines):
    first = _transcribe(monkeypatch)
    second = _transcribe(monkeypatch)

    assert first["transcript"] == "what is the weather"
    assert voice.is_turn_id(first["turn_id"])
    assert voice.is_turn_id(second["turn_id"])
    assert first["turn_id"] != second["turn_id"]


def test_turn_id_correlates_every_stage_into_one_log_line(monkeypatch, log_lines):
    turn_id = _transcribe(monkeypatch, {"session_id": SID})["turn_id"]

    assert voice.bind_stream(turn_id, SID, STREAM) is True
    voice.note_reply_text(STREAM, "It is sunny")
    voice.note_reply_text(STREAM, " today. Expect")
    assert voice.mark_stage(turn_id, "tts_first_byte") is True
    assert log_lines == []  # nothing is written until the turn closes

    outcome = voice.record_client_stage(SID, turn_id, "playback_start", NOW_MS)

    assert outcome == "recorded"
    assert len(log_lines) == 1
    (line,) = _events(log_lines, "voice_turn")
    assert line["turn_id"] == turn_id
    assert line["session_id"] == SID
    assert line["stream_id"] == STREAM
    assert line["closed_by"] == "playback_start"
    assert line["interrupted"] is False
    assert list(line["stages"]) == sorted(voice.STAGES)
    assert all(isinstance(line["stages"][stage], int) for stage in voice.STAGES)
    ordered = [line["stages"][stage] for stage in voice.STAGES]
    assert ordered == sorted(ordered)
    assert line["elapsed_ms"]["transcribe_start"] == 0
    assert line["client_stages"] == {"playback_start": NOW_MS}
    # Message content never reaches the log.
    assert "sunny" not in log_lines[0]
    assert "weather" not in log_lines[0]


def test_each_turn_logs_once_and_late_events_are_dropped(log_lines):
    turn_id = voice.begin_turn(SID, transcribe_start_ms=voice.now_ms())

    assert voice.record_client_stage(SID, turn_id, "playback_start", NOW_MS) == "recorded"
    assert voice.record_client_stage(SID, turn_id, "playback_start", NOW_MS + 5) == "late"
    assert voice.mark_stage(turn_id, "tts_first_byte") is False

    assert len(_events(log_lines, "voice_turn")) == 1


def test_open_turn_is_flushed_once_by_a_later_turn(monkeypatch, log_lines):
    monkeypatch.setattr(voice, "_TURN_FLUSH_SECONDS", 0.0)
    stale = voice.begin_turn(SID, transcribe_start_ms=voice.now_ms())

    voice.begin_turn(SID)
    voice.begin_turn(SID)

    flushed = [entry for entry in _events(log_lines, "voice_turn") if entry["turn_id"] == stale]
    assert len(flushed) == 1
    assert flushed[0]["closed_by"] == "timeout"
    assert flushed[0]["stages"]["playback_start"] is None


def test_first_sentence_waits_for_a_sentence_boundary():
    turn_id = voice.begin_turn(SID)
    voice.bind_stream(turn_id, SID, STREAM)

    voice.note_reply_text(STREAM, "Version 3.5 is out")
    stages = voice._TURNS[turn_id]["stages"]
    assert "first_token" in stages
    assert "first_sentence" not in stages

    # The terminator and the following whitespace arrive in separate tokens.
    voice.note_reply_text(STREAM, ".")
    assert "first_sentence" not in stages
    voice.note_reply_text(STREAM, " More")
    assert "first_sentence" in stages
    # The hot-path entry is released once the first sentence is known.
    assert STREAM not in voice._AWAITING_REPLY


def test_leading_line_break_is_not_a_sentence_boundary():
    turn_id = voice.begin_turn(SID)
    voice.bind_stream(turn_id, SID, STREAM)

    voice.note_reply_text(STREAM, "\nA")
    voice.note_reply_text(STREAM, "BC")

    stages = voice._TURNS[turn_id]["stages"]
    assert "first_token" in stages
    assert "first_sentence" not in stages


def test_reply_that_ends_without_a_boundary_closes_first_sentence():
    turn_id = voice.begin_turn(SID)
    voice.bind_stream(turn_id, SID, STREAM)
    voice.note_reply_text(STREAM, "Sure thing.")
    stages = voice._TURNS[turn_id]["stages"]
    assert "first_sentence" not in stages

    voice.note_reply_end(STREAM)

    assert "first_sentence" in stages
    assert STREAM not in voice._AWAITING_REPLY


def test_streams_without_a_voice_turn_are_ignored(log_lines):
    voice.note_reply_text("not-a-voice-stream", "Hello. World")
    voice.note_reply_end("not-a-voice-stream")

    assert voice._TURNS == {}
    assert voice._AWAITING_REPLY == {}
    assert log_lines == []


def test_bind_stream_fails_closed_for_unknown_or_foreign_turns():
    turn_id = voice.begin_turn(SID)

    assert voice.bind_stream("vt_" + "0" * 32, SID, STREAM) is False
    assert voice.bind_stream("not-a-turn", SID, STREAM) is False
    assert voice.bind_stream(turn_id, OTHER_SID, STREAM) is False
    assert voice.bind_stream(turn_id, SID, STREAM) is True
    # A turn owns one stream; a replayed chat start cannot rebind it.
    assert voice.bind_stream(turn_id, SID, "another-stream") is False
    assert voice.stream_for_turn(turn_id, SID) == ("bound", STREAM)
    assert voice.stream_for_turn(turn_id, OTHER_SID) == ("mismatch", None)


def test_client_stage_for_another_sessions_turn_is_unknown(log_lines):
    turn_id = voice.begin_turn(SID)

    assert voice.record_client_stage(OTHER_SID, turn_id, "playback_start", NOW_MS) == "unknown_turn"
    assert log_lines == []


def test_chat_start_binds_the_voice_turn_to_the_accepted_stream():
    src = (REPO / "api" / "routes.py").read_text(encoding="utf-8")
    start = src.index("def _handle_chat_start(handler, body, diag=None):")
    body = src[start:src.index("def _resolve_chat_workspace_with_recovery", start)]

    assert 'if status == 200 and body.get("voice_turn_id") and response.get("stream_id"):' in body
    assert '_voice.bind_stream(body.get("voice_turn_id"), s.session_id, response["stream_id"])' in body


# ── Truthful turn lines: unbound starts, silent clips, worker stage ─────────


def test_rejected_voice_chat_start_is_logged_and_stamped_on_the_turn(log_lines):
    turn_id = voice.begin_turn(None)

    voice.record_chat_start(turn_id, SID, status=409, error="session already has an active stream")

    (event,) = _events(log_lines, "voice_chat_start")
    assert event["turn_id"] == turn_id
    assert event["status"] == 409
    assert event["reason"] == "rejected"
    voice.close_turn(turn_id, "timeout")
    (line,) = _events(log_lines, "voice_turn")
    assert line["chat_start"] == {"status": 409, "bound": False}
    assert line["stream_id"] is None


def test_failed_bind_on_an_accepted_start_is_logged(log_lines):
    turn_id = voice.begin_turn(SID)
    assert voice.bind_stream(turn_id, SID, STREAM) is True

    # A replayed start for the same turn cannot rebind it: say so.
    rebound = voice.bind_stream(turn_id, SID, "another-stream")
    voice.record_chat_start(turn_id, SID, status=200, stream_id="another-stream", bound=rebound)

    (event,) = _events(log_lines, "voice_chat_start")
    assert event["reason"] == "already_bound"


def test_bound_start_is_quiet_and_a_timeout_line_names_the_worker_stage(monkeypatch, log_lines):
    turn_id = voice.begin_turn(None)
    assert voice.bind_stream(turn_id, SID, STREAM) is True
    voice.record_chat_start(turn_id, SID, status=200, stream_id=STREAM, bound=True)
    assert _events(log_lines, "voice_chat_start") == []
    voice.note_worker_stage(STREAM, "skill_home_wait")

    monkeypatch.setattr(voice, "_TURN_FLUSH_SECONDS", 0.0)
    voice.begin_turn(None)  # the lazy sweep writes the stale turn's line

    (line,) = [entry for entry in _events(log_lines, "voice_turn") if entry["turn_id"] == turn_id]
    assert line["closed_by"] == "timeout"
    assert line["worker_stage"] == "skill_home_wait"
    assert line["chat_start"] == {"status": 200, "bound": True}
    assert line["opened_at"].endswith("Z")


def test_silent_clip_closes_its_turn_as_no_speech(monkeypatch, log_lines):
    fake_mod = types.ModuleType("tools.transcription_tools")
    fake_mod.transcribe_audio = lambda path: {"success": False, "no_speech": True, "error": "no speech"}
    monkeypatch.setitem(sys.modules, "tools.transcription_tools", fake_mod)
    tools_pkg = sys.modules.get("tools")
    if tools_pkg is not None:
        monkeypatch.setattr(tools_pkg, "transcription_tools", fake_mod, raising=False)
    body, content_type = _multipart_body(
        {"session_id": SID},
        {"file": ("voice.webm", b"RIFFfakeaudio", "audio/webm")},
    )
    handler = _FakeHandler(body, content_type)

    upload.handle_transcribe(handler)

    assert handler.status == 200
    payload = handler.payload()
    assert payload["transcript"] == ""
    (line,) = _events(log_lines, "voice_turn")
    assert line["turn_id"] == payload["turn_id"]
    assert line["closed_by"] == "no_speech"
    # A later sweep never re-reports it as a stuck "timeout" turn.
    monkeypatch.setattr(voice, "_TURN_FLUSH_SECONDS", 0.0)
    voice.begin_turn(None)
    assert len([e for e in _events(log_lines, "voice_turn") if e["turn_id"] == payload["turn_id"]]) == 1


def test_chat_start_reports_unbound_voice_starts():
    src = (REPO / "api" / "routes.py").read_text(encoding="utf-8")
    start = src.index("def _handle_chat_start(handler, body, diag=None):")
    body = src[start:src.index("def _resolve_chat_workspace_with_recovery", start)]

    assert "_voice.record_chat_start(" in body
    assert 'elif body.get("voice_turn_id"):' in body


# ── POST /api/voice/metrics ─────────────────────────────────────────────────


def _valid_metric(turn_id="vt_" + "a" * 32, **overrides):
    event = {"session_id": SID, "turn_id": turn_id, "stage": "playback_start", "ts": NOW_MS}
    event.update(overrides)
    return event


@pytest.mark.parametrize(
    "overrides, error",
    [
        ({"stage": "speculative_llm"}, "unknown stage"),
        ({"stage": ""}, "unknown stage"),
        ({"stage": ["playback_start"]}, "unknown stage"),
        ({"turn_id": "turn-1"}, "invalid turn_id"),
        ({"turn_id": None}, "invalid turn_id"),
        ({"session_id": ""}, "session_id required"),
        ({"session_id": "s" * 129}, "invalid session_id"),
        ({"ts": 1_790_000_000}, "ts must be epoch milliseconds"),  # seconds, not ms
        ({"ts": "1790000000000"}, "ts must be epoch milliseconds"),
        ({"ts": True}, "ts must be epoch milliseconds"),
        ({"ts": float("nan")}, "ts must be epoch milliseconds"),
    ],
)
def test_metric_event_validation_rejects_bad_fields(overrides, error):
    turn_id = overrides.pop("turn_id", "vt_" + "a" * 32)
    event, invalid = voice.validate_metric_event(_valid_metric(turn_id, **overrides))

    assert event is None
    assert invalid == error


def test_voice_mode_note_includes_brevity_cap():
    # The spoken-reply length cap is part of the voice contract: without it the
    # model produces multi-minute audio for casual questions.
    assert "60 words" in voice.VOICE_MODE_NOTE
    assert "100 words" not in voice.VOICE_MODE_NOTE


def test_voice_mode_note_includes_sentence_hygiene():
    # Per-sentence TTS synthesizes one sentence at a time: a very long sentence
    # after a short one leaves the client silent while it synthesizes.
    assert "several short sentences over one long one" in voice.VOICE_MODE_NOTE
    assert "25 words" in voice.VOICE_MODE_NOTE


def test_metric_event_validation_rejects_non_objects():
    assert voice.validate_metric_event(["playback_start"]) == (None, "JSON object required")


def test_metric_event_validation_keeps_only_known_fields():
    turn_id = "vt_" + "a" * 32
    event, invalid = voice.validate_metric_event(
        _valid_metric(turn_id, transcript="must not survive", ts=float(NOW_MS))
    )

    assert invalid is None
    assert event == {"session_id": SID, "turn_id": turn_id, "stage": "playback_start", "ts": NOW_MS}


def _post_metrics(monkeypatch, handler):
    monkeypatch.setattr(routes, "_check_csrf", lambda _handler: True)
    monkeypatch.setattr(routes, "_session_id_visible_to_request_profile", lambda *_a, **_k: True)
    return routes.handle_post(handler, SimpleNamespace(path="/api/voice/metrics", query=""))


def test_metrics_endpoint_records_a_client_stage(monkeypatch, log_lines):
    turn_id = voice.begin_turn(SID, transcribe_start_ms=voice.now_ms())
    handler = _json_handler(_valid_metric(turn_id))

    _post_metrics(monkeypatch, handler)

    assert handler.status == 200
    assert handler.payload() == {
        "ok": True,
        "recorded": True,
        "turn_id": turn_id,
        "stage": "playback_start",
    }
    (line,) = _events(log_lines, "voice_turn")
    assert line["client_stages"] == {"playback_start": NOW_MS}


def test_metrics_endpoint_rejects_unknown_stage(monkeypatch, log_lines):
    turn_id = voice.begin_turn(SID)
    handler = _json_handler(_valid_metric(turn_id, stage="speculative_llm"))

    _post_metrics(monkeypatch, handler)

    assert handler.status == 400
    assert handler.payload() == {"error": "unknown stage"}
    assert voice._TURNS[turn_id]["client"] == {}
    assert log_lines == []


def test_metrics_endpoint_rejects_oversized_payload(monkeypatch, log_lines):
    turn_id = voice.begin_turn(SID)
    oversized = json.dumps(
        _valid_metric(turn_id, padding="x" * (voice.MAX_METRIC_BODY_BYTES + 64))
    ).encode("utf-8")
    handler = _FakeHandler(oversized)

    _post_metrics(monkeypatch, handler)

    assert handler.status == 413
    assert handler.payload() == {"error": "request body too large"}
    # The declared body is never buffered past the endpoint cap.
    assert handler.rfile.tell() == voice.MAX_METRIC_BODY_BYTES
    assert handler.close_connection is True
    assert voice._TURNS[turn_id]["client"] == {}
    assert log_lines == []


def test_metrics_endpoint_rejects_invalid_json(monkeypatch):
    handler = _FakeHandler(b"{not json")

    _post_metrics(monkeypatch, handler)

    assert handler.status == 400
    assert handler.payload() == {"error": "invalid JSON body"}


def test_metrics_endpoint_reports_unknown_turn_without_creating_state(monkeypatch, log_lines):
    handler = _json_handler(_valid_metric("vt_" + "b" * 32))

    _post_metrics(monkeypatch, handler)

    assert handler.status == 404
    assert handler.payload() == {"error": "unknown turn_id"}
    assert voice._TURNS == {}
    assert log_lines == []


# ── POST /api/voice/interrupt ───────────────────────────────────────────────


@pytest.fixture
def stop_path(monkeypatch):
    """Mock the Stop machinery the interrupt endpoint must delegate to."""
    calls = {"cancel": [], "gateway": [], "gateway_blocked": False}

    def fake_cancel(stream_id):
        calls["cancel"].append(stream_id)
        return True

    def fake_gateway(stream_id):
        calls["gateway"].append(stream_id)
        return calls["gateway_blocked"]

    monkeypatch.setattr(routes, "cancel_stream", fake_cancel)
    monkeypatch.setattr(routes, "_gateway_stop_blocked_for_stream", fake_gateway)
    return calls


def _known_session(monkeypatch, active_stream_id=None):
    def fake_get_session(session_id, metadata_only=False):
        if session_id != SID:
            raise KeyError(session_id)
        return SimpleNamespace(session_id=SID, active_stream_id=active_stream_id)

    monkeypatch.setattr(routes, "get_session", fake_get_session)


def _interrupt(body) -> _FakeHandler:
    handler = _FakeHandler()
    routes._handle_voice_interrupt(handler, body)
    return handler


def test_interrupt_cancels_the_active_run_through_the_stop_path(monkeypatch, stop_path, log_lines):
    partial = "Hello there, this is"
    _register_active_run(partial=partial)
    _known_session(monkeypatch, active_stream_id=STREAM)
    turn_id = voice.begin_turn(SID, transcribe_start_ms=voice.now_ms())
    voice.bind_stream(turn_id, SID, STREAM)

    handler = _interrupt({"session_id": SID, "turn_id": turn_id})

    assert handler.status == 200
    assert handler.payload() == {"ok": True, "cancelled": True, "truncated_chars": len(partial)}
    # Gateway stop is resolved first, then the existing cancel path, once each.
    assert stop_path["gateway"] == [STREAM]
    assert stop_path["cancel"] == [STREAM]
    (turn_line,) = _events(log_lines, "voice_turn")
    assert turn_line["closed_by"] == "interrupt"
    assert turn_line["interrupted"] is True
    assert turn_line["truncated_chars"] == len(partial)
    (interrupt_line,) = _events(log_lines, "voice_interrupt")
    assert interrupt_line["turn_id"] == turn_id
    assert interrupt_line["stream_id"] == STREAM
    assert interrupt_line["cancelled"] is True
    assert "Hello there" not in "".join(log_lines)


def test_interrupt_finds_the_run_when_the_session_stream_id_is_unset(monkeypatch, stop_path):
    _register_active_run(partial="abc")
    _known_session(monkeypatch, active_stream_id=None)

    handler = _interrupt({"session_id": SID})

    assert handler.payload() == {"ok": True, "cancelled": True, "truncated_chars": 3}
    assert stop_path["cancel"] == [STREAM]


def test_interrupt_with_no_active_run_is_a_safe_noop(monkeypatch, stop_path):
    _known_session(monkeypatch, active_stream_id=None)

    handler = _interrupt({"session_id": SID, "turn_id": "vt_" + "c" * 32})

    assert handler.status == 200
    assert handler.payload() == {"ok": True, "cancelled": False, "truncated_chars": 0}
    assert stop_path["cancel"] == []
    assert stop_path["gateway"] == []


def test_interrupt_ignores_a_stale_session_stream_id(monkeypatch, stop_path):
    # The sidecar still names a stream, but no live run or channel backs it.
    _known_session(monkeypatch, active_stream_id=STREAM)

    handler = _interrupt({"session_id": SID})

    assert handler.payload() == {"ok": True, "cancelled": False, "truncated_chars": 0}
    assert stop_path["cancel"] == []


def test_interrupt_for_an_unknown_session_does_nothing(monkeypatch, stop_path, log_lines):
    _known_session(monkeypatch)

    handler = _interrupt({"session_id": OTHER_SID})

    assert handler.status == 200
    assert handler.payload() == {"ok": True, "cancelled": False, "truncated_chars": 0}
    assert stop_path["cancel"] == []
    assert voice.consume_interrupt_note(OTHER_SID) is None
    assert log_lines == []


def test_interrupt_never_cancels_another_sessions_run(monkeypatch, stop_path):
    _register_active_run(session_id=OTHER_SID)
    _known_session(monkeypatch, active_stream_id=STREAM)

    handler = _interrupt({"session_id": SID})

    assert handler.payload() == {"ok": True, "cancelled": False, "truncated_chars": 0}
    assert stop_path["cancel"] == []


def test_interrupt_skips_a_run_that_is_already_cancelling(monkeypatch, stop_path):
    _register_active_run()
    config.update_active_run(STREAM, phase="cancelling")
    _known_session(monkeypatch, active_stream_id=None)

    handler = _interrupt({"session_id": SID})

    assert handler.payload() == {"ok": True, "cancelled": False, "truncated_chars": 0}
    assert stop_path["cancel"] == []


def test_interrupt_for_a_finished_turn_leaves_the_newer_run_alone(monkeypatch, stop_path):
    # The interrupted turn's stream is gone; the session has moved on to STREAM.
    turn_id = voice.begin_turn(SID)
    voice.bind_stream(turn_id, SID, "finished-stream")
    _register_active_run()
    _known_session(monkeypatch, active_stream_id=STREAM)

    handler = _interrupt({"session_id": SID, "turn_id": turn_id})

    assert handler.payload() == {"ok": True, "cancelled": False, "truncated_chars": 0}
    assert stop_path["cancel"] == []
    # The spoken reply was still cut off, so the next turn is told.
    assert voice.consume_interrupt_note(SID) == voice.INTERRUPT_NOTE


def test_interrupt_rejects_a_turn_owned_by_another_session(monkeypatch, stop_path):
    _register_active_run()
    _known_session(monkeypatch, active_stream_id=STREAM)
    foreign_turn = voice.begin_turn(OTHER_SID)

    handler = _interrupt({"session_id": SID, "turn_id": foreign_turn})

    assert handler.status == 409
    assert stop_path["cancel"] == []
    assert voice.consume_interrupt_note(SID) is None


@pytest.mark.parametrize(
    "body, error",
    [
        ({}, "session_id required"),
        ({"session_id": "../etc/passwd"}, "invalid session_id"),
        ({"session_id": SID, "turn_id": "turn-1"}, "invalid turn_id"),
        ({"session_id": SID, "turn_id": 7}, "invalid turn_id"),
    ],
)
def test_interrupt_validates_its_body(monkeypatch, stop_path, body, error):
    _known_session(monkeypatch, active_stream_id=STREAM)

    handler = _interrupt(body)

    assert handler.status == 400
    assert handler.payload() == {"error": error}
    assert stop_path["cancel"] == []


def test_interrupt_reports_a_blocked_gateway_stop_without_cancelling(monkeypatch, stop_path, log_lines):
    _register_active_run()
    _known_session(monkeypatch, active_stream_id=STREAM)
    stop_path["gateway_blocked"] = True

    handler = _interrupt({"session_id": SID})

    assert handler.status == 502
    assert handler.payload() == {
        "ok": False,
        "cancelled": False,
        "truncated_chars": 0,
        "error": "Gateway stop failed",
    }
    assert stop_path["cancel"] == []
    assert voice.consume_interrupt_note(SID) is None
    assert log_lines == []


def test_interrupt_uses_the_runtime_adapter_when_enabled(monkeypatch, stop_path):
    import api.runtime_adapter as runtime_adapter

    seen = {}

    class FakeAdapter:
        def __init__(self, *, cancel_delegate):
            seen["delegate"] = cancel_delegate

        def cancel_run(self, stream_id):
            seen["run"] = stream_id
            return SimpleNamespace(accepted=True)

    monkeypatch.setattr(runtime_adapter, "runtime_adapter_enabled", lambda *_a, **_k: True)
    monkeypatch.setattr(runtime_adapter, "LegacyJournalRuntimeAdapter", FakeAdapter)
    _register_active_run(partial="abcd")
    _known_session(monkeypatch, active_stream_id=STREAM)

    handler = _interrupt({"session_id": SID})

    assert handler.payload() == {"ok": True, "cancelled": True, "truncated_chars": 4}
    assert seen == {"delegate": routes.cancel_stream, "run": STREAM}
    assert stop_path["cancel"] == []  # the adapter owns the delegate call


def test_interrupt_route_is_wired_and_shares_the_gateway_stop_step():
    src = (REPO / "api" / "routes.py").read_text(encoding="utf-8")

    assert 'if parsed.path == "/api/voice/interrupt":\n        return _handle_voice_interrupt(handler, body)' in src
    cancel_idx = src.index('if parsed.path == "/api/chat/cancel":')
    cancel_body = src[cancel_idx:src.index('if parsed.path == "/api/chat/stream":', cancel_idx)]
    assert "if _gateway_stop_blocked_for_stream(stream_id):" in cancel_body
    handler_idx = src.index("def _handle_voice_interrupt(handler, body)")
    handler_body = src[handler_idx:src.index("def _starts_token(", handler_idx)]
    assert "_gateway_stop_blocked_for_stream(stream_id)" in handler_body
    assert "cancel_stream(stream_id)" in handler_body
    # The endpoint delegates; it never touches the cancel registries itself.
    for forbidden in ("CANCEL_FLAGS", "AGENT_INSTANCES", ".interrupt(", "STREAMS.pop"):
        assert forbidden not in handler_body


# ── next-turn interrupt note ────────────────────────────────────────────────


def test_interrupt_note_reaches_only_the_next_turn(monkeypatch, stop_path):
    _register_active_run()
    _known_session(monkeypatch, active_stream_id=STREAM)
    _interrupt({"session_id": SID})

    first = voice.prepend_turn_notes(SID, "and tomorrow?", "next-stream")
    second = voice.prepend_turn_notes(SID, "thanks", "later-stream")

    assert "the previous spoken reply was interrupted" in first
    assert first == f"{voice.INTERRUPT_NOTE}\n\nand tomorrow?"
    assert second == "thanks"
    assert voice.prepend_turn_notes(OTHER_SID, "hello", "other-stream") == "hello"


def test_interrupt_note_expires(monkeypatch):
    voice.arm_interrupt_note(SID)
    monkeypatch.setattr(voice, "_NOTE_TTL_SECONDS", -1.0)

    assert voice.consume_interrupt_note(SID) is None
    assert voice.consume_interrupt_note(SID) is None


def test_local_turn_injects_the_note_through_the_notification_prefix():
    src = (REPO / "api" / "streaming.py").read_text(encoding="utf-8")
    # The turn's notes are drained once, for this stream, ahead of the
    # process notifications...
    start = src.index("_process_notifications[:0] = _voice.consume_turn_notes(session_id, stream_id)")
    window = src[start:src.index("_run_conversation_kwargs = _build_run_conversation_kwargs(", start)]

    # ...and share their model-only prefix...
    assert '_agent_msg_text = "\\n\\n".join([*_process_notifications, msg_text]).strip()' in window
    # ...while the persisted user message stays the clean text.
    tail = src[start:src.index("result = agent.run_conversation(**_run_conversation_kwargs)", start)]
    assert "persist_user_message=msg_text," in tail


def test_gateway_turns_prepend_the_note_to_the_request_only():
    src = (REPO / "api" / "gateway_chat.py").read_text(encoding="utf-8")

    assert src.count("_voice.prepend_turn_notes(session_id, str(msg_text or \"\"), stream_id)") == 2
    # Transcript writeback keeps using the clean msg_text.
    assert "active_turn_identity = _active_turn_authority(s, stream_id, msg_text)" in src


# ── voice-mode directive ────────────────────────────────────────────────────


VOICE_SID = "voicemodesession1"


def test_voice_mode_directive_reaches_only_the_turn_it_was_armed_for():
    voice.arm_voice_mode_note(SID, STREAM)

    first = voice.prepend_turn_notes(SID, "what is the weather", STREAM)
    second = voice.prepend_turn_notes(SID, "thanks", "text-stream")

    assert first == f"{voice.VOICE_MODE_NOTE}\n\nwhat is the weather"
    assert "This reply will be spoken aloud." in first
    assert "no markdown formatting, no code blocks, no tables, no URLs" in first
    assert second == "thanks"


def test_text_turns_carry_no_directive():
    assert voice.consume_turn_notes(SID, STREAM) == []
    assert voice.prepend_turn_notes(SID, "hello", STREAM) == "hello"

    # Another session's voice turn changes nothing here.
    voice.arm_voice_mode_note(OTHER_SID, "other-stream")
    assert voice.prepend_turn_notes(SID, "hello", STREAM) == "hello"
    assert voice.consume_turn_notes(OTHER_SID, "other-stream") == [voice.VOICE_MODE_NOTE]


@pytest.mark.parametrize("interrupt_first", [True, False])
def test_interrupt_note_and_directive_both_reach_the_same_turn(interrupt_first):
    if interrupt_first:
        voice.arm_interrupt_note(SID)
        voice.arm_voice_mode_note(SID, STREAM)
    else:
        voice.arm_voice_mode_note(SID, STREAM)
        voice.arm_interrupt_note(SID)

    first = voice.prepend_turn_notes(SID, "and tomorrow?", STREAM)
    second = voice.prepend_turn_notes(SID, "thanks", STREAM)

    # Neither arm clobbers the other; the note about the previous reply leads.
    assert first == f"{voice.INTERRUPT_NOTE}\n\n{voice.VOICE_MODE_NOTE}\n\nand tomorrow?"
    assert second == "thanks"


def test_directive_left_by_a_turn_that_never_ran_is_not_inherited():
    # The voice turn's worker never drained its notes (failed launch, early crash).
    voice.arm_voice_mode_note(SID, "dead-stream")
    voice.arm_interrupt_note(SID)

    # The next turn is typed: it still learns about the barge-in, nothing else.
    assert voice.consume_turn_notes(SID, STREAM) == [voice.INTERRUPT_NOTE]
    assert voice.consume_turn_notes(SID, STREAM) == []

    # The session's next voice turn replaces the leftover.
    voice.arm_voice_mode_note(SID, STREAM)
    assert voice.consume_turn_notes(SID, "dead-stream") == []
    assert voice.consume_turn_notes(SID, STREAM) == [voice.VOICE_MODE_NOTE]


def test_voice_mode_directive_expires(monkeypatch):
    voice.arm_voice_mode_note(SID, STREAM)
    monkeypatch.setattr(voice, "_NOTE_TTL_SECONDS", -1.0)

    assert voice.consume_turn_notes(SID, STREAM) == []
    assert voice._VOICE_MODE_NOTES == {}


def test_arming_needs_a_session_and_a_stream():
    voice.arm_voice_mode_note("", STREAM)
    voice.arm_voice_mode_note(SID, None)

    assert voice._VOICE_MODE_NOTES == {}


def _start_turn(monkeypatch, tmp_path, **start_kwargs):
    """Start a chat turn; return what its worker would send the model."""
    from api.models import Session

    delivered = []

    def fake_worker(*_args, **_kwargs):
        return None

    class WorkerThread:
        def __init__(self, *_args, target=None, args=(), **_kwargs):
            self.target = target
            self.args = args

        def start(self):
            if self.target is not fake_worker:
                return
            # Where the real workers drain the turn's notes: at turn start,
            # for their own session and stream.
            session_id, msg, _model, _workspace, stream_id, _attachments = self.args
            delivered.append(voice.prepend_turn_notes(session_id, msg, stream_id))

    monkeypatch.setattr(Session, "save", lambda self, *a, **k: None)
    monkeypatch.setattr(routes, "set_last_workspace", lambda workspace, **_kw: None)
    monkeypatch.setattr(routes, "create_stream_channel", lambda: object())
    monkeypatch.setattr(routes, "_run_agent_streaming", fake_worker)
    monkeypatch.setattr(routes.threading, "Thread", WorkerThread)

    response = routes._start_chat_stream_for_session(
        Session(session_id=VOICE_SID, title="Untitled"),
        msg="what is the weather",
        attachments=[],
        workspace=str(tmp_path),
        model="test-model",
        model_provider=None,
        external_runtime_owned=False,
        **start_kwargs,
    )
    stream_id = response["stream_id"]
    with config.STREAMS_LOCK:
        config.STREAMS.pop(stream_id, None)
    config.unregister_stream_owner(stream_id)
    with config.ACTIVE_RUNS_LOCK:
        config.ACTIVE_RUNS.pop(stream_id, None)
    return delivered


def test_voice_turn_start_delivers_the_directive_to_its_worker(monkeypatch, tmp_path):
    delivered = _start_turn(monkeypatch, tmp_path, voice_mode=True)

    assert delivered == [f"{voice.VOICE_MODE_NOTE}\n\nwhat is the weather"]
    assert voice._VOICE_MODE_NOTES == {}


def test_text_turn_start_delivers_the_bare_message(monkeypatch, tmp_path):
    delivered = _start_turn(monkeypatch, tmp_path)

    assert delivered == ["what is the weather"]
    assert voice._VOICE_MODE_NOTES == {}


def test_voice_turn_after_a_barge_in_delivers_both_notes(monkeypatch, tmp_path):
    voice.arm_interrupt_note(VOICE_SID)

    delivered = _start_turn(monkeypatch, tmp_path, voice_mode=True)

    assert delivered == [
        f"{voice.INTERRUPT_NOTE}\n\n{voice.VOICE_MODE_NOTE}\n\nwhat is the weather"
    ]
    assert voice.consume_interrupt_note(VOICE_SID) is None


@pytest.mark.parametrize("voice_mode, expected", [(True, {"voice_mode": True}), (False, {})])
def test_start_run_passes_voice_mode_only_for_voice_turns(monkeypatch, voice_mode, expected):
    seen = {}

    def fake_start(_session, **kwargs):
        seen.update(kwargs)
        return {"stream_id": STREAM}

    monkeypatch.setattr(routes, "_start_chat_stream_for_session", fake_start)

    routes._start_run(
        SimpleNamespace(session_id=VOICE_SID),
        msg="what is the weather",
        attachments=[],
        workspace="/tmp",
        model="test-model",
        model_provider=None,
        normalized_model=False,
        source="webui",
        route="/api/chat/start",
        voice_mode=voice_mode,
    )

    assert {key: value for key, value in seen.items() if key == "voice_mode"} == expected


def test_chat_start_requests_voice_mode_only_for_a_voice_turn_id():
    src = (REPO / "api" / "routes.py").read_text(encoding="utf-8")
    start = src.index("def _handle_chat_start(handler, body, diag=None):")
    body = src[start:src.index("def _resolve_chat_workspace_with_recovery", start)]

    # Only a well-formed voice turn id opts in, and never for a regeneration.
    assert (
        'if regeneration is None and _voice.is_turn_id(body.get("voice_turn_id")):\n'
        '            start_run_kwargs["voice_mode"] = True'
    ) in body
    assert body.index('start_run_kwargs["voice_mode"] = True') < body.index("response = _start_run(")
    assert voice.is_turn_id("vt_" + "a" * 32) is True
    assert voice.is_turn_id("") is False
    assert voice.is_turn_id(None) is False

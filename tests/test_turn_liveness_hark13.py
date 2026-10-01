"""hark#13: server-side voice turns must not hang when the SSE consumer dies.

The worker's event path is non-blocking (``put`` -> ``StreamChannel.put_nowait``),
so a dead socket can never raise into the agent loop. What it stranded was a
wait only that consumer could answer, plus no backstop for a wedged turn:
retries 409'd until a server restart. Covered here:

* a consumer that dies mid-turn (SSE write raises) detaches the run; the worker
  completes and persists, and a follow-up chat/start is admitted at once;
* a detached voice turn's clarify falls back instead of waiting out its 3600s
  timeout, and the finished turn still lands in the transcript;
* a genuinely wedged voice turn is force-finalized by the hard deadline
  (cancelled, unregistered, cached Agent retired), a retry succeeds at once,
  and the wedged worker's late result never overwrites the successor;
* a normal turn with a live consumer is unchanged: no detached marker, no
  early finalize;
* the deadline env var is honoured.
"""
import io
import queue
import sys
import threading
import time
import types
from contextlib import contextmanager
from unittest import mock
from urllib.parse import urlparse

import pytest

import api.config as config
import api.models as models
import api.routes as routes
import api.streaming as streaming
import api.turn_liveness as turn_liveness
import api.voice as voice

_MISSING = object()


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    monkeypatch.setattr(models, "SESSION_DIR", session_dir)
    monkeypatch.setattr(models, "SESSION_INDEX_FILE", session_dir / "_index.json")
    monkeypatch.setattr(streaming, "SESSION_DIR", session_dir)
    # Tests drive the reaper explicitly; no background pass may race them.
    monkeypatch.setattr(turn_liveness, "_ensure_reaper", lambda: None)
    monkeypatch.setattr(routes, "_stream_id_visible_to_request_profile", lambda *a, **k: True)
    # An idle dead socket is found by the next keepalive write; don't wait 15s+.
    monkeypatch.setattr(routes, "_SSE_HEARTBEAT_INTERVAL_SECONDS", 0.05)
    monkeypatch.delenv(turn_liveness._VOICE_TURN_DEADLINE_ENV, raising=False)
    _reset()
    yield
    _reset()


def _reset():
    for registry in (
        config.STREAMS, config.CANCEL_FLAGS, config.AGENT_INSTANCES, config.ACTIVE_RUNS,
        config.STREAM_SESSION_OWNERS, config.SESSION_WRITEBACK_OWNERS,
        config.SESSION_AGENT_LOCKS, config.SESSION_AGENT_CACHE,
    ):
        registry.clear()
    turn_liveness.reset_for_tests()
    voice.reset_for_tests()


class FakeSession:
    def __init__(self, session_id, stream_id):
        self.session_id = session_id
        self.title = "Voice turn"
        self.workspace = "/tmp"
        self.worktree_path = None
        self.model = "gpt-5.4"
        self.model_provider = None
        self.profile = None
        self.personality = None
        self.messages = [{"role": "user", "content": "old"}, {"role": "assistant", "content": "old answer"}]
        self.context_messages = list(self.messages)
        self.input_tokens = self.output_tokens = 0
        self.estimated_cost = 0
        self.cache_read_tokens = self.cache_write_tokens = 0
        self.tool_calls = []
        self.gateway_routing = None
        self.gateway_routing_history = []
        self.active_stream_id = stream_id
        self.pending_user_message = None
        self.pending_user_source = None
        self.pending_attachments = []
        self.pending_started_at = None
        self.context_length = self.threshold_tokens = self.last_prompt_tokens = 0
        self.llm_title_generated = True
        self.saves = 0

    def save(self, *args, **kwargs):
        self.saves += 1

    def compact(self):
        return {"session_id": self.session_id, "title": self.title, "workspace": self.workspace,
                "model": self.model, "created_at": 0, "updated_at": 0, "pinned": False,
                "archived": False, "project_id": None, "profile": self.profile}


def _agent_class(run):
    """Fake AIAgent whose run_conversation delegates to ``run(agent, **kwargs)``."""

    class Agent:
        def __init__(self, model=None, provider=None, base_url=None, api_key=None, platform=None,
                     quiet_mode=False, enabled_toolsets=None, fallback_model=None, session_id=None,
                     session_db=None, stream_delta_callback=None, reasoning_callback=None,
                     tool_progress_callback=None, clarify_callback=None):
            self.session_id = session_id
            self.context_compressor = None
            self.session_prompt_tokens = self.session_completion_tokens = 0
            self.session_estimated_cost_usd = 0
            self.session_cache_read_tokens = self.session_cache_write_tokens = 0
            self.reasoning_config = None
            self.ephemeral_system_prompt = None
            self._last_error = None
            self.stream_delta_callback = stream_delta_callback
            self.clarify_callback = clarify_callback
            self.interrupts = []

        def run_conversation(self, **kwargs):
            reply = run(self, **kwargs)
            history = kwargs.get("conversation_history", [])
            return {"messages": history + [
                {"role": "user", "content": kwargs["persist_user_message"]},
                {"role": "assistant", "content": reply},
            ]}

        def interrupt(self, message):
            self.interrupts.append(message)  # a wedged tool call ignores this

    return Agent


@contextmanager
def _agent_runtime(agent_cls, session):
    runtime = types.ModuleType("hermes_cli.runtime_provider")
    runtime.resolve_runtime_provider = mock.Mock(return_value={
        "provider": "openai", "base_url": None, "api_key": "sk-test", "api_mode": "chat_completions",
        "command": None, "args": [], "credential_pool": None,
    })
    cli = types.ModuleType("hermes_cli")
    cli.runtime_provider = runtime
    state = types.ModuleType("hermes_state")
    state.SessionDB = mock.Mock(return_value=None)
    injected = {"hermes_cli": cli, "hermes_cli.runtime_provider": runtime, "hermes_state": state}
    saved = {k: sys.modules.get(k, _MISSING) for k in injected}
    sys.modules.update(injected)
    try:
        with mock.patch.object(streaming, "get_session", return_value=session), \
             mock.patch.object(streaming, "_get_ai_agent", return_value=agent_cls), \
             mock.patch.object(streaming, "resolve_model_provider", return_value=("gpt-5.4", "openai", None)), \
             mock.patch("api.config.get_config", return_value={}), \
             mock.patch("api.config._resolve_cli_toolsets", return_value=[]):
            yield
    finally:
        for k, prev in saved.items():
            if prev is _MISSING:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = prev


def _launch(session, stream_id, *, voice_turn=False):
    """What /api/chat/start does before the worker runs, then start the worker."""
    channel = config.create_stream_channel()
    config.register_stream_owner(stream_id, session.session_id)
    config.register_session_writeback_owner(session.session_id, stream_id)
    with config.STREAMS_LOCK:
        config.STREAMS[stream_id] = channel
    if voice_turn:
        turn_liveness.track_voice_stream(stream_id)
    worker = threading.Thread(
        target=streaming._run_agent_streaming,
        kwargs=dict(session_id=session.session_id, msg_text="hey", model="gpt-5.4",
                    workspace="/tmp", stream_id=stream_id),
        daemon=True,
    )
    worker.start()
    return channel, worker


class _Handler:
    def __init__(self, wfile):
        self.wfile = wfile
        self.headers = {}

    def send_response(self, _status):
        pass

    def send_header(self, _k, _v):
        pass

    def end_headers(self):
        pass


class _DeadSocket:
    """A crashed device: every write after the headers fails."""

    def write(self, _data):
        raise BrokenPipeError("device gone")

    def flush(self):
        pass


def _sse(stream_id, wfile):
    handler = _Handler(wfile)
    routes._handle_sse_stream(handler, urlparse(f"/api/chat/stream?stream_id={stream_id}"))
    return handler


def _wait(predicate, timeout=5.0):
    stop = time.monotonic() + timeout
    while time.monotonic() < stop:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _start_successor(monkeypatch, tmp_path, session):
    class NoopThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            return None

    monkeypatch.setattr(routes.uuid, "uuid4", lambda: type("FakeUuid", (), {"hex": "new-stream"})())
    monkeypatch.setattr(routes, "set_last_workspace", lambda workspace, **_kw: None)
    monkeypatch.setattr(routes, "create_stream_channel", lambda: queue.Queue())
    # NOTE: deliberately do NOT monkeypatch routes.threading.Thread here.
    # routes.threading IS the global threading module, so patching it leaks into
    # turn_liveness._ensure_reaper: the reaper thread becomes NoopThread
    # ('WorkerThread') with no .is_alive(), and the barge-in test fails with
    # AttributeError. The successor's worker thread starting for real is
    # harmless — it exits when the stream queue closes.
    return routes._start_chat_stream_for_session(
        session, msg="retry", attachments=[], workspace=str(tmp_path), model="gpt-5.4", model_provider=None,
    )


# ── 1. Consumer dies mid-turn: worker completes, persists, retry admitted ────


def test_consumer_disconnect_mid_turn_worker_completes_and_retry_succeeds(monkeypatch, tmp_path):
    sid, stream_id = "hark13_disconnect", "stream_hark13_disconnect"
    session = FakeSession(sid, stream_id)
    streamed, resume = threading.Event(), threading.Event()
    seen_rows = []

    def run(agent, **_kw):
        agent.stream_delta_callback("Working on it. ")
        streamed.set()
        assert resume.wait(5), "test never released the agent"
        with config.ACTIVE_RUNS_LOCK:
            seen_rows.append(dict(config.ACTIVE_RUNS.get(stream_id) or {}))
        return "Done: lights are off."

    with _agent_runtime(_agent_class(run), session):
        channel, worker = _launch(session, stream_id)
        assert streamed.wait(5)
        _sse(stream_id, _DeadSocket())  # write raises -> handler exits
        assert channel.subscriber_count() == 0
        resume.set()
        worker.join(10)
        assert not worker.is_alive(), "worker hung after its consumer died"

    # The turn ran on detached and persisted like any other turn.
    assert seen_rows and seen_rows[0].get("consumer") == "detached"
    assert seen_rows[0].get("phase") == "running"  # detach never touches phase
    assert session.messages[-1].get("content") == "Done: lights are off."
    assert session.active_stream_id is None
    assert stream_id not in config.ACTIVE_RUNS
    assert stream_id not in config.STREAMS

    response = _start_successor(monkeypatch, tmp_path, session)
    assert "error" not in response, response
    assert response["stream_id"] == "new-stream"


def test_reattach_clears_detached_marker():
    stream_id = "stream_hark13_reattach"
    channel = config.create_stream_channel()
    config.STREAMS[stream_id] = channel
    config.register_active_run(stream_id, session_id="s", phase="running")
    try:
        _sse(stream_id, _DeadSocket())
        assert config.ACTIVE_RUNS[stream_id]["consumer"] == "detached"
        assert turn_liveness.consumer_detached_seconds(stream_id) is not None
        q = channel.subscribe()
        turn_liveness.note_consumer_attached(stream_id)
        assert config.ACTIVE_RUNS[stream_id]["consumer"] == "attached"
        assert turn_liveness.consumer_detached_seconds(stream_id) is None
        channel.unsubscribe(q)
    finally:
        config.unregister_active_run(stream_id)


def test_disconnect_with_another_subscriber_does_not_detach():
    stream_id = "stream_hark13_two_tabs"
    channel = config.create_stream_channel()
    config.STREAMS[stream_id] = channel
    config.register_active_run(stream_id, session_id="s", phase="running")
    other = channel.subscribe()
    channel.put_nowait(("token", {"text": "x"}, None))
    try:
        _sse(stream_id, _DeadSocket())
        assert "consumer" not in config.ACTIVE_RUNS[stream_id]
    finally:
        channel.unsubscribe(other)
        config.unregister_active_run(stream_id)


# ── 2. Detached voice turn: clarify falls back, finished turn persists ───────


def test_detached_voice_turn_clarify_falls_back_and_result_persists(monkeypatch):
    monkeypatch.setattr(turn_liveness, "DETACHED_PROMPT_GRACE_SECONDS", 0.0)
    sid, stream_id = "hark13_voice_clarify", "stream_hark13_voice_clarify"
    session = FakeSession(sid, stream_id)
    streamed, resume = threading.Event(), threading.Event()
    answers = []

    def run(agent, **_kw):
        agent.stream_delta_callback("Let me check. ")
        streamed.set()
        assert resume.wait(5)
        # Default clarify timeout here is 3600s; only a live voice client
        # could answer, and it is gone.
        answers.append(agent.clarify_callback("Which room?", ["kitchen", "den"]))
        return "I turned off the kitchen lights."

    with _agent_runtime(_agent_class(run), session):
        _channel, worker = _launch(session, stream_id, voice_turn=True)
        assert streamed.wait(5)
        _sse(stream_id, _DeadSocket())
        resume.set()
        worker.join(10)
        assert not worker.is_alive(), "detached voice turn hung on clarify"

    assert answers and "did not provide a response" in answers[0]
    assert session.messages[-1].get("content") == "I turned off the kitchen lights."
    assert stream_id not in config.ACTIVE_RUNS


def test_clarify_wait_unchanged_for_attached_or_desktop_turns(monkeypatch):
    """Abandonment is voice-only and detach-only; everything else keeps waiting."""
    monkeypatch.setattr(turn_liveness, "DETACHED_PROMPT_GRACE_SECONDS", 0.0)
    desktop, voice_live = "stream_hark13_desktop", "stream_hark13_voice_live"
    for stream_id in (desktop, voice_live):
        config.register_active_run(stream_id, session_id="s", phase="running")
    turn_liveness.track_voice_stream(voice_live)
    with config.ACTIVE_RUNS_LOCK:
        config.ACTIVE_RUNS[desktop].update(consumer="detached", detached_at=time.time() - 3600)
    assert turn_liveness.voice_prompt_abandoned(desktop) is False  # tab may come back
    assert turn_liveness.voice_prompt_abandoned(voice_live) is False  # consumer attached

    entry = types.SimpleNamespace(event=threading.Event(), result=None)
    started = time.monotonic()
    assert streaming._await_clarify_response(entry, 1, threading.Event(), abandoned=lambda: False) == ("", True)
    assert time.monotonic() - started >= 0.9  # waited its full timeout
    assert streaming._await_clarify_response(entry, 3600, threading.Event(), abandoned=lambda: True) == ("", True)


# ── 3. Wedged voice turn: deadline force-finalizes, retry works ──────────────


def test_wedged_voice_turn_deadline_force_finalizes_and_retry_succeeds(monkeypatch, tmp_path):
    sid, stream_id = "hark13_wedged", "stream_hark13_wedged"
    session = FakeSession(sid, stream_id)
    in_tool, release = threading.Event(), threading.Event()
    successor_release = threading.Event()
    lines = []
    monkeypatch.setattr(voice, "emit_request_log", lines.append)

    def run(agent, **_kw):
        if in_tool.is_set():
            # The successor admitted after the deadline really runs (no turn
            # waits on the wedged worker's skill-home hold); keep it in flight.
            successor_release.wait(10)
            return "successor answer"
        in_tool.set()
        release.wait(30)  # a tool call that never returns (ignores interrupt)
        return "late answer from the wedged worker"

    with _agent_runtime(_agent_class(run), session):
        turn_id = voice.begin_turn(sid)
        _channel, worker = _launch(session, stream_id, voice_turn=True)
        assert voice.bind_stream(turn_id, sid, stream_id)
        assert in_tool.wait(5)
        assert _wait(lambda: config.AGENT_INSTANCES.get(stream_id) is not None)
        wedged_agent = config.AGENT_INSTANCES[stream_id]
        # Pin the reusable-cache entry the successor would otherwise borrow.
        config.SESSION_AGENT_CACHE[sid] = (wedged_agent, "sig")

        # Today's bug: every retry 409s while the wedged run owns the session.
        blocked = _start_successor(monkeypatch, tmp_path, session)
        assert blocked["_status"] == 409

        # Not yet overdue: nothing happens.
        assert turn_liveness.reap_overdue_voice_turns(unwind_grace=0.1) == []
        assert stream_id in config.ACTIVE_RUNS

        reaped = turn_liveness.reap_overdue_voice_turns(
            now=time.time() + turn_liveness.voice_turn_deadline_secs() + 1, unwind_grace=0.1,
        )
        assert reaped == [stream_id]
        assert worker.is_alive()  # still wedged; settlement did not wait on it
        assert stream_id not in config.ACTIVE_RUNS
        assert stream_id not in config.STREAMS
        cached = config.SESSION_AGENT_CACHE.get(sid)
        assert cached is None or cached[0] is not wedged_agent  # successor must not borrow it
        assert wedged_agent.interrupts  # the Stop path interrupted it
        assert session.active_stream_id is None
        assert session.messages[-1].get("_error") is True  # cancel marker persisted
        assert any('"closed_by":"deadline"' in line for line in lines)

        # Idempotent: a second pass finds nothing to settle.
        assert turn_liveness.reap_overdue_voice_turns(now=time.time() + 10_000, unwind_grace=0) == []
        assert turn_liveness.force_finalize_turn(stream_id, unwind_grace=0) is False

        response = _start_successor(monkeypatch, tmp_path, session)
        assert "error" not in response, response
        assert session.active_stream_id == "new-stream"

        # The wedged worker eventually returns: its stale result is rejected.
        release.set()
        worker.join(10)
        assert not worker.is_alive()
    assert session.active_stream_id == "new-stream"
    assert all(m.get("content") != "late answer from the wedged worker" for m in session.messages)
    successor_release.set()


def test_deadline_stamp_makes_cancelling_row_immediately_stale():
    fresh_user_cancel = {"phase": "cancelling", "cancelled_at": time.time()}
    deadline_cancel = dict(fresh_user_cancel, closed_by="deadline")
    assert routes._cancelled_run_is_stale(fresh_user_cancel) is False
    assert routes._cancelled_run_is_stale(deadline_cancel) is True


def test_deadline_leaves_a_user_cancel_to_its_owner():
    stream_id = "stream_hark13_user_cancel"
    config.register_active_run(stream_id, session_id="s", phase="cancelling", cancelled_at=time.time())
    try:
        assert turn_liveness.force_finalize_turn(stream_id, unwind_grace=0) is False
        assert "closed_by" not in config.ACTIVE_RUNS[stream_id]
    finally:
        config.unregister_active_run(stream_id)


# ── 4. Live consumer: behaviour unchanged ─────────────────────────────────────


def test_normal_voice_turn_with_live_consumer_is_unchanged():
    sid, stream_id = "hark13_live", "stream_hark13_live"
    session = FakeSession(sid, stream_id)
    attached = threading.Event()
    rows = []
    lines = []

    def run(agent, **_kw):
        assert _wait(lambda: config.STREAMS[stream_id].subscriber_count() == 1)
        attached.set()
        agent.stream_delta_callback("All good. ")
        with config.ACTIVE_RUNS_LOCK:
            rows.append(dict(config.ACTIVE_RUNS.get(stream_id) or {}))
        # A pass mid-turn, inside the deadline, finalizes nothing.
        rows.append(turn_liveness.reap_overdue_voice_turns(unwind_grace=0))
        return "All good."

    with mock.patch.object(voice, "emit_request_log", lines.append), \
         _agent_runtime(_agent_class(run), session):
        turn_id = voice.begin_turn(sid)
        channel, worker = _launch(session, stream_id, voice_turn=True)
        voice.bind_stream(turn_id, sid, stream_id)
        sink = io.BytesIO()
        reader = threading.Thread(target=_sse, args=(stream_id, sink), daemon=True)
        reader.start()
        worker.join(10)
        reader.join(10)
        assert not worker.is_alive() and not reader.is_alive()
        voice.record_client_stage(sid, turn_id, "playback_start", voice.now_ms())

    assert attached.is_set()
    assert "consumer" not in rows[0] and "closed_by" not in rows[0]
    assert rows[1] == []
    body = sink.getvalue().decode()
    assert "event: token" in body and "event: stream_end" in body and "event: cancel" not in body
    assert session.messages[-1].get("content") == "All good."
    assert not any('"detached"' in line for line in lines)
    # Finished streams leave the tracker without being reaped.
    assert turn_liveness.reap_overdue_voice_turns(now=time.time() + 10_000, unwind_grace=0) == []
    assert not turn_liveness.is_voice_stream(stream_id)


# ── 5. Barge-in: a new turn never waits on a running turn's skill-home hold ──


def test_voice_turn_reaches_the_model_while_another_turn_is_mid_reply(monkeypatch):
    """A voice turn started while an earlier turn is still running reaches its
    model call promptly.

    Before the fix, static skill modules made every turn hold the process-wide
    skill-home lock until its outer teardown. A second turn on any session (a
    barge-in, or Hark's 409 fork to a fresh session) parked before building its
    Agent: no model call, no token, "Thinking" until the first turn ended. The
    first turn is never cancelled here.
    """
    import api.profiles as profiles

    # Force the static-module (legacy) path the stall needs.
    monkeypatch.setattr(streaming, "_set_streaming_hermes_home_override", lambda home: (None, None, False))
    busy = FakeSession("hark13_busy", "stream_hark13_busy")
    nxt = FakeSession("hark13_next", "stream_hark13_next")
    sessions = {busy.session_id: busy, nxt.session_id: nxt}
    a_streaming, a_release, b_entered = threading.Event(), threading.Event(), threading.Event()
    stages = []
    monkeypatch.setattr(voice, "note_worker_stage", lambda sid, stage: stages.append((sid, stage)))

    def run(agent, **kw):
        if kw.get("task_id", agent.session_id) == busy.session_id:
            agent.stream_delta_callback("A long spoken reply. ")
            a_streaming.set()
            assert a_release.wait(10), "test never released the first turn"
            return "A done."
        b_entered.set()
        agent.stream_delta_callback("B reply. ")
        return "B done."

    with _agent_runtime(_agent_class(run), busy), \
         mock.patch.object(streaming, "get_session", side_effect=lambda sid, *a, **k: sessions[sid]):
        _c1, a_worker = _launch(busy, busy.active_stream_id, voice_turn=True)
        try:
            assert a_streaming.wait(5)
            assert profiles._SKILL_HOME_MODULE_PATCH_LOCK.holder_count() == 1
            started = time.monotonic()
            _c2, b_worker = _launch(nxt, nxt.active_stream_id, voice_turn=True)
            assert b_entered.wait(5), "second turn stalled before its model call while the first turn ran"
            assert time.monotonic() - started < 5
            b_worker.join(10)
            assert not b_worker.is_alive()
            assert a_worker.is_alive()  # the first turn kept running
        finally:
            a_release.set()
            a_worker.join(10)
    assert not a_worker.is_alive()
    assert nxt.messages[-1].get("content") == "B done."
    assert busy.messages[-1].get("content") == "A done."
    assert profiles._SKILL_HOME_MODULE_PATCH_LOCK.holder_count() == 0
    b_stages = [stage for sid, stage in stages if sid == "stream_hark13_next"]
    assert "skill_home_acquired" in b_stages and "model_call" in b_stages and "first_token" in b_stages


def test_skill_home_lease_shares_one_home_and_bounds_a_different_one(tmp_path):
    import api.profiles as profiles

    lease = profiles._SkillHomeModuleLease()
    calls = []
    hooks = dict(
        snapshot=lambda: calls.append("snapshot") or {"snap": 1},
        patch=lambda path: calls.append(("patch", path)),
        restore=lambda snap: calls.append(("restore", snap)),
    )
    alpha, beta = tmp_path / "alpha", tmp_path / "beta"
    outcome = {}

    def other_thread(name, home, timeout):
        got = lease.acquire(timeout=timeout, home=home, **hooks)
        outcome[name] = got
        if got:
            lease.release()

    assert lease.acquire(home=alpha, **hooks)
    same = threading.Thread(target=other_thread, args=("same", alpha, 1.0))
    same.start()
    same.join(5)
    assert outcome["same"] is True  # same home: joined without waiting
    started = time.monotonic()
    other = threading.Thread(target=other_thread, args=("other", beta, 0.2))
    other.start()
    other.join(5)
    assert outcome["other"] is False  # different home: bounded refusal
    assert time.monotonic() - started < 2
    assert calls == ["snapshot", ("patch", alpha)]  # one patch for the shared home

    # Same-thread re-entry for another home patches/restores only itself.
    assert lease.acquire(home=beta, **hooks)
    assert calls[-1] == ("patch", beta)
    lease.release()
    assert calls[-1] == ("restore", {"snap": 1}) and lease.holder_count() == 1

    lease.release()
    assert calls[-1] == ("restore", {"snap": 1})
    assert lease.holder_count() == 0
    with pytest.raises(RuntimeError):
        lease.release()
    # Drained: the other home is admitted now.
    late = threading.Thread(target=other_thread, args=("late", beta, 1.0))
    late.start()
    late.join(5)
    assert outcome["late"] is True


# ── 6. Env override ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("raw,expected", [
    (None, 600.0), ("45", 45.0), ("2.5", 2.5), ("0", 600.0), ("-5", 600.0), ("soon", 600.0),
])
def test_voice_turn_deadline_env(monkeypatch, raw, expected):
    assert turn_liveness._VOICE_TURN_DEADLINE_ENV == "HERMES_WEBUI_VOICE_TURN_DEADLINE"
    if raw is not None:
        monkeypatch.setenv(turn_liveness._VOICE_TURN_DEADLINE_ENV, raw)
    assert turn_liveness.voice_turn_deadline_secs() == expected


def test_reaper_honours_deadline_override(monkeypatch):
    stream_id = "stream_hark13_override"
    config.register_active_run(stream_id, session_id="s", phase="running")
    turn_liveness.track_voice_stream(stream_id, started_at=time.time() - 45)
    cancelled = []
    monkeypatch.setattr(turn_liveness, "_cancel_like_stop", lambda sid, backend: cancelled.append(sid) or True)
    try:
        assert turn_liveness.reap_overdue_voice_turns(unwind_grace=0) == []  # 45s < 600s default
        monkeypatch.setenv(turn_liveness._VOICE_TURN_DEADLINE_ENV, "30")
        assert turn_liveness.reap_overdue_voice_turns(unwind_grace=0) == [stream_id]
        assert cancelled == [stream_id]
        assert stream_id not in config.ACTIVE_RUNS
    finally:
        config.unregister_active_run(stream_id)

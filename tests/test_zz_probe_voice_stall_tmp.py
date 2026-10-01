"""TEMPORARY probe (deleted after investigation): turn B after/while turn A."""
import faulthandler
import sys
import threading
import time

import api.config as config
import api.routes as routes
import api.streaming as streaming
import api.voice as voice

from tests.test_turn_liveness_hark13 import (  # noqa: F401  (autouse fixture)
    FakeSession, _agent_class, _agent_runtime, _launch, _wait, _isolate,
)


def _start_b(monkeypatch, tmp_path, session):
    monkeypatch.setattr(routes, "set_last_workspace", lambda workspace, **_kw: None)
    return routes._start_chat_stream_for_session(
        session, msg="second", attachments=[], workspace=str(tmp_path), model="gpt-5.4",
        model_provider=None, voice_mode=True,
    )


def _run_probe(monkeypatch, tmp_path, mode):
    sid, a_stream = f"probe_{mode}", f"stream_a_{mode}"
    session = FakeSession(sid, a_stream)
    a_streaming, a_release = threading.Event(), threading.Event()
    b_entered = threading.Event()
    calls = []

    def run(agent, **kw):
        calls.append(kw.get("persist_user_message"))
        if len(calls) == 1:
            agent.stream_delta_callback("A long spoken reply. ")
            a_streaming.set()
            a_release.wait(20)
            return "A done."
        b_entered.set()
        agent.stream_delta_callback("B reply. ")
        return "B done."

    with _agent_runtime(_agent_class(run), session):
        _ch, a_worker = _launch(session, a_stream, voice_turn=True)
        assert a_streaming.wait(5)
        t0 = time.monotonic()
        if mode == "after_complete":
            a_release.set()
            a_worker.join(10)
        elif mode == "barge_cancel":
            h = type("H", (), {})()
            out = {}
            monkeypatch.setattr(routes, "j", lambda _h, payload, status=200: out.update(payload, _s=status))
            routes._handle_voice_interrupt(h, {"session_id": sid})
            print("[probe] interrupt ->", out, flush=True)
        # Poll chat/start like a client would.
        resp = None
        deadline = time.monotonic() + 30
        attempts = 0
        while time.monotonic() < deadline:
            attempts += 1
            resp = _start_b(monkeypatch, tmp_path, session)
            if attempts == 1 or resp.get("_status", 200) == 200:
                print(f"[probe]   attempt {attempts}: status={resp.get('_status', 200)} error={resp.get('error')!r}", flush=True)
            if resp.get("_status", 200) == 200 and "error" not in resp:
                break
            time.sleep(0.05)
        t_admit = time.monotonic() - t0
        print(f"[probe] {mode}: B admitted after {t_admit:.2f}s attempts={attempts} resp={resp}", flush=True)
        faulthandler.dump_traceback_later(8, exit=False, file=sys.stderr)
        ok = b_entered.wait(10)
        faulthandler.cancel_dump_traceback_later()
        print(f"[probe] {mode}: B entered run_conversation={ok} after {time.monotonic() - t0:.2f}s", flush=True)
        a_release.set()
        a_worker.join(10)
        assert ok


def test_probe_cross_session_concurrent(monkeypatch, tmp_path):
    """A on session 1 parked mid-reply; does B on session 2 reach the model?"""
    from unittest import mock
    s1, s2 = FakeSession("probe_x1", "stream_x1"), FakeSession("probe_x2", "stream_x2")
    a_streaming, a_release, b_entered = threading.Event(), threading.Event(), threading.Event()

    def run(agent, **kw):
        if agent.session_id == "probe_x1":
            agent.stream_delta_callback("A long spoken reply. ")
            a_streaming.set()
            a_release.wait(20)
            return "A done."
        b_entered.set()
        return "B done."

    with _agent_runtime(_agent_class(run), s1), \
         mock.patch.object(streaming, "get_session", side_effect=lambda sid, *a, **k: {"probe_x1": s1, "probe_x2": s2}[sid]):
        _c1, a_worker = _launch(s1, "stream_x1", voice_turn=True)
        assert a_streaming.wait(5)
        t0 = time.monotonic()
        _c2, b_worker = _launch(s2, "stream_x2", voice_turn=True)
        faulthandler.dump_traceback_later(4, exit=False, file=sys.stderr)
        ok = b_entered.wait(6)
        faulthandler.cancel_dump_traceback_later()
        print(f"[probe] cross-session: B entered={ok} after {time.monotonic() - t0:.2f}s", flush=True)
        a_release.set()
        a_worker.join(10)
        b_worker.join(10)
        assert ok


def test_probe_after_complete(monkeypatch, tmp_path):
    _run_probe(monkeypatch, tmp_path, "after_complete")


def test_probe_barge_cancel(monkeypatch, tmp_path):
    _run_probe(monkeypatch, tmp_path, "barge_cancel")

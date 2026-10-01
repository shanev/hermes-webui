"""Turn liveness: SSE consumer detachment and the voice-turn hard deadline.

A chat turn's worker never writes to a socket. It journals each event and
hands it to the stream's ``StreamChannel`` with ``put_nowait`` (bounded,
drop-oldest), and the SSE handler thread does the writes. A dead consumer
therefore cannot raise into or block the agent loop. What it CAN strand is a
wait that only that consumer could answer, such as a clarify prompt (default
3600s, ``<= 0`` waits forever), or any other wedge with nobody left to Stop
it. Three layers bound that (hark#13):

1. Detachment. When an SSE handler's write fails (``_CLIENT_DISCONNECT_ERRORS``)
   and no other subscriber remains, the run row is marked
   ``consumer="detached"``. The turn keeps running and persists normally. A
   reattaching subscriber flips it back to ``"attached"``. ``phase`` is not
   touched: Steer admission and finalizing depend on it.
2. Voice prompts. A voice turn detached past ``DETACHED_PROMPT_GRACE_SECONDS``
   stops waiting on clarify (the agent gets its existing "no response, use
   your best judgement" fallback), so the tool loop runs to completion.
3. Deadline. A voice turn older than ``HERMES_WEBUI_VOICE_TURN_DEADLINE``
   (600s) is force-finalized through the Stop path (Gateway stop, then
   ``cancel_stream``). Its run row is stamped ``closed_by="deadline"``, which
   the chat/start stale-cancel check treats as immediately stale. If the
   worker does not unwind within a short grace, the row and the session's
   cached Agent are released so a retry never 409s or borrows the wedged Agent.

State layer: ``api.config.ACTIVE_RUNS`` row metadata (``consumer``,
``detached_at``, ``closed_by``, ``closed_at``) plus this module's voice-stream
tracker. Lock order follows Stop: STREAMS_LOCK -> ACTIVE_RUNS_LOCK.
``StreamChannel``'s own lock is a leaf. ``_LOCK`` here is a leaf too. No
Agent call, session save or HTTP write happens under any registry lock.
"""

from __future__ import annotations

import logging
import os
import threading
import time

# The reaper must always be a REAL thread even when tests monkeypatch
# `threading.Thread` (via any module alias) with worker fakes.
_REAL_THREAD_CLASS = threading.Thread

logger = logging.getLogger(__name__)

_VOICE_TURN_DEADLINE_ENV = "HERMES_WEBUI_VOICE_TURN_DEADLINE"
_VOICE_TURN_DEADLINE_DEFAULT = 600.0

# A voice client that briefly loses its socket (network handoff, app resume)
# reconnects well inside this window and still sees a pending clarify prompt.
DETACHED_PROMPT_GRACE_SECONDS = 60.0

# After the deadline cancel, how long the worker gets to unwind by itself
# before its registry row and cached Agent are force-released.
_FORCE_UNWIND_GRACE_SECONDS = 5.0

# A tracked stream that never registered a worker (or whose worker is gone)
# is forgotten after this long.
_UNTRACK_GRACE_SECONDS = 60.0

_LOCK = threading.Lock()
_VOICE_STREAMS: dict[str, float] = {}  # stream_id -> chat/start epoch seconds
_REAPER_THREAD: threading.Thread | None = None


def voice_turn_deadline_secs() -> float:
    """Hard cap on a voice turn's lifetime (default 600s, env-tunable)."""
    raw = os.environ.get(_VOICE_TURN_DEADLINE_ENV)
    if raw:
        try:
            val = float(raw)
            if val > 0:
                return val
        except (TypeError, ValueError):
            pass
    return _VOICE_TURN_DEADLINE_DEFAULT


# ── Voice-stream tracking ────────────────────────────────────────────────────


def track_voice_stream(stream_id: str, started_at: float | None = None) -> None:
    """Put a voice-originated stream under the deadline. Call before its worker starts."""
    stream_id = str(stream_id or "").strip()
    if not stream_id:
        return
    with _LOCK:
        _VOICE_STREAMS[stream_id] = time.time() if started_at is None else float(started_at)
    _ensure_reaper()


def is_voice_stream(stream_id: str) -> bool:
    with _LOCK:
        return str(stream_id or "") in _VOICE_STREAMS


def _claim_voice_stream(stream_id: str) -> bool:
    """Single-owner settlement: only the caller that removes the entry proceeds."""
    with _LOCK:
        return _VOICE_STREAMS.pop(stream_id, None) is not None


# ── Consumer detachment ──────────────────────────────────────────────────────


def _subscriber_count(stream) -> int:
    counter = getattr(stream, "subscriber_count", None)
    if callable(counter):
        try:
            return int(counter())
        except Exception:
            return 0
    # Legacy single-reader queue: the reader that just left was the only one.
    return 0


def note_consumer_lost(stream_id: str, stream) -> bool:
    """An SSE handler for ``stream_id`` died on a write. Mark the run detached.

    Called after the handler unsubscribed. The run is marked only while the
    turn is live (still in STREAMS, not cancelling) and no other subscriber
    remains. The subscriber count is read under ACTIVE_RUNS_LOCK, the same
    lock ``note_consumer_attached`` writes under, so a racing reattach can't
    be overwritten by a stale "detached". Returns True when it marked.
    """
    stream_id = str(stream_id or "").strip()
    if not stream_id:
        return False
    from api import config as cfg

    marked = False
    with cfg.STREAMS_LOCK:
        live = cfg.STREAMS.get(stream_id) is stream and stream is not None
        with cfg.ACTIVE_RUNS_LOCK:
            row = cfg.ACTIVE_RUNS.get(stream_id)
            if (
                live
                and isinstance(row, dict)
                and str(row.get("phase") or "") != "cancelling"
                and not row.get("closed_by")
                and row.get("consumer") != "detached"
                and _subscriber_count(stream) == 0
            ):
                row["consumer"] = "detached"
                row["detached_at"] = time.time()
                marked = True
    if marked:
        logger.info("turn %s detached: SSE consumer lost; worker continues", stream_id)
        try:
            from api import voice

            voice.note_stream_detached(stream_id)
        except Exception:
            logger.debug("voice detach note failed for %s", stream_id, exc_info=True)
    return marked


def note_consumer_attached(stream_id: str) -> None:
    """A subscriber (re)attached to ``stream_id``. Clear a detached marker."""
    stream_id = str(stream_id or "").strip()
    if not stream_id:
        return
    from api import config as cfg

    with cfg.ACTIVE_RUNS_LOCK:
        row = cfg.ACTIVE_RUNS.get(stream_id)
        if isinstance(row, dict) and row.get("consumer") == "detached":
            row["consumer"] = "attached"
            row["detached_at"] = None


def consumer_detached_seconds(stream_id: str, now: float | None = None) -> float | None:
    """Seconds the run has had no consumer, or None while attached / unknown."""
    from api import config as cfg

    with cfg.ACTIVE_RUNS_LOCK:
        row = cfg.ACTIVE_RUNS.get(str(stream_id or ""))
        if not isinstance(row, dict) or row.get("consumer") != "detached":
            return None
        detached_at = row.get("detached_at")
    try:
        return max(0.0, (time.time() if now is None else now) - float(detached_at))
    except (TypeError, ValueError):
        return None


def voice_prompt_abandoned(stream_id: str) -> bool:
    """True when a voice turn's consumer has been gone past the prompt grace.

    A voice client is the only party that can answer a prompt on its turn, and
    a crashed one never will. Desktop turns keep today's behaviour: a closed
    tab may come back and answer.
    """
    if not is_voice_stream(stream_id):
        return False
    detached = consumer_detached_seconds(stream_id)
    return detached is not None and detached >= DETACHED_PROMPT_GRACE_SECONDS


# ── Deadline reaper ──────────────────────────────────────────────────────────


def _cancel_like_stop(stream_id: str, backend: str | None) -> bool:
    """Run the /api/chat/cancel sequence: Gateway outcome first, then local cancel.

    A failed Gateway stop is logged but does not stop the local settlement:
    the deadline is the last resort that keeps a wedged run from owning the
    session forever, and a late Gateway writeback is rejected by the
    session's writeback guard once ``active_stream_id`` has moved on.
    """
    if backend == "gateway":
        try:
            from api.routes import _gateway_stop_blocked_for_stream

            if _gateway_stop_blocked_for_stream(stream_id):
                logger.warning("deadline: Gateway stop failed for %s; settling locally", stream_id)
        except Exception:
            logger.debug("deadline: Gateway stop raised for %s", stream_id, exc_info=True)
    from api.runtime_adapter import LegacyJournalRuntimeAdapter, runtime_adapter_enabled
    from api.streaming import cancel_stream

    if runtime_adapter_enabled():
        adapter = LegacyJournalRuntimeAdapter(cancel_delegate=cancel_stream)
        return bool(adapter.cancel_run(stream_id).accepted)
    return bool(cancel_stream(stream_id))


def force_finalize_turn(
    stream_id: str,
    *,
    closed_by: str = "deadline",
    unwind_grace: float | None = None,
) -> bool:
    """Settle a wedged turn once so the session is immediately reusable.

    Idempotent: the run row's ``closed_by`` stamp, set under STREAMS_LOCK ->
    ACTIVE_RUNS_LOCK, admits exactly one finalizer, and a row already
    cancelling (a user Stop) or already gone is left to its owner. Returns
    True when this call settled the turn.
    """
    stream_id = str(stream_id or "").strip()
    if not stream_id:
        return False
    from api import config as cfg

    if unwind_grace is None:
        unwind_grace = _FORCE_UNWIND_GRACE_SECONDS
    with cfg.STREAMS_LOCK:
        stream_live = stream_id in cfg.STREAMS
        agent = cfg.AGENT_INSTANCES.get(stream_id)
        with cfg.ACTIVE_RUNS_LOCK:
            row = cfg.ACTIVE_RUNS.get(stream_id)
            if isinstance(row, dict):
                if row.get("closed_by") or str(row.get("phase") or "") == "cancelling":
                    return False
                row["closed_by"] = closed_by
                row["closed_at"] = time.time()
                backend = row.get("backend")
                session_id = str(row.get("session_id") or "").strip() or None
            elif stream_live:
                backend, session_id = None, None
            else:
                return False
    session_id = session_id or cfg.stream_owner_session_id(stream_id)
    logger.warning(
        "turn %s exceeded its %s; force-finalizing (session=%s backend=%s)",
        stream_id, closed_by, session_id, backend,
    )

    try:
        _cancel_like_stop(stream_id, backend)
    except Exception:
        logger.warning("deadline cancel failed for %s; releasing registry anyway", stream_id, exc_info=True)

    # Give a responsive worker the chance to unwind through its own finally,
    # which is the normal owner of this cleanup.
    stop_at = time.monotonic() + max(0.0, float(unwind_grace))
    while time.monotonic() < stop_at:
        with cfg.ACTIVE_RUNS_LOCK:
            if stream_id not in cfg.ACTIVE_RUNS:
                break
        time.sleep(0.05)

    with cfg.STREAMS_LOCK:
        cfg.STREAMS.pop(stream_id, None)
        cfg.CANCEL_FLAGS.pop(stream_id, None)
        agent = cfg.AGENT_INSTANCES.pop(stream_id, None) or agent
        with cfg.ACTIVE_RUNS_LOCK:
            row = cfg.ACTIVE_RUNS.get(stream_id)
            wedged = isinstance(row, dict) and row.get("closed_by") == closed_by
        # A wedged worker still holds its Agent mid-call. Retire it from the
        # reusable cache (same edge and ownership check as _agent_can_invoke)
        # so the successor builds a fresh one instead of borrowing it.
        if wedged and agent is not None and session_id:
            with cfg.SESSION_WRITEBACK_OWNERS_LOCK:
                if cfg.SESSION_WRITEBACK_OWNERS.get(session_id) == stream_id:
                    with cfg.SESSION_AGENT_CACHE_LOCK:
                        entry = cfg.SESSION_AGENT_CACHE.get(session_id)
                        if entry and entry[0] is agent:
                            cfg.SESSION_AGENT_CACHE.pop(session_id, None)
                            try:
                                from api.session_lifecycle import unregister_agent

                                unregister_agent(session_id)
                            except Exception:
                                logger.debug("unregister_agent failed for %s", session_id, exc_info=True)
    if wedged:
        cfg.unregister_active_run(stream_id)
        if session_id:
            cfg.clear_session_writeback_owner_if_owned(session_id, stream_id)
    try:
        from api import voice

        voice.close_turn_for_stream(stream_id, closed_by)
    except Exception:
        logger.debug("voice close failed for %s", stream_id, exc_info=True)
    return True


def reap_overdue_voice_turns(now: float | None = None, *, unwind_grace: float | None = None) -> list[str]:
    """Force-finalize voice turns past the deadline; forget finished ones."""
    from api import config as cfg

    now = time.time() if now is None else float(now)
    deadline = voice_turn_deadline_secs()
    with _LOCK:
        tracked = dict(_VOICE_STREAMS)
    if not tracked:
        return []
    with cfg.STREAMS_LOCK:
        live_streams = set(cfg.STREAMS.keys())
    with cfg.ACTIVE_RUNS_LOCK:
        live_runs = set(cfg.ACTIVE_RUNS.keys())
    overdue, finished = [], []
    for stream_id, started_at in tracked.items():
        age = now - started_at
        alive = stream_id in live_streams or stream_id in live_runs
        if not alive:
            if age >= _UNTRACK_GRACE_SECONDS:
                finished.append(stream_id)
        elif age >= deadline:
            overdue.append(stream_id)
    with _LOCK:
        for stream_id in finished:
            if _VOICE_STREAMS.get(stream_id) == tracked[stream_id]:
                _VOICE_STREAMS.pop(stream_id, None)
    reaped = []
    for stream_id in overdue:
        if not _claim_voice_stream(stream_id):
            continue
        if force_finalize_turn(stream_id, closed_by="deadline", unwind_grace=unwind_grace):
            reaped.append(stream_id)
    return reaped


def _reaper_loop() -> None:
    while True:
        time.sleep(min(30.0, max(1.0, voice_turn_deadline_secs() / 10.0)))
        try:
            reap_overdue_voice_turns()
        except Exception:
            logger.warning("voice-turn deadline reaper pass failed", exc_info=True)


def _ensure_reaper() -> None:
    global _REAPER_THREAD
    # Capture the real Thread class at import time (below): tests that monkeypatch
    # `routes.threading.Thread` (or any module alias of the threading module) with
    # worker fakes would otherwise replace the reaper's thread class too, and the
    # fakes lack .is_alive() — crashing production-callable code paths.
    with _LOCK:
        if _REAPER_THREAD is not None and getattr(_REAPER_THREAD, "is_alive", None) and _REAPER_THREAD.is_alive():
            return
        _REAPER_THREAD = _REAL_THREAD_CLASS(target=_reaper_loop, name="voice-turn-reaper", daemon=True)
        _REAPER_THREAD.start()


def reset_for_tests() -> None:
    with _LOCK:
        _VOICE_STREAMS.clear()

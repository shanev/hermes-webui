"""Voice-turn stage timing and barge-in bookkeeping (voice Phase 1).

A voice client (Hark) drives one reply turn through ``/api/transcribe``,
``/api/chat/start`` + the token stream, and per-sentence ``/api/tts``. This
module correlates those requests under one ``turn_id`` and emits one structured
``voice_turn`` log line per turn so stage latency can be read from the service
log.

State layer: process-local, in-memory, bounded. Nothing here is persisted and
nothing here is authoritative for run state — cancellation stays in
``api.streaming.cancel_stream`` and the active-run registry. A restart drops
open turns and armed turn notes (the barge-in note and the voice-mode
directive); both degrade to "no instrumentation" and a chat-formatted reply.

Privacy: records carry ids, stage names, timestamps and character counts only.
Reply text passes through ``note_reply_text`` solely to find the first sentence
boundary; at most a three-character tail is retained and it is never logged.

Locking: ``_LOCK`` is a leaf lock. No other lock is taken and no I/O happens
while it is held (log lines are built under it and written after release), so
it is safe to call from the streaming worker and from request threads that
hold no registry lock.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
import uuid
from collections import OrderedDict

from api.request_logging import emit_request_log


# Stage vocabulary, in pipeline order. The metrics endpoint rejects anything else.
STAGES = (
    "transcribe_start",
    "transcribe_done",
    "first_token",
    "first_sentence",
    "tts_first_byte",
    "playback_start",
)
_STAGE_SET = frozenset(STAGES)
# Stages the server cannot observe. A client report stamps the server clock too
# (receipt time), and the last pipeline stage closes the turn's log line.
_CLIENT_OBSERVED_STAGES = frozenset({"playback_start"})
_CLOSING_STAGE = "playback_start"

# POST /api/voice/metrics body cap: four short scalar fields.
MAX_METRIC_BODY_BYTES = 2048

_TURN_ID_RE = re.compile(r"^vt_[0-9a-f]{32}$")
_MAX_SESSION_ID_CHARS = 128
# Client timestamps are epoch milliseconds; the window rejects seconds-resolution
# values and garbage without trusting the client clock to agree with ours.
_MIN_CLIENT_TS_MS = 946_684_800_000  # 2000-01-01
_MAX_CLIENT_TS_MS = 4_102_444_800_000  # 2100-01-01

_MAX_TURNS = 512
# An open turn with no playback_start is logged once it is this old.
_TURN_FLUSH_SECONDS = 120.0
# Turn records outlive their log line so a late interrupt can still resolve
# turn_id -> stream_id.
_TURN_TTL_SECONDS = 900.0

INTERRUPT_NOTE = (
    "[System note: the previous spoken reply was interrupted by the user before "
    "it finished playing; they may not have heard all of it.]"
)
# Armed by /api/chat/start for a voice-originated turn. It rides the same
# one-shot, model-only note drain as INTERRUPT_NOTE.
VOICE_MODE_NOTE = (
    "[System note: This reply will be spoken aloud. Use plain prose: no markdown "
    "formatting, no code blocks, no tables, no URLs (describe links in words). "
    "Keep paragraphs short. Keep the reply under about 100 words unless the user "
    "asks for more detail — spoken replies much longer than that are exhausting. "
    "If code is essential, briefly describe it in words "
    "and note that the full code is in the chat transcript.]"
)
_NOTE_TTL_SECONDS = 900.0
_MAX_NOTES = 1024

# First-sentence heuristic: terminal punctuation (plus closing quotes/brackets)
# followed by whitespace, a CJK full stop, or a line break. It approximates the
# point where a per-sentence TTS client can dispatch its first request.
_SENTENCE_END_RE = re.compile(r"[.!?…]+[\"'”’)\]]*\s|[。！？]|\n")
_TAIL_CHARS = 3

_LOCK = threading.Lock()
_TURNS: "OrderedDict[str, dict]" = OrderedDict()  # turn_id -> record, oldest first
# stream_id -> turn_id while the reply still owes first_token/first_sentence.
# Read lock-free on the per-token hot path; written under _LOCK.
_AWAITING_REPLY: dict = {}
_INTERRUPT_NOTES: "OrderedDict[str, float]" = OrderedDict()  # session_id -> armed_at
# session_id -> (stream_id, armed_at). Bound to the stream it was armed for, so
# a turn that never drained it cannot hand the directive to a later turn.
_VOICE_MODE_NOTES: "OrderedDict[str, tuple]" = OrderedDict()


def now_ms() -> int:
    return int(time.time() * 1000)


def is_turn_id(value) -> bool:
    return isinstance(value, str) and bool(_TURN_ID_RE.fullmatch(value))


def _log_line(payload: dict) -> str:
    return "[webui] " + json.dumps(payload, separators=(",", ":"), sort_keys=True)


def _emit(lines) -> None:
    for line in lines:
        emit_request_log(line)


def _turn_line_locked(rec: dict, closed_by: str) -> str:
    """Build the turn's single log line and mark it emitted. Caller holds _LOCK."""
    rec["emitted"] = True
    stages = rec["stages"]
    origin = stages.get("transcribe_start")
    payload = {
        "event": "voice_turn",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "turn_id": rec["turn_id"],
        "session_id": rec["session_id"],
        "stream_id": rec["stream_id"],
        "closed_by": closed_by,
        "interrupted": bool(rec["interrupted"]),
        "stages": {stage: stages.get(stage) for stage in STAGES},
        "elapsed_ms": {
            stage: stages[stage] - origin
            for stage in STAGES
            if origin is not None and stage in stages
        },
    }
    if rec["client"]:
        payload["client_stages"] = dict(rec["client"])
    if rec["interrupted"]:
        payload["truncated_chars"] = int(rec["truncated_chars"] or 0)
    return _log_line(payload)


def _drop_turn_locked(turn_id: str, rec: dict) -> None:
    _TURNS.pop(turn_id, None)
    stream_id = rec.get("stream_id")
    if stream_id and _AWAITING_REPLY.get(stream_id) == turn_id:
        _AWAITING_REPLY.pop(stream_id, None)


def _sweep_locked(now: float) -> list:
    """Flush overdue open turns and drop expired ones. Caller holds _LOCK."""
    lines = []
    for turn_id, rec in list(_TURNS.items()):
        age = now - rec["created"]
        if age < _TURN_FLUSH_SECONDS:
            break  # insertion-ordered: everything after this is younger
        if not rec["emitted"]:
            lines.append(_turn_line_locked(rec, "timeout"))
        if age >= _TURN_TTL_SECONDS:
            _drop_turn_locked(turn_id, rec)
    return lines


def begin_turn(session_id=None, *, transcribe_start_ms=None, transcribe_done_ms=None) -> str:
    """Open a voice turn for a finished transcription and return its turn_id."""
    turn_id = "vt_" + uuid.uuid4().hex
    done_ms = now_ms() if transcribe_done_ms is None else int(transcribe_done_ms)
    stages = {"transcribe_done": done_ms}
    if transcribe_start_ms is not None:
        stages["transcribe_start"] = int(transcribe_start_ms)
    rec = {
        "turn_id": turn_id,
        "session_id": str(session_id) if session_id else None,
        "stream_id": None,
        "created": time.time(),
        "stages": stages,
        "client": {},
        "interrupted": False,
        "truncated_chars": 0,
        "emitted": False,
        "_seen_text": False,
        "_tail": "",
    }
    with _LOCK:
        lines = _sweep_locked(rec["created"])
        _TURNS[turn_id] = rec
        while len(_TURNS) > _MAX_TURNS:
            old_id, old = next(iter(_TURNS.items()))
            if not old["emitted"]:
                lines.append(_turn_line_locked(old, "evicted"))
            _drop_turn_locked(old_id, old)
    _emit(lines)
    return turn_id


def _owned_turn_locked(turn_id, session_id):
    """Return the turn record if it may be used by ``session_id``, else None."""
    rec = _TURNS.get(turn_id) if is_turn_id(turn_id) else None
    if rec is None:
        return None
    owner = rec["session_id"]
    if owner and owner != session_id:
        return None
    return rec


def bind_stream(turn_id, session_id, stream_id) -> bool:
    """Attach an accepted chat stream to a voice turn.

    Fails closed: an unknown turn, a turn owned by another session, or a turn
    already bound to a stream is left untouched.
    """
    session_id = str(session_id or "").strip()
    stream_id = str(stream_id or "").strip()
    if not session_id or not stream_id:
        return False
    with _LOCK:
        rec = _owned_turn_locked(turn_id, session_id)
        if rec is None or rec["stream_id"] or rec["emitted"]:
            return False
        rec["session_id"] = session_id
        rec["stream_id"] = stream_id
        _AWAITING_REPLY[stream_id] = rec["turn_id"]
    return True


def _first_sentence_complete(rec: dict, text: str) -> bool:
    tail = rec["_tail"]
    window = tail + text
    seen = rec["_seen_text"]
    complete = False
    for match in _SENTENCE_END_RE.finditer(window):
        # A boundary must end inside the new text (the tail was already
        # examined) and must follow some actual reply content.
        if match.end() > len(tail) and (seen or window[:match.start()].strip()):
            complete = True
            break
    rec["_seen_text"] = seen or bool(text.strip())
    rec["_tail"] = window[-_TAIL_CHARS:]
    return complete


def note_reply_text(stream_id, text) -> None:
    """Per-token hook from the streaming workers. Never raises.

    One lock-free dict miss for every stream that is not an open voice turn,
    and for voice turns once their first sentence has been seen.
    """
    turn_id = _AWAITING_REPLY.get(stream_id)
    if turn_id is None or not text:
        return
    try:
        now = now_ms()
        with _LOCK:
            if _AWAITING_REPLY.get(stream_id) != turn_id:
                return
            rec = _TURNS.get(turn_id)
            if rec is None:
                _AWAITING_REPLY.pop(stream_id, None)
                return
            rec["stages"].setdefault("first_token", now)
            if _first_sentence_complete(rec, str(text)):
                rec["stages"].setdefault("first_sentence", now)
                _AWAITING_REPLY.pop(stream_id, None)
    except Exception:
        pass


def note_reply_end(stream_id) -> None:
    """Worker-teardown hook: a reply that ended mid-sentence is one sentence."""
    turn_id = _AWAITING_REPLY.get(stream_id)
    if turn_id is None:
        return
    try:
        now = now_ms()
        with _LOCK:
            if _AWAITING_REPLY.pop(stream_id, None) != turn_id:
                return
            rec = _TURNS.get(turn_id)
            if rec is not None and "first_token" in rec["stages"]:
                rec["stages"].setdefault("first_sentence", now)
    except Exception:
        pass


def mark_stage(turn_id, stage: str) -> bool:
    """Stamp a server-observed stage on the server clock. First write wins."""
    if stage not in _STAGE_SET or not is_turn_id(turn_id):
        return False
    now = now_ms()
    with _LOCK:
        rec = _TURNS.get(turn_id)
        if rec is None or rec["emitted"]:
            return False
        rec["stages"].setdefault(stage, now)
    return True


def record_tts_sanitize(turn_id, *, code_blocks, spoken_chars, original_chars) -> bool:
    """Log what speech sanitization did to one TTS request of a voice turn.

    Counts only. A request without a well-formed turn_id logs nothing; the turn
    need not still be open, since per-sentence TTS outlives its log line.
    """
    if not is_turn_id(turn_id):
        return False
    _emit([_log_line({
        "event": "voice_tts_sanitize",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "turn_id": turn_id,
        "code_blocks": int(code_blocks),
        "spoken_chars": int(spoken_chars),
        "original_chars": int(original_chars),
    })])
    return True


def validate_metric_event(payload):
    """Validate a POST /api/voice/metrics body. Returns ``(event, error)``."""
    if not isinstance(payload, dict):
        return None, "JSON object required"
    session_id = payload.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        return None, "session_id required"
    session_id = session_id.strip()
    if len(session_id) > _MAX_SESSION_ID_CHARS:
        return None, "invalid session_id"
    turn_id = payload.get("turn_id")
    if not is_turn_id(turn_id):
        return None, "invalid turn_id"
    stage = payload.get("stage")
    if not isinstance(stage, str) or stage not in _STAGE_SET:
        return None, "unknown stage"
    ts = payload.get("ts")
    if isinstance(ts, bool) or not isinstance(ts, (int, float)) or not math.isfinite(ts):
        return None, "ts must be epoch milliseconds"
    if not _MIN_CLIENT_TS_MS <= ts <= _MAX_CLIENT_TS_MS:
        return None, "ts must be epoch milliseconds"
    return {"session_id": session_id, "turn_id": turn_id, "stage": stage, "ts": int(ts)}, None


def record_client_stage(session_id, turn_id, stage, ts) -> str:
    """Record a validated client-reported stage.

    Returns ``"recorded"``, ``"late"`` (the turn's line is already written) or
    ``"unknown_turn"`` (no such turn, or it belongs to another session).
    """
    now = now_ms()
    lines = []
    with _LOCK:
        lines.extend(_sweep_locked(time.time()))
        rec = _owned_turn_locked(turn_id, session_id)
        if rec is None:
            status = "unknown_turn"
        elif rec["emitted"]:
            status = "late"
        else:
            status = "recorded"
            rec["session_id"] = rec["session_id"] or session_id
            rec["client"].setdefault(stage, ts)
            if stage in _CLIENT_OBSERVED_STAGES:
                rec["stages"].setdefault(stage, now)
            if stage == _CLOSING_STAGE:
                lines.append(_turn_line_locked(rec, _CLOSING_STAGE))
    _emit(lines)
    return status


def stream_for_turn(turn_id, session_id):
    """Resolve a turn to its bound stream for ``session_id``.

    Returns ``(status, stream_id)`` with status ``"bound"``, ``"unbound"``
    (known turn, no chat stream attached), ``"unknown"`` or ``"mismatch"``
    (the turn belongs to another session).
    """
    if not is_turn_id(turn_id):
        return "unknown", None
    with _LOCK:
        rec = _TURNS.get(turn_id)
        if rec is None:
            return "unknown", None
        if rec["session_id"] and rec["session_id"] != session_id:
            return "mismatch", None
        if rec["stream_id"]:
            return "bound", rec["stream_id"]
        return "unbound", None


def record_interrupt(session_id, turn_id, *, truncated_chars: int, cancelled: bool, stream_id=None) -> None:
    """Log a barge-in and close the interrupted turn's line if still open."""
    truncated_chars = max(0, int(truncated_chars or 0))
    lines = []
    with _LOCK:
        rec = _owned_turn_locked(turn_id, session_id)
        if rec is not None:
            rec["interrupted"] = True
            rec["truncated_chars"] = truncated_chars
            if rec["stream_id"]:
                _AWAITING_REPLY.pop(rec["stream_id"], None)
            if not rec["emitted"]:
                lines.append(_turn_line_locked(rec, "interrupt"))
    lines.append(_log_line({
        "event": "voice_interrupt",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "session_id": session_id,
        "turn_id": turn_id if is_turn_id(turn_id) else None,
        "stream_id": stream_id,
        "cancelled": bool(cancelled),
        "truncated_chars": truncated_chars,
    }))
    _emit(lines)


def _arm_note_locked(notes: OrderedDict, session_id: str, value) -> None:
    """Arm (or re-arm) a session's note, newest last. Caller holds _LOCK."""
    notes.pop(session_id, None)
    notes[session_id] = value
    while len(notes) > _MAX_NOTES:
        notes.popitem(last=False)


def arm_interrupt_note(session_id) -> None:
    """Remember that this session's spoken reply was interrupted."""
    session_id = str(session_id or "").strip()
    if not session_id:
        return
    with _LOCK:
        _arm_note_locked(_INTERRUPT_NOTES, session_id, time.time())


def arm_voice_mode_note(session_id, stream_id) -> None:
    """Ask for a speech-friendly reply on the voice turn that owns ``stream_id``.

    Call before the turn's worker starts; the worker drains it with
    ``consume_turn_notes``. Arming never touches an armed interrupt note.
    """
    session_id = str(session_id or "").strip()
    stream_id = str(stream_id or "").strip()
    if not session_id or not stream_id:
        return
    with _LOCK:
        _arm_note_locked(_VOICE_MODE_NOTES, session_id, (stream_id, time.time()))


def consume_interrupt_note(session_id):
    """Return the one-shot interrupt note for the session's next turn, if armed."""
    if not _INTERRUPT_NOTES:
        return None
    with _LOCK:
        armed_at = _INTERRUPT_NOTES.pop(str(session_id or "").strip(), None)
    if armed_at is None or time.time() - armed_at > _NOTE_TTL_SECONDS:
        return None
    return INTERRUPT_NOTE


def consume_turn_notes(session_id, stream_id=None) -> list:
    """Drain the one-shot, model-only notes for the turn running on ``stream_id``.

    Returns the armed notes in delivery order: the interrupt note (about the
    previous reply), then the voice-mode directive (about this one). The
    directive is released only to the stream it was armed for. Any other
    stream leaves it in place, so a turn that never drained it cannot pass it
    to a later turn; the leftover is replaced by the session's next voice turn
    or evicted by the cap.
    """
    if not _INTERRUPT_NOTES and not _VOICE_MODE_NOTES:
        return []
    session_id = str(session_id or "").strip()
    stream_id = str(stream_id or "").strip()
    with _LOCK:
        interrupted_at = _INTERRUPT_NOTES.pop(session_id, None)
        voice_mode = _VOICE_MODE_NOTES.get(session_id)
        if voice_mode is not None and voice_mode[0] == stream_id:
            del _VOICE_MODE_NOTES[session_id]
        else:
            voice_mode = None
    now = time.time()
    notes = []
    if interrupted_at is not None and now - interrupted_at <= _NOTE_TTL_SECONDS:
        notes.append(INTERRUPT_NOTE)
    if voice_mode is not None and now - voice_mode[1] <= _NOTE_TTL_SECONDS:
        notes.append(VOICE_MODE_NOTE)
    return notes


def prepend_turn_notes(session_id, text: str, stream_id=None) -> str:
    """Prefix ``text`` with the turn's armed notes, consuming them."""
    return "\n\n".join([*consume_turn_notes(session_id, stream_id), text])


def reset_for_tests() -> None:
    with _LOCK:
        _TURNS.clear()
        _AWAITING_REPLY.clear()
        _INTERRUPT_NOTES.clear()
        _VOICE_MODE_NOTES.clear()

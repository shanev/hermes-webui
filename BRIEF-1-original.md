# Task: Voice turns hang with zero model output ("Thinking" stuck 2+ minutes) — find where the worker stalls before the first model API call

## Repo
`~/hermes-webui` (dev, branch feat/voice-phase1 — deploy via `git -C ~/hermes-webui-prod merge --ff-only feat/voice-phase1 && launchctl kickstart -k gui/$(id -u)/ai.tensor-systems.hark-webui`).

## Symptom (Hark iOS voice client)
Turns often hang in "Thinking" (and sometimes the previous "Speaking" persists); barge-in (speak-to-interrupt) does not work during the hang. User force-quits the app. Server keeps running. Happens "often", especially mid-reply (barge-in during TTS playback).

## Hard evidence from prod logs (~/hermes-webui-prod/logs/server.log)
1. Two stuck turns, both with the SAME signature — the voice_turn metric flushed at closed_by:"timeout" (voice.py:170 — turn open 120s, sweep found NO stages marked, NO client report):
   - ts 2026-10-01T05:30:55Z: `stages: {transcribe_done: 1790832560438, transcribe_start: 1790832559355}` — transcribe completed in 1249ms, then first_token: null, first_sentence: null, tts_first_byte: null, stream_id: null, session_id: null.
   - ts 2026-10-01T05:32:52Z: identical shape (transcribe_done 1083ms, all null).
2. For the 05:30 turn: POST /api/transcribe 200 at 05:30:55, POST /api/chat/start 200 at 05:30:56 (53ms), GET /api/chat/stream?stream_id=b7cb7b8795f3419e... 200, GET /api/tts/stream?stream_id=... 200. So the stream CONNECTED — but not a single token event arrived in 120s.
3. ~/.hermes/logs/agent.log has NO "API call" line for those turns (the agent never made the model call — the hang is BEFORE the first model request, inside the streaming worker startup path).
4. NOT model latency: the model was deepseek-v4p1-flash via ModelRelay; direct relay streaming measured TTFT 1.1s at 65K-token prompts; text chat through the WebUI with the same model streams fine (reproduced).
5. Context: the stuck turn came right after a long speaking turn (8 consecutive /api/tts 200s over 40s, 05:30:01–05:30:40) — i.e. the user barged in during TTS playback. Earlier architecture decision: "interruptions never cancel the server turn" — the interrupted turn's worker keeps running.
6. There are also [crash-visibility] process-exit lines in server.err.log for OTHER pids (70989, 71165, 71241, 71977, 72558) — worker/subprocess exits, may or may not be related.

## Where to look (what I verified so far)
- `api/voice.py`: voice turn registry. `bind_stream` (line ~218) attaches stream → turn; `_AWAITING_REPLY[stream_id] = turn_id`; `note_reply_text` marks first_token on the first token event; `_sweep_locked` flushes 120s-old stage-less turns as "timeout". The stuck turns were never marked → either bind_stream failed or note_reply_text never fired (no tokens).
- `api/routes.py` ~25547: chat/start calls `_voice.bind_stream(body["voice_turn_id"], s.session_id, response["stream_id"])` when status==200. bind_stream FAILS CLOSED (returns False) if the turn is unknown/foreign/already bound — if binding failed, note_reply_text's `_AWAITING_REPLY.get(stream_id)` is None and stages NEVER mark even if the model streams! CHECK THIS FIRST: is a failed bind the whole story (turn metric misleading) or is the model really not streaming? Note the user ALSO reports the app stuck in Thinking/Speaking and barge-in not working — consistent with a genuinely hung turn, not just missing metrics.
- `api/streaming.py` `_run_agent_streaming` (line 9742, ~5200 lines): worker startup — agent cache lookup under SESSION_AGENT_CACHE_LOCK (line ~11628), `_refresh_cached_agent_runtime`, `_adopt_session_db_for_cached_agent`, agent construction `_AIAgent(**_agent_kwargs)` (11602/11711), session migration handling (~12321), checkpoint saves holding the per-session agent lock (`_get_session_agent_lock`, routes.py 18049-18057 comment: the streaming thread holds this lock during checkpoint saves).
- The interrupted previous turn's worker is STILL RUNNING during the barge-in (turns never cancel server-side). If the old worker holds a lock the new worker needs (session agent lock during checkpoint save, SESSION_AGENT_CACHE_LOCK, SessionDB write lock), the new turn's worker blocks BEFORE creating/using the agent → no API call → no token → matches ALL the evidence.
- Also suspicious: `pending_started_at` in the chat/start response and the pending-queue code (`_pending` in routes.py ~11392, 21961) — a turn may sit in a pending queue while the old stream is still active.

## Task
1. Reproduce: start a turn on a session, and while its reply is STREAMING/TTS-ing, fire a second chat/start on the same session with a new voice_turn_id (the interrupted turn keeps running per design). Watch which thread blocks where — add targeted logging if needed (print with flush, prefix [webui] — matches existing style).
2. Find the exact blocking point (lock, queue, or DB) that leaves the second turn with zero tokens for 120s+.
3. Fix root cause. Constraints from the product decisions:
   - Interruptions must NOT cancel the in-flight turn (existing decision).
   - NO fallback modes; fix the real mechanism.
   - The new turn's worker must not be blocked by the old turn's checkpoint saves / agent-cache work. If serialization is by design, it must be bounded and the voice client must see progress or an error, never a silent 2-minute stall.
   - The turn metric must be truthful: if bind_stream fails or the worker stalls, that should be observable.
4. Tests: add a regression test reproducing turn-during-active-turn with the blocking point asserted bounded (no 120s stall).
5. Run: `./scripts/test.sh tests/test_voice_phase1.py tests/test_turn_liveness_hark13.py` — all must pass. Do NOT touch tests/test_voice.py or unrelated files.
6. CONSTRAINTS: commit NOTHING, do NOT push, do NOT restart the server, do NOT touch ~/hermes-webui-prod. I will review, deploy, and verify. Report files changed + root cause + test results.

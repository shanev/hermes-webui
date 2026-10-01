# USER CORRECTION (AUTHORITATIVE) — reframes the investigation

The user says: "I barged because it hung. I don't think barging caused it."

The barge-in is a RESPONSE to the hang, not the trigger. The turn hangs FIRST — ordinary voice turn goes into "Thinking" and never produces a token; the user then speaks to interrupt (which fails to help), and the app sits stuck in "Thinking" or the previous "Speaking" until force-quit.

## What this changes
1. Do NOT treat barge-in-during-TTS as the cause. The probe's 20s/409 blocking after a cancel is real but is a SECONDARY problem (recovery path), not the root cause.
2. Primary bug: an ordinary voice turn's worker never reaches the first model API call:
   - ~/.hermes/logs/agent.log has NO "API call" line for the stuck turns (05:30:56Z, 05:32:52Z chat/start 200 → zero model output for 120s+).
   - The hang SURVIVES a session fork (Hark retried on a brand-new session at 05:32:52 and hung the same way) → whatever blocks is GLOBAL, not per-session state: a process-wide lock, thread/subprocess/resource exhaustion, a dead shared component, or the worker crashing silently.
3. INVESTIGATE THE WORKER DYING: ~/hermes-webui-prod/logs/server.err.log has [crash-visibility] process-exit lines (pids 70989, 71165, 71241, 71977, 72558). Correlate those timestamps with the stuck turns. If the turn worker (or a subprocess it depends on, e.g. the agent venv python) exits mid-startup, the turn would hang exactly like this with no traceback. Find what those pids are (bootstrap.py child? agent subprocess?) and why they exit.
4. Also correlate the [crash-visibility] exits with the TTS sentence burst — if a worker/subprocess crash kills a shared resource (a forked python, a connection pool), later turns hang globally.

## Still valid
- The staging-log plan (worker_started → session_loaded → agent_ready → first_token, [webui] prefix, print flush) is the right diagnostic: deploy it, wait for one occurrence, read which stage never fires.
- barge-in cancel path review (why interrupt can return cancelled:False / not stop the stream) is still bug #2, now REFRAMED as: recovery doesn't work either, making the hang worse.
- Constraints: commit NOTHING, don't push, don't restart the server, don't touch ~/hermes-webui-prod. Regression test once root cause is confirmed; run ./scripts/test.sh tests/test_voice_phase1.py tests/test_turn_liveness_hark13.py at the end.

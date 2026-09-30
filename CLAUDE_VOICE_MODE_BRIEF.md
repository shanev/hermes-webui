# Task: voice-mode output adaptation — voice-optimized replies for voice-originated turns

Branch: feat/voice-phase1 (already checked out; commit as a SECOND commit, separate from the sanitize work). Do NOT push.

## Problem
Hermes replies in chat format (markdown, code blocks, tables). Spoken aloud, that is garbage.
The right fix is at the source: when a turn originates from voice, the agent should be told
to produce speech-friendly output. Defense-in-depth TTS stripping is handled separately.

## Scope
1. When /api/chat/start receives a turn with voice_turn_id (the voice-originated marker
   already wired), inject a voice-mode directive into the turn's context using the WebUI's
   EXISTING system-note/steer mechanism (the same one the interrupt note uses — read how
   api/voice.py arms the interruption note; reuse that exact mechanism, do not invent a second).
   Directive text (concise, imperative):
   "This reply will be spoken aloud. Use plain prose: no markdown formatting, no code blocks,
   no tables, no URLs (describe links in words). Keep paragraphs short. If code is essential,
   briefly describe it in words and note that the full code is in the chat transcript."
2. Persist nothing new: the directive rides the existing per-turn note mechanism (in-memory,
   15-min expiry is fine). It must apply to THIS turn only (or until the next voice turn —
   match whatever semantics the note mechanism already has).
3. Text-originated turns are unaffected (no directive).
4. Tests: directive injected for voice turns, absent for text turns, combines correctly when
   an interrupt note is ALSO armed on the same session (both notes reach the model, no clobbering).
5. Run ./scripts/test.sh on tests/test_voice_phase1.py and any new test file. Conventional commit.

Report: where the directive lands in the turn payload, test counts, any interplay issues
with the interrupt note.

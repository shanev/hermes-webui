# Task: TTS text sanitization — strip markdown/code syntax before speech synthesis

Branch: feat/voice-phase1 (already checked out). Add a commit on top. Do NOT push, do NOT restart the service.

## Problem (empirically confirmed)
_handle_tts in api/routes.py passes the raw text to the TTS engine with NO normalization.
Markdown-heavy or code-heavy replies produce audio that literally speaks syntax:
"triple backtick python def add open paren..." — gibberish for voice clients (Hark).

## Scope
1. New module api/tts_sanitize.py with a function `speech_text(text: str) -> tuple[str, str]`
   returning (spoken_text, code_note) where:
   - Markdown structure is stripped: headers (#, ##), bold/italic markers, inline code
     backticks, links (keep the label, drop the URL), images (drop or "image" placeholder),
     tables (convert rows to natural clauses or drop the table, keep nothing syntactic),
     bullet/numbered list markers (keep the text as short clauses), blockquotes.
   - Fenced code blocks (``` or ~~~) are REPLACED with a short spoken note like
     "There is a code block in the chat transcript." (code_note returns the count).
     NEVER speak code contents.
   - Inline code spans: speak the identifier if short (<= 20 chars, alnum/underscore/dot),
     else drop with the same transcript note.
   - URLs: replace with "link" or the domain ("link to example.com").
   - HTML entities (&amp; &lt; etc.) decode to plain text.
   - Collapse whitespace; cap result at the existing 5000-char limit AFTER sanitization.
   - Idempotent and safe on plain text (no-op, empty code_note).
2. Wire into _handle_tts: sanitize ONCE after reading the text field, before the
   engine-dispatch. If the original text contained a code block, append the code note
   sentence at the end of the spoken text. Log a metric (the existing voice_turn style)
   only if a voice_turn_id is present: fields {code_blocks: N, spoken_chars: M, original_chars: K}.
3. Tests (tests/test_tts_sanitize.py):
   - plain text unchanged (idempotence)
   - headers/bold/italic/links stripped
   - fenced code replaced with note; inline short identifier spoken; long identifier dropped
   - table flattened; entities decoded; whitespace collapsed
   - >5000 char input after sanitization behavior (matches existing cap behavior)
   - _handle_tts integration: POST /api/tts with markdown body returns 200 and the ENGINE
     receives the sanitized text (mock/patch the engine dispatch point used by tests)
   - code-block count metric emitted with voice_turn_id
4. Run: ./scripts/test.sh tests/test_tts_sanitize.py plus existing suite must stay green
   (test_voice_phase1.py, test_raw_audio_upload.py, test_voice_transcribe_endpoint.py at minimum).
5. Do NOT touch hermes-agent code, .env, auth middleware. Conventional commits.

Report: files changed, test counts, behavior decisions (esp. what a table becomes).

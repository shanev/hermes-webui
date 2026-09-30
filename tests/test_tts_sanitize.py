"""Speech sanitization for /api/tts (``api/tts_sanitize.py``).

Reply text is written for a chat surface. These tests pin what an engine is
asked to speak: prose only, markdown syntax gone, code never read aloud, and
the 5000-char cap applied to the sanitized text. Engine dispatch is mocked at
the same points the other TTS suites use (the ``edge_tts`` module and
``routes._tts_open``); no network and no real synthesis.
"""
import io
import json
import sys
import types

import pytest

import api.routes as routes
import api.tts_sanitize as tts_sanitize
import api.voice as voice
from api.tts_sanitize import sanitize, speech_text


BLOCK_NOTE = "There is a code block in the chat transcript."
INLINE_NOTE = "There is code in the chat transcript."
TURN_ID = "vt_" + "a" * 32

FENCED = (
    "Here is the function:\n\n```python\ndef add(a, b):\n    return a + b\n```\n\n"
    "Call it with two numbers."
)
INLINE = (
    "Call `my_func` then `os.path.join` and run "
    "`subprocess.run(['ls', '-la'], check=True)` now."
)
STYLED = (
    "# Title\n\nSome **bold** and *italic* and __strong__ text with a "
    "[label](https://example.com/page)."
)
TABLE = "Results:\n\n| Name | Age |\n|------|-----|\n| Alice | 30 |\n| Bob | 25 |\n\nDone."
LISTS = (
    "- First item\n- Second item!\n1. Third\n> quoted words\n\n---\n\n* [x] done task"
)
LINKS = (
    "![a cat](https://img.example.com/cat.png) See "
    "https://www.example.com/docs/page?x=1, or <https://api.example.org>."
)
ENTITIES = "Tom &amp; Jerry &lt;3 &quot;cheese&quot; &#39;now&#39;&nbsp;ok"
# What _handle_tts is sent, and what the engine must be asked to speak.
MARKDOWN_REPLY = (
    "## Result\n\nUse the **bold** move, see [the docs](https://example.com/docs).\n\n"
    "```py\nx = 1\n```"
)
SPOKEN_REPLY = "Result.\nUse the bold move, see the docs. " + BLOCK_NOTE


# ── plain text ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        "Hello there. It's 5 * 3 = 15, right? Email me at a_b@example.com.",
        "First line.\nSecond line.",
        "Use snake_case and 2*3*4 math.",
        "Version 3.5 costs $4.50 (approx.) — see section 2.",
        "Bonjour ! Comment ça va ?",
        "你好，世界。",
    ],
)
def test_plain_text_is_unchanged(text):
    assert speech_text(text) == (text, "")
    assert sanitize(text).spoken() == text


def test_empty_and_non_string_input_is_empty():
    assert speech_text("") == ("", "")
    assert sanitize(None) == ("", "", 0)


@pytest.mark.parametrize(
    "text", [FENCED, INLINE, STYLED, TABLE, LISTS, LINKS, ENTITIES, MARKDOWN_REPLY]
)
def test_sanitizing_is_idempotent(text):
    spoken = sanitize(text).spoken()

    again = sanitize(spoken)

    assert again.text == spoken
    assert again.code_note == ""
    assert again.code_blocks == 0


# ── markdown structure ──────────────────────────────────────────────────────


def test_headers_emphasis_and_links_are_stripped():
    assert speech_text(STYLED) == (
        "Title.\nSome bold and italic and strong text with a label.",
        "",
    )


def test_nested_and_intraword_emphasis_is_stripped():
    assert speech_text("***very*** important")[0] == "very important"
    # CJK text has no word boundary around the markers.
    assert speech_text("这是**重点**内容")[0] == "这是重点内容"


def test_list_markers_blockquotes_and_rules_are_stripped():
    assert speech_text(LISTS)[0] == (
        "First item.\nSecond item!\nThird.\nquoted words\ndone task."
    )


def test_images_and_urls_are_described_not_spelled_out():
    spoken, note = speech_text(LINKS)

    assert spoken == "image: a cat See link to example.com, or link to api.example.org."
    assert note == ""
    assert "http" not in spoken
    assert speech_text("![](diagram.png)")[0] == "image"
    assert speech_text("Visit www.example.com/path now")[0] == "Visit link to example.com now"
    # The closing bracket belongs to the sentence, not the URL.
    assert speech_text("(see https://en.wikipedia.org/wiki/Python_(language))")[0] == (
        "(see link to en.wikipedia.org)"
    )


# ── code ────────────────────────────────────────────────────────────────────


def test_fenced_code_is_replaced_by_the_transcript_note():
    result = sanitize(FENCED)

    assert result.text == "Here is the function:\nCall it with two numbers."
    assert result.code_note == BLOCK_NOTE
    assert result.code_blocks == 1
    assert result.spoken() == "Here is the function:\nCall it with two numbers. " + BLOCK_NOTE
    assert speech_text(FENCED) == (result.text, BLOCK_NOTE)
    for leaked in ("def add", "return", "python", "`"):
        assert leaked not in result.spoken()


def test_tilde_and_unclosed_fences_are_counted_and_never_spoken():
    result = sanitize("Intro.\n~~~\nsecret = 1\n~~~\nMiddle.\n```js\nlet hidden = 2;")

    assert result.text == "Intro.\nMiddle."
    assert result.code_blocks == 2
    assert result.code_note == "There are 2 code blocks in the chat transcript."


def test_a_fence_inside_a_line_is_still_a_code_block():
    result = sanitize("Use this: ```print('hi')``` and done.")

    assert result.text == "Use this: and done."
    assert result.code_blocks == 1


def test_a_reply_that_is_only_code_speaks_only_the_note():
    result = sanitize("```\nx = 1\n```")

    assert result.text == ""
    assert result.spoken() == BLOCK_NOTE


def test_inline_code_speaks_short_identifiers_and_drops_the_rest():
    result = sanitize(INLINE)

    assert result.text == "Call my func then os.path.join and run now."
    assert result.code_note == INLINE_NOTE
    assert result.code_blocks == 0
    assert "subprocess" not in result.spoken()


def test_long_identifiers_are_dropped():
    assert speech_text("Set `this_identifier_is_far_too_long` first.") == (
        "Set first.",
        INLINE_NOTE,
    )
    # The limit is 20 characters, inclusive.
    assert speech_text("`" + "a" * 20 + "`") == ("a" * 20, "")
    assert speech_text("`" + "a" * 21 + "`") == ("", INLINE_NOTE)


# ── tables, entities, whitespace ────────────────────────────────────────────


def test_table_rows_become_sentences():
    spoken, note = speech_text(TABLE)

    assert spoken == "Results:\nName: Alice, Age: 30.\nName: Bob, Age: 25.\nDone."
    assert note == ""
    assert "|" not in spoken
    assert "--" not in spoken


def test_table_cell_markup_and_alignment_rows_are_stripped():
    assert speech_text("| **Plan** | Price |\n|:--|--:|\n| Pro | $20 |")[0] == (
        "Plan: Pro, Price: $20."
    )


def test_a_table_row_without_its_header_is_a_plain_clause():
    assert speech_text("| Alice | 30 |")[0] == "Alice, 30."


def test_html_entities_are_decoded():
    assert speech_text(ENTITIES)[0] == "Tom & Jerry <3 \"cheese\" 'now' ok"


def test_whitespace_is_collapsed():
    assert speech_text("Hello   world\t\tagain\n\n\n\nNext   line  ")[0] == (
        "Hello world again\nNext line"
    )


def test_the_sanitizer_never_truncates():
    # Over-long speech is rejected by the endpoint, matching the existing cap.
    assert tts_sanitize.MAX_SPEECH_CHARS == 5000
    assert speech_text("x" * 6000) == ("x" * 6000, "")


# ── POST /api/tts ───────────────────────────────────────────────────────────


class _FakeHandler:
    def __init__(self, body: bytes, client="1.2.3.4"):
        self.command = "POST"
        self.rfile = io.BytesIO(body)
        self.wfile = io.BytesIO()
        self.headers = {"Content-Length": str(len(body))}
        self.client_address = (client, 12345)
        self.status = None
        self.sent_headers = {}

    def send_response(self, status):
        self.status = status

    def send_header(self, key, value):
        self.sent_headers[key] = value

    def end_headers(self):
        pass

    def payload(self):
        try:
            return json.loads(self.wfile.getvalue().decode("utf-8"))
        except Exception:
            return None


def _tts(text, client, **fields) -> _FakeHandler:
    body = {"text": text, "voice": "en-US-AriaNeural", "engine": "edge", **fields}
    handler = _FakeHandler(json.dumps(body).encode(), client=client)
    routes._handle_tts(handler, None)
    return handler


def _reset_limiter():
    if hasattr(routes._handle_tts, "_tts_limiter"):
        del routes._handle_tts._tts_limiter


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    # Auth + limiter sit before the engine branch; pin them off/clean so these
    # assertions are deterministic regardless of suite order.
    import api.auth as _auth
    monkeypatch.setattr(_auth, "is_auth_enabled", lambda: False)
    monkeypatch.setattr(routes, "is_auth_enabled", lambda: False, raising=False)
    monkeypatch.delenv("HERMES_WEBUI_TRUST_FORWARDED_FOR", raising=False)
    voice.reset_for_tests()
    _reset_limiter()
    yield
    _reset_limiter()
    voice.reset_for_tests()


@pytest.fixture(autouse=True)
def log_lines(monkeypatch):
    """Capture voice log lines instead of writing them to the service log."""
    lines = []
    monkeypatch.setattr(voice, "emit_request_log", lines.append)
    return lines


@pytest.fixture
def engine_text(monkeypatch):
    """Fake edge_tts module recording the text each synthesis is asked to speak."""
    seen = []
    fake_module = types.ModuleType("edge_tts")

    class FakeCommunicate:
        def __init__(self, text, voice, **kwargs):
            seen.append(text)

        def stream_sync(self):
            yield {"type": "audio", "data": b"\xff\xfb\x90" * 8}

    fake_module.Communicate = FakeCommunicate
    monkeypatch.setitem(sys.modules, "edge_tts", fake_module)
    return seen


def test_tts_engine_receives_the_sanitized_text(engine_text):
    handler = _tts(MARKDOWN_REPLY, "10.77.0.1")

    assert handler.status == 200
    assert handler.sent_headers.get("Content-Type") == "audio/mpeg"
    assert engine_text == [SPOKEN_REPLY]
    assert "x = 1" not in engine_text[0]


def test_tts_sanitizes_before_the_elevenlabs_engine_too(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk-test")
    import api.config as _cfg
    monkeypatch.setattr(_cfg, "get_config", lambda: {"tts": {"elevenlabs": {"voice_id": "pNInz6obpgDQGcFmaJgB", "model": "eleven_multilingual_v2"}}})
    captured = {}

    class _Resp:
        def __init__(self):
            self._chunks = [b"ID3fakeaudio", b""]
            self._i = 0

        def read(self, n=-1):
            c = self._chunks[self._i] if self._i < len(self._chunks) else b""
            self._i += 1
            return c

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _fake_tts_open(req, timeout=30, opener_factory=None, **_kw):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _Resp()

    monkeypatch.setattr(routes, "_tts_open", _fake_tts_open)

    handler = _tts(MARKDOWN_REPLY, "10.77.0.2", engine="elevenlabs")

    assert handler.status == 200
    assert captured["body"]["text"] == SPOKEN_REPLY


def test_tts_speaks_only_the_note_for_a_code_only_reply(engine_text):
    handler = _tts("```\nx = 1\n```", "10.77.0.3")

    assert handler.status == 200
    assert engine_text == [BLOCK_NOTE]


def test_tts_logs_the_code_block_metric_for_a_voice_turn(engine_text, log_lines):
    handler = _tts(MARKDOWN_REPLY, "10.77.0.4", turn_id=TURN_ID)

    assert handler.status == 200
    assert len(log_lines) == 1
    assert log_lines[0].startswith("[webui] ")
    entry = json.loads(log_lines[0].split(" ", 1)[1])
    assert entry.pop("ts")
    assert entry == {
        "event": "voice_tts_sanitize",
        "turn_id": TURN_ID,
        "code_blocks": 1,
        "spoken_chars": len(SPOKEN_REPLY),
        "original_chars": len(MARKDOWN_REPLY),
    }
    # Counts only: reply text never reaches the log.
    assert "bold" not in log_lines[0]


@pytest.mark.parametrize("fields", [{}, {"turn_id": "turn-1"}, {"turn_id": 7}])
def test_tts_logs_no_metric_without_a_voice_turn_id(engine_text, log_lines, fields):
    handler = _tts(MARKDOWN_REPLY, "10.77.0.5", **fields)

    assert handler.status == 200
    assert log_lines == []


def test_tts_rejected_requests_log_no_metric(engine_text, log_lines):
    handler = _tts("x" * 5001, "10.77.0.6", turn_id=TURN_ID)

    assert handler.status == 400
    assert log_lines == []


def test_tts_cap_applies_to_the_sanitized_text(engine_text):
    # Same behavior as before for plain text: 5000 passes, 5001 is rejected.
    at_cap = _tts("x" * 5000, "10.77.1.1")
    over_cap = _tts("x" * 5001, "10.77.1.2")
    # Markup does not count: raw text over the cap that speaks short is fine...
    raw = "Summary first.\n\n```\n" + "y = 2\n" * 1000 + "```"
    shrunk = _tts(raw, "10.77.1.3")
    # ...and markup cannot smuggle over-long speech past it.
    still_long = _tts("# " + "x" * 5001, "10.77.1.4")

    assert at_cap.status == 200
    assert over_cap.status == 400
    assert "too long" in over_cap.payload()["error"]
    assert len(raw) > 5000
    assert shrunk.status == 200
    assert still_long.status == 400
    assert "too long" in still_long.payload()["error"]
    assert engine_text == ["x" * 5000, "Summary first. " + BLOCK_NOTE]


def test_tts_cap_counts_the_code_note(engine_text):
    handler = _tts("x" * 4990 + "\n```\ncode\n```", "10.77.1.5")

    assert handler.status == 400
    assert "too long" in handler.payload()["error"]
    assert engine_text == []


def test_tts_bounds_raw_text_before_sanitizing(monkeypatch, engine_text):
    def _fail_if_called(_text):
        raise AssertionError("sanitized an over-long request")

    monkeypatch.setattr(tts_sanitize, "sanitize", _fail_if_called)

    handler = _tts("```\n" + "z" * tts_sanitize.MAX_INPUT_CHARS + "\n```", "10.77.1.6")

    assert handler.status == 400
    assert "too long" in handler.payload()["error"]
    assert engine_text == []


def test_tts_rejects_text_with_nothing_speakable(engine_text):
    handler = _tts("---", "10.77.1.7")

    assert handler.status == 400
    assert handler.payload() == {"error": "no speakable text"}
    assert engine_text == []


# --- Trivial code replies are spoken, not announced (issue #5) ---


def test_whole_reply_inline_code_span_is_spoken():
    assert sanitize("`localStorage`").spoken() == "localStorage"


def test_whole_reply_single_word_fenced_block_is_spoken():
    result = sanitize("```text\nlocalStorage\n```")
    assert result.spoken() == "localStorage"
    assert result.code_note == ""


def test_whole_reply_fenced_identifier_underscores_become_spaces():
    assert sanitize("```text\nmy_func_name\n```").spoken() == "my func name"


def test_code_note_stays_for_code_inside_prose():
    result = sanitize("Use `git rebase` to fix it.")
    assert "There is code" in result.spoken()


def test_code_note_stays_for_multiline_fenced_block():
    result = sanitize("```python\ndef add(a, b):\n    return a + b\n```")
    assert "There is a code block" in result.spoken()


def test_code_note_stays_when_fence_has_surrounding_prose():
    result = sanitize("```text\nlocalStorage\n```\nUse that.")
    assert "There is a code block" in result.spoken()

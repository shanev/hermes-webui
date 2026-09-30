"""Speech-text normalization for ``POST /api/tts``.

Replies are written for a chat surface: markdown, tables, code fences, URLs.
Handed to a TTS engine verbatim, that syntax is read aloud. ``sanitize`` turns
reply text into prose an engine can speak and reports what it left out, so the
listener can be told that code stayed behind in the chat transcript.

Pure text-in/text-out: no state, no I/O, no logging. Code contents are never
returned — fenced blocks are removed whole (an unclosed fence runs to the end
of the text, which is what a streamed or per-sentence chunk looks like), and
inline spans survive only when they are a short plain identifier.

Every pattern is bounded or built from delimiter-excluding character classes,
so cost stays linear in the input; ``MAX_INPUT_CHARS`` bounds the input itself.
"""

from __future__ import annotations

import html
import re
from typing import NamedTuple
from urllib.parse import urlsplit


# Cap on the text handed to a TTS engine (the long-standing /api/tts limit).
MAX_SPEECH_CHARS = 5000
# Cap on the raw request text. Markup and code shrink a lot, so raw text may
# exceed the speech cap; this only bounds the work done before that check.
MAX_INPUT_CHARS = 50_000

# Inline code is spoken only when it is a short plain identifier.
_MAX_IDENTIFIER_CHARS = 20
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9_.]{1,%d}" % _MAX_IDENTIFIER_CHARS)

_BACKTICK_FENCE_RE = re.compile(r"`{3,}")
_QUOTE_PREFIX = r"[ \t]*(?:>[ \t]*)*"
# A tilde fence must own its line; "~~~struck~~~ text" is not a fence.
_TILDE_OPEN_RE = re.compile(r"^" + _QUOTE_PREFIX + r"(~{3,})[^~\n]*$")
_TILDE_CLOSE_RE = re.compile(r"^" + _QUOTE_PREFIX + r"(~{3,})[ \t]*$")

_TABLE_DIVIDER_RE = re.compile(r"[\s|:\-]+")
_CELL_SPLIT_RE = re.compile(r"(?<!\\)\|")
_QUOTE_RE = re.compile(r"^[ \t]*(?:>[ \t]?)+")
_HEADING_RE = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+")
_HEADING_TAIL_RE = re.compile(r"[ \t]+#+[ \t]*$")
_LIST_RE = re.compile(r"^[ \t]*(?:[-*+]|\d{1,3}[.)])[ \t]+(?:\[[ xX]\][ \t]+)?")
_REF_DEF_RE = re.compile(r"^[ \t]{0,3}\[[^\]\n]+\]:[ \t]*\S")
# Characters that already end a spoken clause.
_TERMINAL = ".!?:;,…。！？"

_CODE_SPAN_RE = re.compile(r"(?<!\\)(`+)(.{1,300}?)\1(?!`)")
_SOFT_ESCAPE_RE = re.compile(r"\\[*`~]")
_ESCAPE_RE = re.compile(r"\\([\\{}\[\]()#+\-.!|>])")
_HTML_TAG_RE = re.compile(
    r"</?(?:a|b|blockquote|br|center|code|del|details|div|em|font|h[1-6]|hr|i|img|ins"
    r"|kbd|li|mark|ol|p|pre|s|small|span|strong|sub|summary|sup|table|tbody|td|th"
    r"|thead|tr|u|ul)\b[^<>\n]*>",
    re.IGNORECASE,
)
_LINK_TARGET = r"\([^()\n]*(?:\([^()\n]*\)[^()\n]*)*\)"
_IMAGE_RE = re.compile(r"!\[([^\[\]\n]*)\]" + _LINK_TARGET)
_LINK_RE = re.compile(r"\[([^\[\]\n]+)\]" + _LINK_TARGET)
_REF_LINK_RE = re.compile(r"(?<!\w)\[([^\[\]\n]+)\]\[[^\[\]\n]*\]")
_FOOTNOTE_RE = re.compile(r"\[\^[^\[\]\s]+\]")
_AUTOLINK_RE = re.compile(r"<((?:https?://|www\.)[^\s<>]+)>", re.IGNORECASE)
_URL_RE = re.compile(r"(?<![\w.@/])(?:https?://|www\.)[^\s<>\"]+", re.IGNORECASE)
_URL_TRAILING = ".,;:!?'\"*_~"
_HOST_RE = re.compile(r"[\w.-]+")

# Emphasis markers are removed where they hug a word; "5 * 3" and snake_case
# are left alone.
_STAR_OPEN_RE = re.compile(r"(?<![\w*])\*{1,3}(?=[^\s*])")
_STAR_CLOSE_RE = re.compile(r"(?<=[^\s*])\*{1,3}(?![\w*])")
# Intraword pairs (common in CJK text, where no word boundary surrounds them).
_STAR_PAIR_RE = re.compile(r"(?<![\d*])\*(?=[^\s*])([^*\n]*?)(?<=[^\s*])\*(?![\d*])")
_UNDERSCORE_OPEN_RE = re.compile(r"(?<!\w)_{1,3}(?=[^\s_])")
_UNDERSCORE_CLOSE_RE = re.compile(r"(?<=[^\s_])_{1,3}(?!\w)")
_STRAY_MARKER_RE = re.compile(r"\*{2,}|~~+|`+")

_ENTITY_RE = re.compile(r"&(?:#\d{1,7}|#[xX][0-9a-fA-F]{1,6}|[A-Za-z][A-Za-z0-9]{1,31});")
_HSPACE_RE = re.compile(r"[^\S\n]+")
_LINE_BREAK_RE = re.compile(r" ?\n[ \n]*")
_SPACE_BEFORE_STOP_RE = re.compile(r" +([,.])(?=\s|$)")


class SpeechText(NamedTuple):
    """Result of :func:`sanitize`."""

    text: str  # speakable prose, without the code note
    code_note: str  # sentence telling the listener code was left out, or ""
    code_blocks: int  # fenced code blocks removed

    def spoken(self) -> str:
        """The text to synthesize: the prose followed by the code note."""
        if not self.code_note:
            return self.text
        return f"{_terminated(self.text)} {self.code_note}".strip()


def _terminated(text: str) -> str:
    """End a clause with a full stop unless it already ends one."""
    core = text.rstrip("*_~` \t")
    if not core or core[-1] in _TERMINAL:
        return text
    return text + "."


def _code_note(code_blocks: int, dropped_inline: int) -> str:
    if code_blocks == 1:
        return "There is a code block in the chat transcript."
    if code_blocks > 1:
        return f"There are {code_blocks} code blocks in the chat transcript."
    if dropped_inline:
        return "There is code in the chat transcript."
    return ""


def _strip_fenced_code(text: str) -> tuple[str, int]:
    """Remove fenced code blocks. Returns ``(text, block_count)``."""
    out = []
    count = 0
    fence = None  # (fence character, run length) while inside a block
    for line in text.split("\n"):
        if fence is not None and fence[0] == "~":
            match = _TILDE_CLOSE_RE.match(line)
            if match and len(match.group(1)) >= fence[1]:
                fence = None
            out.append("")
            continue
        if fence is None:
            match = _TILDE_OPEN_RE.match(line)
            if match:
                fence = ("~", len(match.group(1)))
                count += 1
                out.append("")
                continue
        # Backtick fences may open or close mid-line (a client that joined
        # lines, or prose running straight into a fence), so pair runs rather
        # than whole lines. Text outside the runs is kept.
        kept = []
        pos = 0
        for match in _BACKTICK_FENCE_RE.finditer(line):
            run = len(match.group())
            if fence is None:
                kept.append(line[pos:match.start()])
                fence = ("`", run)
                count += 1
            elif run >= fence[1]:
                fence = None
                pos = match.end()
        if fence is None:
            kept.append(line[pos:])
        out.append(" ".join(kept))
    return "\n".join(out), count


def _is_rule(line: str) -> bool:
    """Horizontal rule or setext heading underline."""
    compact = "".join(line.split())
    return len(compact) >= 3 and len(set(compact)) == 1 and compact[0] in "*-_="


def _is_table_divider(line: str) -> bool:
    return "|" in line and "-" in line and bool(_TABLE_DIVIDER_RE.fullmatch(line))


def _table_cells(line: str) -> list:
    row = line.strip()
    if row.startswith("|"):
        row = row[1:]
    if row.endswith("|") and not row.endswith("\\|"):
        row = row[:-1]
    return [cell.strip() for cell in _CELL_SPLIT_RE.split(row)]


def _row_clause(cells: list, headers: list) -> str:
    """One table row as a sentence: ``Name: Alice, Age: 30.``"""
    parts = []
    for index, cell in enumerate(cells):
        if not cell:
            continue
        header = headers[index] if index < len(headers) else ""
        parts.append(f"{header}: {cell}" if header else cell)
    return _terminated(", ".join(parts)) if parts else ""


def _plain_line(line: str) -> str:
    line = _QUOTE_RE.sub("", line)
    if _is_rule(line) or _REF_DEF_RE.match(line):
        return ""
    match = _HEADING_RE.match(line)
    if match:
        return _terminated(_HEADING_TAIL_RE.sub("", line[match.end():]).strip())
    match = _LIST_RE.match(line)
    if match:
        return _terminated(line[match.end():].strip())
    return line


def _flatten_blocks(text: str) -> str:
    """Drop block-level syntax line by line; tables become one sentence per row."""
    lines = text.split("\n")
    out = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if "|" in line and index + 1 < len(lines) and _is_table_divider(lines[index + 1]):
            headers = _table_cells(line)
            index += 2
            rows = 0
            while index < len(lines) and "|" in lines[index]:
                out.append(_row_clause(_table_cells(lines[index]), headers))
                rows += 1
                index += 1
            if not rows:
                out.append(_row_clause(headers, []))
            continue
        index += 1
        if _is_table_divider(line):
            continue
        stripped = line.strip()
        if len(stripped) > 1 and stripped.startswith("|") and stripped.endswith("|"):
            # A row that arrived without its header (per-sentence chunking).
            out.append(_row_clause(_table_cells(line), []))
            continue
        out.append(_plain_line(line))
    return "\n".join(out)


def _spoken_image(match) -> str:
    alt = match.group(1).strip()
    return f"image: {alt}" if alt else "image"


def _spoken_url(match) -> str:
    url = match.group()
    # Sentence punctuation, emphasis markers and an unbalanced closing bracket
    # belong to the surrounding prose, not the URL.
    opens = {")": url.count("("), "]": url.count("[")}
    closes = {")": url.count(")"), "]": url.count("]")}
    end = len(url)
    while end:
        last = url[end - 1]
        if last in _URL_TRAILING:
            end -= 1
        elif last in closes and closes[last] > opens[last]:
            closes[last] -= 1
            end -= 1
        else:
            break
    core, tail = url[:end], url[end:]
    try:
        host = urlsplit(core if "://" in core else "http://" + core).hostname or ""
    except ValueError:
        host = ""
    host = host.removeprefix("www.")
    if host and _HOST_RE.fullmatch(host):
        return f"link to {host}{tail}"
    return f"link{tail}"


def _strip_inline(text: str) -> tuple[str, int]:
    """Remove inline syntax. Returns ``(text, dropped_code_span_count)``."""
    dropped = 0

    def code_span(match) -> str:
        nonlocal dropped
        code = match.group(2).strip()
        if _IDENTIFIER_RE.fullmatch(code):
            # Underscores become spaces so an engine says "my func", not
            # "my underscore func".
            return " ".join(code.replace("_", " ").split())
        dropped += 1
        return " "

    text = _CODE_SPAN_RE.sub(code_span, text)
    text = _SOFT_ESCAPE_RE.sub("", text)
    text = text.replace("\\_", "_")
    text = _HTML_TAG_RE.sub(" ", text)
    text = _FOOTNOTE_RE.sub("", text)
    text = _IMAGE_RE.sub(_spoken_image, text)
    text = _LINK_RE.sub(r"\1", text)
    text = _REF_LINK_RE.sub(r"\1", text)
    text = _AUTOLINK_RE.sub(r"\1", text)
    text = _URL_RE.sub(_spoken_url, text)
    text = _STAR_OPEN_RE.sub("", text)
    text = _STAR_CLOSE_RE.sub("", text)
    text = _STAR_PAIR_RE.sub(r"\1", text)
    text = _UNDERSCORE_OPEN_RE.sub("", text)
    text = _UNDERSCORE_CLOSE_RE.sub("", text)
    text = _STRAY_MARKER_RE.sub("", text)
    text = _ESCAPE_RE.sub(r"\1", text)
    return text, dropped


def _collapse_whitespace(text: str) -> str:
    text = _HSPACE_RE.sub(" ", text)
    text = _LINE_BREAK_RE.sub("\n", text)
    text = _SPACE_BEFORE_STOP_RE.sub(r"\1", text)
    return text.strip()


def sanitize(text) -> SpeechText:
    """Turn reply text into speakable prose. Plain prose passes through."""
    if not isinstance(text, str) or not text:
        return SpeechText("", "", 0)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text, code_blocks = _strip_fenced_code(text)
    text = _flatten_blocks(text)
    text, dropped_inline = _strip_inline(text)
    text = _ENTITY_RE.sub(lambda match: html.unescape(match.group()), text)
    text = _collapse_whitespace(text)
    return SpeechText(text, _code_note(code_blocks, dropped_inline), code_blocks)


def speech_text(text: str) -> tuple[str, str]:
    """Return ``(spoken_text, code_note)`` for ``text``.

    ``code_note`` is empty unless code was left out; otherwise it is the
    sentence to speak after ``spoken_text`` (it carries the block count).
    """
    result = sanitize(text)
    return result.text, result.code_note

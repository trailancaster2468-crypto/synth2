"""AI Director: a natural-language production assistant built on Claude tool use.

The director drives the studio exclusively through the tool layer
(:mod:`synth2.ai.tools`), so everything it does is visible, undoable and
identical to what a human or an MCP client could do::

    from synth2.ai.tools import Session
    from synth2.ai.director import Director

    d = Director(Session())
    out = d.chat("Write a 4-bar lullaby in F major for seraph and render it")
    print(out["reply"])
    for a in out["actions"]:
        print(a["tool"], a["ok"], a["result_summary"])

Requires the optional ``anthropic`` package and credentials (``ANTHROPIC_API_KEY``
or an ``ant auth login`` profile).  The model defaults to ``claude-opus-5-5`` and
can be overridden with the ``SYNTH2_MODEL`` environment variable or the ``model``
argument; ``SYNTH2_EFFORT`` sets the reasoning effort (``low``..``max``).

Implementation notes (Claude API):

* manual agentic loop over streamed requests (``client.beta.messages.stream`` +
  ``get_final_message()``), executing every ``tool_use`` block with
  :func:`~synth2.ai.tools.call_tool` and returning all ``tool_result`` blocks in
  one user message (``is_error`` for :class:`~synth2.ai.tools.ToolError`), until
  ``stop_reason`` is no longer ``tool_use``;
* the conversation is append-only (full ``response.content`` echoed back, the
  system prompt and tool list never change), which keeps prompt caching and
  thinking-block validity intact; the volatile studio state is sent inside each
  user message instead of the system prompt;
* prompt-cache breakpoints on the system prompt (covers tools + system) and a
  top-level automatic breakpoint for the growing message history;
* server-side refusal fallbacks (``fallbacks="default"``) on models that support them.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ..timing import midi_to_note
from ..voices import VOICES
from .tools import TOOLS, Session, ToolError, call_tool

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT = "high"
DEFAULT_MAX_TOKENS = 32000
FALLBACK_BETA = "server-side-fallback-2026-07-01"
# models that accept the server-side ``fallbacks="default"`` parameter
FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-fable-5", "claude-sonnet-5-5"}
MAX_TOOL_RESULT_CHARS = 60000
_CREDENTIAL_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE", "ANTHROPIC_FEDERATION_RULE_ID")


class DirectorUnavailable(RuntimeError):
    """The director cannot run: the ``anthropic`` package or API credentials are missing."""


SYSTEM_PROMPT = """\
You are the AI Director of Synth2 Vocal Studio, a singing-voice synthesizer studio (like Synthesizer V Studio). \
You are a skilled producer, vocal arranger and topliner. You compose, arrange, perform and mix vocal tracks by \
calling the studio's tools; the user watches the same project in a piano-roll GUI and can also edit it. Every \
tool call is undoable, so act decisively, then verify your work.

# Studio conventions
- Time is in beats: quarter notes, beat 0 is the start of the song. In 4/4, bar N starts at beat 4*(N-1) (bar 1 = \
beat 0, bar 2 = beat 4, bar 9 = beat 32). Seconds only appear in render/analysis reports and in a few expression \
parameters that say so.
- Pitch is a MIDI number or a note name: C4 = 60 = middle C, A4 = 69, F#3, Bb5.
- Each track is one monophonic singer: notes in a track must not overlap. Harmony and doubles go on extra tracks.
- Lyrics, one token per note: a word or syllable; "+" = the next syllable of the previous word (a note "hello" \
followed by a note "+" sings hel-lo; always write the WHOLE word on the first note - "twin" "+" would not sing \
"twinkle"); "-" = melisma, keep the previous vowel on a new pitch; "br" = a breath; \
"[s t aa r]" = explicit phonemes (ARPABET-style, see list_phonemes); Japanese uses kana or romaji, one mora per \
note. set_lyrics splits free text into syllables for you (call phonemize to see how many notes a text needs).
- Per-note attributes: velocity 0..2 (1 = normal dynamics), vibrato_depth (semitones), vibrato_rate (Hz), \
vibrato_start (s), transition (glide time into the note, s), overshoot 0..1, tension -1..1, breathiness -1..1, \
detune (cents), consonant_length (multiplier), language.
- Automation curves per track (set_curve, points [[beat, value], ...]): loudness (dB, -48..12), pitchDelta \
(cents), tension, breathiness and gender (-1..1), voicing (0 whisper .. 1), toneShift (cents, formants), \
vibratoEnv (0..2 multiplier).
- Expression styles (pop, ballad, rock, jazz, rnb, soft, classical, robotic, flat) automatically generate \
glides, scoops, falls, vibrato, drift, phrase dynamics and breaths. Choose one per track, then refine with \
apply_expression (style_overrides, voice_params, note_attrs).
- Voices and comfortable ranges: {voices}. Write the melody mostly in the middle of the range; save the top few \
notes for climaxes.

# Score notation (write_melody, new_project(score=...))
Notes are "PITCH DURATION LYRIC" separated by spaces, placed one after another:
- DURATION: w=4 beats, h=2, q=1, e=1/2, s=1/4, t=1/8; dot "q."=1.5; triplet "e3"=1/3; tie "h+q"=3; or a number.
- "r DURATION" is a rest. "|" bar lines are optional but recommended: a bar line that is not on a bar boundary \
produces a warning, which catches rhythm mistakes. "@ 16" jumps to beat 16, "@ bar 5" to bar 5. "#" comments.
- The lyric may be omitted (it becomes "la") when you plan to call set_lyrics afterwards.
- "word{velocity=1.2,vibrato_depth=0.5}" sets note attributes; "word[s t aa r]" overrides the pronunciation.
- Lines "curve: loudness 0:-6 8:0 (linear)" add automation; "tempo: 90" adds a tempo change at that point.
Example (4/4, starting in bar 2):
  @ bar 2
  E4 q twinkle  E4 q +  B4 q twinkle  B4 q +  | C#5 q little  C#5 q +  B4 h star |
  A4 q how  A4 q I  G#4 q wonder  G#4 q +  | F#4 e what  F#4 e -  F#4 q you  E4 h are | r w
("twinkle" + "+" sings twin-kle over two notes; "what" "-" holds the vowel across two pitches.)
A full score adds headers and track lines:
  title: My Song / tempo: 96 / time: 4/4 / key: E major  (one per line)
  track: Lead voice=aurora style=pop gain=0 pan=0
  ...note lines...
  track: Harmony voice=atlas style=pop gain=-4 pan=-0.3

# Workflow
1. Understand the request. For a new song, decide and briefly state a plan: genre/mood, key, tempo, time \
signature, voice(s), song form (e.g. verse 8 bars, chorus 8 bars) with bar numbers.
2. Set up: new_project (or update_project/set_tempo on the current one), add_track/update_track for voices.
3. Write the melody with write_melody, one phrase or section per call, using "@ bar N" to place sections. \
Leave a lead-in: start the first phrase at bar 2 (beat 4), or at least at beat 1, so consonants and the first \
breath are not cut off.
4. Lyrics: write them inline or call set_lyrics per phrase (start_beat/end_beat), after checking syllable counts \
with phonemize. Make syllable counts match note counts.
5. Performance: pick the style; use apply_expression for vibrato on long notes, velocity accents on downbeats and \
climaxes, breathiness for intimate verses, tension for powerful choruses; loudness curves for crescendos.
6. Arrangement: add_harmony for harmonies (steps=2: a third above, -2: a third below, -3: a fourth below), \
usually in choruses only (start_beat/end_beat); duplicate_track with detune_cents for doubles; set gains and pans \
(harmonies 3-6 dB below the lead, panned apart).
7. Check: call check_project and fix every error and the relevant warnings.
8. Render: preview sections with render(start_beat, end_beat) while working; render the whole song at the end. \
Then call analyze_render and act on its issues (pitch errors over ~50 cents, unvoiced notes, clipping), re-render \
and re-check. Do not claim something sounds good without having rendered and analysed it.
9. Save with save_project when the user asks or when a project path exists.

# Musical good practice
- Singable melodies move mostly by step with occasional leaps (a leap of a 5th or more usually resolves by step \
in the opposite direction); keep each phrase within about an octave; avoid long runs of repeated fast notes.
- Phrases are typically 2 or 4 bars. Leave at least half a beat of rest between phrases so the singer can breathe; \
no phrase should run longer than about 8-10 seconds without a rest. Put a "br" note in a rest only when you want \
an audible breath (most styles add automatic breaths).
- One syllable per note. Put stressed syllables on strong beats and longer notes; unstressed syllables on weak \
beats. Use melismas ("-") sparingly, mostly on open vowels and at phrase ends. Avoid notes shorter than about \
0.1 s (a sixteenth at 150 bpm) for syllables with consonant clusters.
- Prefer real dictionary words; split multi-syllable words over notes with "+" (never type fragments like "kle"). \
If a word is pronounced wrongly, give the note explicit phonemes.
- Harmonies follow the lead's rhythm and lyrics; thirds and sixths sound sweet, fourths/fifths open and powerful. \
Keep each harmony inside its voice's range.
- End phrases on stable scale tones (1, 3, 5) and hold final notes long enough for vibrato (about 1 s or more).

# Using the tools well
- Each user message starts with a <studio_state> block: a summary of the current project. Use get_notes or \
get_project(detail="score") to see notes and their ids; never invent note ids.
- Batch work: one write_melody per section, one set_lyrics per phrase or section, edit_notes with many edits at once.
- Tool errors explain exactly what was wrong; fix the arguments and try again instead of giving up.
- If a change made things worse, use undo.
- Finish with a short, friendly summary for the user: what you did (with bar numbers), the render path if you \
rendered, any analysis findings, and suggestions for next steps. Do not paste raw JSON or long note lists.
"""

SYSTEM_PROMPT = SYSTEM_PROMPT.replace("{voices}", "; ".join(
    f"{v.id} ({v.description.rstrip('.')}, {midi_to_note(v.range_low)}-{midi_to_note(v.range_high)})"
    for v in VOICES.values()))


def director_available() -> tuple[bool, str]:
    """``(True, "")`` when the director can run, else ``(False, reason)``."""
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False, "the 'anthropic' package is not installed (pip install 'synth2[ai]' or pip install anthropic)"
    if any(os.environ.get(k) for k in _CREDENTIAL_ENV) or (Path.home() / ".config" / "anthropic").is_dir():
        return True, ""
    return False, ("no Anthropic API credentials found: set ANTHROPIC_API_KEY (or log in with `ant auth login`, "
                   "which stores a profile the SDK reads automatically)")


def _make_client():
    ok, reason = director_available()
    if not ok:
        raise DirectorUnavailable(f"AI director unavailable: {reason}")
    import anthropic

    return anthropic.Anthropic()


def tool_params(eager_input_streaming: bool = True) -> list[dict]:
    """The studio tools as Claude API tool definitions."""
    out = []
    for t in TOOLS:
        d = {"name": t["name"], "description": t["description"], "input_schema": t["input_schema"]}
        if eager_input_streaming:
            d["eager_input_streaming"] = True
        out.append(d)
    return out


def _btype(block) -> str | None:
    return block.get("type") if isinstance(block, dict) else getattr(block, "type", None)


def _battr(block, name, default=None):
    return block.get(name, default) if isinstance(block, dict) else getattr(block, name, default)


def _echo_content(content: list) -> list:
    """Content to append to the history for an assistant response.

    Normally the full content, unchanged.  After a mid-output server-side
    fallback, model-internal blocks that precede the last ``fallback`` marker
    (thinking, tool calls of the declined attempt) are omitted, as the API requires.
    """
    idx = max((i for i, b in enumerate(content) if _btype(b) == "fallback"), default=None)
    if idx is None:
        return list(content)
    drop = {"thinking", "redacted_thinking", "tool_use", "server_tool_use"}
    return [b for i, b in enumerate(content) if i > idx or _btype(b) not in drop]


def _summarize(result: Any, limit: int = 240) -> str:
    """One-line human summary of a tool result."""
    if not isinstance(result, dict):
        s = str(result)
        return s if len(s) <= limit else s[: limit - 3] + "..."
    parts = []
    for k, v in result.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            sv = str(v)
            parts.append(f"{k}={sv[:60] + '...' if len(sv) > 60 else sv}")
        elif isinstance(v, list):
            parts.append(f"{k}=[{len(v)}]")
        elif isinstance(v, dict):
            parts.append(f"{k}={{...}}")
        if sum(len(p) + 2 for p in parts) > limit:
            break
    s = ", ".join(parts)
    return s if len(s) <= limit else s[: limit - 3] + "..."


def _state_block(session: Session) -> str:
    try:
        summary = call_tool(session, "get_project", {"detail": "summary"})
    except Exception as exc:  # pragma: no cover - defensive
        summary = {"error": str(exc)}
    return "<studio_state>\n" + json.dumps(summary, ensure_ascii=False) + "\n</studio_state>"


class Director:
    """Natural-language production assistant operating a :class:`~synth2.ai.tools.Session`.

    ``chat(message)`` runs Claude with the studio tools until it stops calling
    tools and returns ``{"reply": str, "actions": [{tool, args, ok, result_summary}],
    "stop_reason": str, "turns": int}``.  The conversation history persists across
    calls; :meth:`reset` clears it.
    """

    def __init__(self, session: Session, model: str | None = None, max_turns: int = 25, *, client=None,
                 effort: str | None = None, max_tokens: int = DEFAULT_MAX_TOKENS, fallbacks: bool | None = None,
                 eager_input_streaming: bool = True, include_state: bool = True):
        self.session = session
        self.model = model or os.environ.get("SYNTH2_MODEL") or DEFAULT_MODEL
        self.max_turns = int(max_turns)
        self.effort = effort if effort is not None else (os.environ.get("SYNTH2_EFFORT") or DEFAULT_EFFORT)
        self.max_tokens = int(max_tokens)
        self.fallbacks = (self.model in FALLBACK_MODELS) if fallbacks is None else bool(fallbacks)
        self.include_state = include_state
        self.client = client if client is not None else _make_client()
        self.tools = tool_params(eager_input_streaming)
        self.system = [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]
        self.history: list[dict] = []

    # ------------------------------------------------------------------ API
    def reset(self):
        self.history = []

    def chat(self, message: str) -> dict:
        start_len = len(self.history)
        text = f"{_state_block(self.session)}\n\n{message}" if self.include_state else message
        self.history.append({"role": "user", "content": [{"type": "text", "text": text}]})
        actions: list[dict] = []
        last_texts: list[str] = []
        stop_reason = None
        tool_uses: list = []
        turns = 0
        limited = False
        while True:
            if turns >= self.max_turns:
                limited = True
                break
            turns += 1
            resp = self._request()
            stop_reason = getattr(resp, "stop_reason", None)
            content = _echo_content(list(getattr(resp, "content", None) or []))
            if stop_reason == "refusal":
                # declined (after any fallback): drop this exchange so it does not poison later turns
                del self.history[start_len:]
                return {"reply": "Sorry - that request was declined by the model's safety system, so I could not "
                                 "work on it. Try rephrasing it.", "actions": actions, "stop_reason": stop_reason,
                        "turns": turns}
            if content:
                self.history.append({"role": "assistant", "content": content})
            texts = [_battr(b, "text", "") for b in content if _btype(b) == "text" and _battr(b, "text")]
            if texts:
                last_texts = texts
            if stop_reason == "pause_turn":
                continue
            tool_uses = [b for b in content if _btype(b) == "tool_use"]
            if not tool_uses:
                break
            # every tool_use gets a tool_result (all in one user message) so the history stays valid
            truncated = stop_reason == "max_tokens"
            self.history.append({"role": "user",
                                 "content": [self._run_tool(tu, actions, truncated) for tu in tool_uses]})
            if stop_reason not in ("tool_use", "max_tokens"):
                break
        reply = "\n\n".join(last_texts)
        if stop_reason == "max_tokens" and not tool_uses:
            reply = (reply + "\n\n(The response was cut off at the output limit.)").strip()
        if limited:
            reply = (reply + f"\n\n(Stopped after {self.max_turns} model turns; ask me to continue.)").strip()
        return {"reply": reply, "actions": actions, "stop_reason": stop_reason, "turns": turns}

    # ------------------------------------------------------------- internals
    def _request(self):
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": self.system,
            "tools": self.tools,
            "messages": self.history,
            "cache_control": {"type": "ephemeral"},
        }
        if self.effort:
            params["output_config"] = {"effort": self.effort}
        if self.fallbacks:
            params["betas"] = [FALLBACK_BETA]
            params["fallbacks"] = "default"
        with self.client.beta.messages.stream(**params) as stream:
            return stream.get_final_message()

    def _run_tool(self, tu, actions: list, truncated: bool = False) -> dict:
        name = _battr(tu, "name")
        args = _battr(tu, "input") or {}
        tid = _battr(tu, "id")
        if truncated:
            msg = ("this tool call was cut off because the response hit the output token limit, so it was not "
                   "run; issue it again with a shorter input (e.g. split a long melody into several write_melody "
                   "calls)")
            actions.append({"tool": name, "args": args, "ok": False, "result_summary": msg})
            return {"type": "tool_result", "tool_use_id": tid, "content": msg, "is_error": True}
        try:
            if not isinstance(args, dict):
                raise ToolError(f"tool input must be a JSON object, got {type(args).__name__}")
            result = call_tool(self.session, name, args)
        except ToolError as exc:
            actions.append({"tool": name, "args": args, "ok": False, "result_summary": str(exc)[:300]})
            return {"type": "tool_result", "tool_use_id": tid, "content": f"Error: {exc}", "is_error": True}
        except Exception as exc:  # engine/IO failures are reported to the model, not raised
            msg = f"{type(exc).__name__}: {exc}"
            actions.append({"tool": name, "args": args, "ok": False, "result_summary": msg[:300]})
            return {"type": "tool_result", "tool_use_id": tid, "content": f"Error: {msg}", "is_error": True}
        text = json.dumps(result, ensure_ascii=False)
        if len(text) > MAX_TOOL_RESULT_CHARS:
            text = (text[:MAX_TOOL_RESULT_CHARS] + f"... [truncated {len(text) - MAX_TOOL_RESULT_CHARS} characters; "
                    "request less detail, a single track or a beat range]")
        actions.append({"tool": name, "args": args, "ok": True, "result_summary": _summarize(result)})
        return {"type": "tool_result", "tool_use_id": tid, "content": text}


def main(argv: list[str] | None = None) -> int:
    """Interactive director REPL: ``python -m synth2.ai.director [--project PATH]``."""
    import argparse

    ap = argparse.ArgumentParser(prog="synth2-director", description="Chat with the Synth2 AI director")
    ap.add_argument("--project", help="project file to load")
    ap.add_argument("--model", help=f"Claude model (default {DEFAULT_MODEL} or $SYNTH2_MODEL)")
    args = ap.parse_args(argv)
    session = Session.open(args.project) if args.project and Path(args.project).exists() else Session(path=args.project)
    try:
        director = Director(session, model=args.model)
    except DirectorUnavailable as exc:
        print(exc)
        return 2
    print(f"Synth2 AI director ({director.model}). Type your request; Ctrl-D to quit.")
    while True:
        try:
            msg = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not msg:
            continue
        out = director.chat(msg)
        for a in out["actions"]:
            print(f"  [{'ok' if a['ok'] else 'error'}] {a['tool']}: {a['result_summary']}")
        print(out["reply"])


if __name__ == "__main__":
    raise SystemExit(main())

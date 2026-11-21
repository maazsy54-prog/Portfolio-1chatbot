"""Safety checks.

Input guard   : blocks prompt-injection / jailbreak / extraction attempts in the question.
Output guard  : blocks leaks of the system prompt or secrets in the answer.

Defence in depth: these heuristics are cheap and catch known attack shapes, but
regexes can always be evaded. The real protection is layered: the system prompt
treats retrieved text as data, citations are validated, and the output scan
catches leaks even if an attack gets through.
"""
import re
import unicodedata
from dataclasses import dataclass

# ---------------------------------------------------------------- output guard

_SECRET_PATTERNS = (
    re.compile(r"pcsk_[A-Za-z0-9_]{10,}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"postgres(?:ql)?(?:\+\w+)?://\S+:\S+@\S+"),
)


def scan_output(text: str, *, canary: str, secrets: list[str]) -> str | None:
    """Return a reason if the text leaks the system prompt or a secret, else None.

    Args:
        text: model output about to be shown to the user.
        canary: marker planted in the system prompt; seeing it means a prompt leak.
        secrets: real secret values loaded from the environment.
    """
    if canary and canary in text:
        return "system_prompt_leak"
    if any(len(s) >= 8 and s in text for s in secrets):
        return "secret_leak"
    if any(p.search(text) for p in _SECRET_PATTERNS):
        return "secret_pattern"
    return None


# ----------------------------------------------------------------- input guard

BLOCKED_INPUT_TEXT = (
    "I can't help with that request. Please ask a question about your documents."
)

_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\u00ad\u180e"))
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass(frozen=True)
class InputVerdict:
    """Result of checking a question."""

    blocked: bool
    category: str | None = None  # prompt_injection | jailbreak | prompt_extraction | secret_extraction
    rule: str | None = None


def strip_control_chars(text: str) -> str:
    """Remove control characters (NUL would crash Postgres; others hide text)."""
    return _CONTROL_CHARS.sub("", text)


_INSTR = r"(?:instructions?|prompts?|rules|directives?|guidelines)"
_QUAL = r"(?:previous|prior|above|earlier|preceding|original|system|initial)"
_VERB_SHOW = (
    r"(?:reveal|show|print|display|repeat|output|tell\s+me|give\s+me|share|leak|dump|recite|"
    r"what\s+(?:is|are|was|were)|send)"
)

# (category, rule name, compiled pattern). Applied to normalised, lower-cased text.
_RULES: tuple[tuple[str, str, re.Pattern[str]], ...] = tuple(
    (cat, name, re.compile(pattern))
    for cat, name, pattern in (
        # --- prompt injection ---
        ("prompt_injection", "override_instructions",
         rf"\b(?:ignore|disregard|forget|override)\s+(?:(?:all|any|every|the|your|my|these|those)\s+)*"
         rf"(?:{_QUAL}\s+)?{_INSTR}\b"),
        ("prompt_injection", "ignore_everything_above",
         r"\b(?:ignore|disregard|forget)\s+(?:everything|anything|all)\s+(?:above|before|prior|you\s+(?:were|have\s+been)\s+(?:told|instructed|given))"),
        ("prompt_injection", "new_instructions_header",
         r"\b(?:new|updated|revised|real|actual)\s+(?:instructions?|system\s+prompt)\s*[:\-]"),
        # --- jailbreak ---
        ("jailbreak", "persona_unrestricted",
         r"\b(?:you\s+are\s+now|from\s+now\s+on\s+you\s+are|act\s+as|pretend\s+(?:to\s+be|you\s+are|you're)|"
         r"roleplay\s+as|behave\s+as|simulate)\b[^.?!\n]{0,60}\b(?:dan|jailbroken|unfiltered|unrestricted|"
         r"uncensored|evil|without\s+(?:any\s+)?(?:rules|restrictions|limits|filters)|"
         r"no\s+(?:rules|restrictions|limits|filters))"),
        ("jailbreak", "known_jailbreak_terms",
         r"\b(?:dan\s+mode|do\s+anything\s+now|developer\s+mode|god\s+mode|jailbreak(?:ed)?|dude\s+mode)\b"),
        ("jailbreak", "disable_safety",
         r"\b(?:disable|turn\s+off|remove|bypass|circumvent|evade)\b[^.?!\n]{0,20}\b(?:your|the)\s+"
         r"(?:safety|filters?|guardrails?|restrictions?|rules|censorship|content\s+polic(?:y|ies))"),
        # --- prompt extraction ---
        ("prompt_extraction", "reveal_system_prompt",
         rf"\b{_VERB_SHOW}\b[^.?!\n]{{0,25}}\b(?:your|the|ur)\s+(?:(?:hidden|secret|full|complete|initial)\s+)*"
         r"(?:system|hidden|secret|initial|developer|underlying)\s+(?:prompt|instructions?|message)\b"),
        ("prompt_extraction", "system_prompt_phrase",
         r"\b(?:your\s+(?:system\s+)?prompt|system\s+prompt)\b"),
        ("prompt_extraction", "what_were_you_told",
         r"\bwhat\s+(?:were|are)\s+you\s+(?:told|instructed|programmed|configured)\b"),
        ("prompt_extraction", "repeat_everything_above",
         r"\b(?:repeat|print|output|copy)\s+(?:everything|all|the\s+text)\s+(?:above|before|prior)\b"),
        ("prompt_extraction", "internal_marker",
         r"\b(?:internal\s+(?:marker|identifier)|canary)\b"),
        # --- secret extraction ---
        ("secret_extraction", "reveal_secrets",
         rf"\b{_VERB_SHOW}\b[^.?!\n]{{0,25}}\b(?:your|its|the\s+(?:server|system|backend|bot|assistant)'?s?)\s+"
         r"(?:\w+\s+){0,2}(?:api[\s_-]?keys?|secret[\s_-]?keys?|access[\s_-]?tokens?|passwords?|credentials|"
         r"environment\s+variables|env\s+vars|connection\s+string|database\s+url)"),
        ("secret_extraction", "env_access",
         r"(?:\bos\.environ\b|\bprocess\.env\b|\bprintenv\b|\bcat\s+\.env\b)"),
    )
)

# Role/delimiter markers need line structure, so they run on the un-collapsed text.
_MARKER_RULES: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    ("prompt_injection", "fake_chat_markers",
     re.compile(r"<\|?\s*(?:im_start|im_end|system|endoftext)\s*\|?>|\[/?(?:inst|sys)\]")),
    ("prompt_injection", "fake_role_line",
     re.compile(r"(?:^|\n)\s*(?:system|assistant|developer)\s*:", re.MULTILINE)),
)


def _normalise(text: str) -> tuple[str, str]:
    """Return (collapsed, with_newlines): NFKC, zero-width removed, lower-cased."""
    base = unicodedata.normalize("NFKC", text).translate(_ZERO_WIDTH).casefold()
    return re.sub(r"\s+", " ", base).strip(), base


def check_input(question: str) -> InputVerdict:
    """Decide whether a question looks like an attack instead of a document question."""
    collapsed, with_lines = _normalise(question)
    for category, name, pattern in _MARKER_RULES:
        if pattern.search(with_lines):
            return InputVerdict(True, category, name)
    for text in (collapsed, collapsed.translate(_LEET)):  # second pass defeats 1gn0re-style spelling
        for category, name, pattern in _RULES:
            if pattern.search(text):
                return InputVerdict(True, category, name)
    return InputVerdict(False)

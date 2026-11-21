"""Safety checks. Output checks live here now; input and upload checks are added later."""
import re

# Shapes of credentials that must never appear in an answer.
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

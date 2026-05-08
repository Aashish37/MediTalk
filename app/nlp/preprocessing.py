import re


_WHITESPACE_RE = re.compile(r"\s+")


def messages_to_transcript(messages: list[dict]) -> str:
    lines = []
    for message in messages:
        raw_speaker = message.get("speaker", "unknown")
        speaker = getattr(raw_speaker, "value", str(raw_speaker)).title()
        text = clean_text(str(message.get("text", "")))
        if text:
            lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def clean_text(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text.replace("\x00", " ")).strip()


def sentence_split(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", clean_text(text))
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z\-']+", text.lower())

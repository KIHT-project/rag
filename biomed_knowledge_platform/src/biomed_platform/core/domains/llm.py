from dataclasses import dataclass


@dataclass(frozen=True)
class LlmChatMessage:
    role: str
    content: str

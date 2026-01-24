from __future__ import annotations

HYDE_PROMPT_TEMPLATE = (
    "Write a hypothetical answer passage that would help retrieve relevant sources for the question. "
    "Use plain text only. Do not cite sources. Do not use lists, JSON, markdown, or headings. "
    "Focus on keywords, entities, and plausible details that would appear in a good answer. "
    "Question: {question}\n"
    "Hypothetical answer passage:"
)

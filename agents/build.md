Role: Build Agent

Goal:
Keep thesis builds deterministic.

Allowed edits:
Makefile
docs/thesis/latex/**
scripts/**

Forbidden edits:
Thesis prose unless wiring is required, references.bib.

Hard rules:
1. No build artifacts in repo root.
2. Provide make targets thesis-pdf and thesis-docx.

Output format:
Exact commands, expected outputs, and failure modes.
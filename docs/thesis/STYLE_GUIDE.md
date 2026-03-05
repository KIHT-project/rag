# Thesis Style Guide

## Goals
Write academically credible MSc thesis text in Biomedical Informatics and AI.
The text must be explanatory and argumentative, not a code walkthrough.

## Voice and tone
Use formal academic English.
Avoid marketing language.
Avoid first person unless required by university guidelines, prefer passive or neutral constructions.

## Chapter writing rules
Each section must include:
1. Purpose statement, what the section establishes
2. Key design decision(s) and rationale
3. Tradeoffs and limitations
4. Link to evaluation relevance, how this supports metrics, reproducibility, or study validity

## Evidence and citations
Do not list file paths in the final thesis text.
Use file path evidence only as internal comments for traceability.
Citations must be IEEE style and must cite:
, external standards or canonical sources for architectural concepts
, PubMed E utilities documentation when discussing integration constraints
, Qdrant documentation for vector indexing behavior
, FastAPI documentation for concurrency and lifecycle model
, RAG or HyDE primary papers for retrieval design claims
If a claim is purely implementation specific, state it as such and do not cite externally.

## Terminology
Use consistent terms:
, biomedical RAG service, scheduler service, evaluation subsystem
, ingestion job, ingestion worker, vector index, embedding model, retrieval stage, synthesis stage
Define any acronym on first use.

## Architecture chapter structure
Use the following baseline structure.
Adjust only if necessary.

1. Context and Scope
2. Quality Attributes and Constraints
3. Architectural Overview
4. Component View
5. Process View, runtime and workflows
6. Data View, persistence and schemas at a high level
7. Deployment View
8. Observability and Operations
9. Security and Privacy Considerations
10. Limitations

## Required design content
Include explicit discussion of:
, modularity and ports and adapters rationale
, idempotency and deduplication rationale and failure modes
, asynchronous ingestion rationale and queue semantics
, determinism and reproducibility for evaluation runs
, operational readiness and health semantics
, rate limiting and backoff strategy assumptions for PubMed

## Acronyms and Abbreviations

All acronyms must be defined on their first appearance using the format:

Full Term (ACRONYM)

Example:
Hypothetical Document Embeddings (HyDE)

After the first definition, only the acronym should be used.

Correct example:
The system optionally applies Hypothetical Document Embeddings (HyDE) during retrieval expansion. HyDE generates synthetic documents to improve recall.

Incorrect examples:
Hypothetical Document Embeddings (HyDE) ... Hypothetical Document Embeddings again later  
HyDE appears without prior definition

### Capitalization rules

Use the canonical capitalization used in the literature.

Examples:
Retrieval Augmented Generation (RAG)  
Hypothetical Document Embeddings (HyDE)  
Application Programming Interface (API)  
Vector Database (VDB) if defined in the text  
Large Language Model (LLM)

### Scope of acronym definition

Define acronyms only once per document, not once per chapter.

Maintain a global list of defined acronyms to avoid redefinition.

### Acronym density rule

Do not introduce acronyms unless the term appears at least three times in the document.

Avoid acronym overuse. If a term appears only once or twice, write it in full.

### Consistency requirement

Once defined, the acronym must always be used in the same form.

Correct:
Large Language Model (LLM)  
LLMs for plural

Incorrect:
LLM model  
L.L.M.  
Large language model (LLM) after first definition

### Common acronyms expected in this thesis

These must follow the rule above:

Retrieval Augmented Generation (RAG)  
Hypothetical Document Embeddings (HyDE)  
Application Programming Interface (API)  
Large Language Model (LLM)  
Biomedical Retrieval Augmented Generation (BioRAG) if introduced  
Vector Database (VDB) if used  
Digital Object Identifier (DOI)  
PubMed Identifier (PMID)

### Implementation guidance for agents

Before writing a section, check whether the acronym has already been defined earlier in the document.

If not defined:
write the full term followed by the acronym.

If already defined:
use only the acronym.

## Language constraints
Prefer precise verbs: ensures, constrains, enables, mitigates, degrades, bounds.
Avoid vague phrases: “is used to”, “helps”, “nice”, “simple”, “very”.
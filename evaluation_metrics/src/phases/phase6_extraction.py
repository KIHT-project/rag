from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from evaluation_metrics.src.utils.output_writer import write_outputs

log = logging.getLogger(__name__)


DEFAULT_CANONICAL_FACTORS: dict[str, list[str]] = {
    "prior_vte": [
        "prior vte",
        "previous vte",
        "history of vte",
        "vte history",
        "venous thromboembolism history",
        "prior dvt",
        "history of dvt",
        "deep vein thrombosis history",
        "prior pulmonary embolism",
        "history of pulmonary embolism",
        "previous thrombosis",
    ],
    "age": [
        "advanced age",
        "older age",
        "elderly",
        "aging",
    ],
    "obesity": [
        "obesity",
        "obese",
        "body mass index",
        "bmi",
    ],
    "cancer": [
        "cancer",
        "malignancy",
        "neoplasm",
        "tumor",
    ],
    "immobility": [
        "immobility",
        "immobilization",
        "bed rest",
        "prolonged inactivity",
        "long term braking",
        "long term break",
        "long term bed rest",
    ],
    "surgery": [
        "surgery",
        "surgical",
        "postoperative",
        "operation",
        "prolonged anesthesia",
        "anesthesia duration",
        "blood transfusion",
        "transfusion",
    ],
    "trauma": [
        "trauma",
        "injury",
        "fracture",
    ],
    "pregnancy_postpartum": [
        "pregnancy",
        "pregnant",
        "postpartum",
        "puerperium",
    ],
    "hormone_therapy": [
        "oral contraceptive",
        "oestrogen",
        "estrogen",
        "hormone therapy",
        "hormonal therapy",
    ],
    "thrombophilia": [
        "thrombophilia",
        "inherited thrombophilia",
        "factor v leiden",
        "prothrombin mutation",
        "protein c deficiency",
        "protein s deficiency",
        "antithrombin deficiency",
        "antiphospholipid syndrome",
    ],
    "smoking": [
        "smoking",
        "smoker",
        "tobacco use",
    ],
    "hypertension": [
        "hypertension",
        "high blood pressure",
    ],
    "diabetes": [
        "diabetes",
        "diabetes mellitus",
        "hyperglycemia",
        "dm",
    ],
    "infection": [
        "infection",
        "sepsis",
        "inflammatory state",
        "inflammation",
        "covid",
        "covid 19",
        "sars cov 2",
    ],
    "heart_failure": [
        "heart failure",
        "cardiac failure",
        "reduced ejection fraction",
    ],
    "atrial_fibrillation": [
        "atrial fibrillation",
        "afib",
        "af",
    ],
    "ckd": [
        "chronic kidney disease",
        "renal insufficiency",
        "kidney failure",
        "hemodialysis",
        "haemodialysis",
    ],
    "catheter": [
        "catheter",
        "central venous catheter",
        "venous access",
        "line related",
    ],
    "catheter_occlusion": [
        "occlusion",
        "catheter occlusion",
        "line occlusion",
    ],
    "catheter_infiltration": [
        "infiltration",
        "catheter infiltration",
        "extravasation",
    ],
    "thrombophlebitis": [
        "thrombophlebitis",
        "phlebitis",
    ],
    "catheter_dislodgement": [
        "dislodgement",
        "catheter dislodgement",
        "line dislodgement",
    ],
    "anticoagulation": [
        "anticoagulation",
        "anticoagulant",
        "heparin",
        "low molecular weight heparin",
        "lmwh",
        "doac",
        "warfarin",
        "rivaroxaban",
        "enoxaparin",
        "aspirin",
        "thromboprophylaxis",
        "dvt prophylaxis",
    ],
    "catheter_directed_thrombolysis": [
        "catheter directed thrombolysis",
        "cdt",
        "scdt",
    ],
    "ultrasound_assisted_thrombolysis": [
        "ultrasound assisted catheter directed thrombolysis",
        "ultrasound assisted thrombolysis",
        "usat",
    ],
    "systemic_thrombolysis": [
        "systemic thrombolysis",
        "thrombolytic therapy",
        "st",
    ],
    "major_bleeding": [
        "major bleeding",
        "bleeding risk",
        "hemorrhage",
        "haemorrhage",
    ],
    "mrbti": [
        "mrbti",
        "mr black blood thrombus imaging",
        "black blood thrombus imaging",
    ],
    "ce_mri": [
        "ce mri",
        "contrast enhanced mri",
        "contrast enhanced mr imaging",
    ],
    "nce_mrv": [
        "nce mrv",
        "noncontrast enhanced mr venography",
        "non contrast enhanced mr venography",
        "noncontrast mr venography",
        "non contrast mr venography",
    ],
    "liver_cirrhosis": [
        "liver cirrhosis",
        "cirrhosis",
    ],
    "portal_vein_resection": [
        "portal vein resection",
    ],
    "right_hepatectomy": [
        "right sided hepatectomy",
        "right hepatectomy",
    ],
    "thromboembolism": [
        "thromboembolic events",
        "thromboembolism",
        "venous thromboembolism",
    ],
    "neutropenia": [
        "neutropenia",
        "neutropenic",
    ],
    "ck_elevation": [
        "ck elevation",
        "creatine kinase elevation",
        "creatine kinase",
    ],
    "cardiovascular_risk": [
        "cardiovascular risk",
    ],
}


def _norm_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    s = value.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _norm_doi(value: Any) -> str:
    s = _norm_text(value)
    if not s:
        return ""
    for prefix in ("https doi org ", "http doi org ", "http dx doi org ", "doi "):
        if s.startswith(prefix):
            s = s[len(prefix) :].strip()
    return s


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        if s.isdigit():
            return int(s)
    return None


def _extract_choice(results: Sequence[dict[str, Any]], field: str) -> str:
    for r in results:
        if r.get("from_name") != field:
            continue
        if r.get("type") != "choices":
            continue
        value = r.get("value")
        if not isinstance(value, dict):
            continue
        choices = value.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        first = choices[0]
        if isinstance(first, str):
            return first.strip()
    return ""


def _extract_rating(results: Sequence[dict[str, Any]], field: str) -> int | None:
    for r in results:
        if r.get("from_name") != field:
            continue
        if r.get("type") != "rating":
            continue
        value = r.get("value")
        if not isinstance(value, dict):
            continue
        return _safe_int(value.get("rating"))
    return None


def _extract_reason_texts(results: Sequence[dict[str, Any]], field: str) -> list[str]:
    out: list[str] = []
    for r in results:
        if r.get("from_name") != field:
            continue
        value = r.get("value")
        if not isinstance(value, dict):
            continue
        text = value.get("text")
        if isinstance(text, str) and text.strip():
            out.append(text.strip())
            continue
        if isinstance(text, list):
            for item in text:
                if isinstance(item, str) and item.strip():
                    out.append(item.strip())
    return out


def _build_alias_index(
    factors: Mapping[str, Sequence[str]],
) -> tuple[dict[str, str], dict[str, re.Pattern[str]]]:
    alias_to_canonical: dict[str, str] = {}
    patterns: dict[str, re.Pattern[str]] = {}

    for canonical_raw, aliases_raw in factors.items():
        canonical = _norm_text(canonical_raw)
        if not canonical:
            continue

        aliases: set[str] = {canonical}
        for a in aliases_raw:
            alias = _norm_text(a)
            if alias:
                aliases.add(alias)

        for alias in aliases:
            alias_to_canonical[alias] = canonical

        aliases_sorted = sorted(aliases, key=len, reverse=True)
        joined = "|".join([re.escape(a) for a in aliases_sorted if a])
        if not joined:
            continue
        patterns[canonical] = re.compile(rf"(?<![a-z0-9])(?:{joined})(?![a-z0-9])")

    return alias_to_canonical, patterns


def _canonicalize_factor_name(
    value: str,
    *,
    alias_to_canonical: Mapping[str, str],
    patterns: Mapping[str, re.Pattern[str]],
) -> str:
    norm = _norm_text(value)
    if not norm:
        return ""
    direct = alias_to_canonical.get(norm)
    if direct:
        return direct

    for canonical, pattern in patterns.items():
        if pattern.search(norm):
            return canonical

    return norm


def _extract_factors_from_text(
    *,
    text: str,
    patterns: Mapping[str, re.Pattern[str]],
) -> set[str]:
    norm = _norm_text(text)
    if not norm:
        return set()
    out: set[str] = set()
    for canonical, pattern in patterns.items():
        if pattern.search(norm):
            out.add(canonical)
    return out


def _load_doc_factor_map(
    *,
    tasks_clean_json: Path,
    reports_risk_field: str,
    reports_positive_value: str,
    reports_confidence_field: str,
    min_confidence: int,
    allow_missing_confidence: bool,
    factor_source_field: str,
    factor_source_fallback_to_abstract: bool,
    patterns: Mapping[str, re.Pattern[str]],
) -> tuple[dict[str, set[str]], dict[str, int]]:
    raw = json.loads(tasks_clean_json.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise RuntimeError("tasks_clean must be a JSON array")

    positive_docs = 0
    kept_docs = 0
    docs_with_factor_hits = 0

    doi_to_factors: dict[str, set[str]] = {}
    positive_value_norm = _norm_text(reports_positive_value)

    for task in raw:
        if not isinstance(task, dict):
            continue

        data = task.get("data")
        if not isinstance(data, dict):
            continue

        doi = _norm_doi(data.get("doi"))
        if not doi:
            continue

        annotations = task.get("annotations")
        if not isinstance(annotations, list) or not annotations:
            continue
        ann0 = annotations[0]
        if not isinstance(ann0, dict):
            continue
        results = ann0.get("result")
        if not isinstance(results, list):
            continue

        choice = _extract_choice(results, reports_risk_field)
        if _norm_text(choice) != positive_value_norm:
            continue
        positive_docs += 1

        confidence = _extract_rating(results, reports_confidence_field)
        if confidence is None and not allow_missing_confidence:
            continue
        if confidence is not None and int(confidence) < int(min_confidence):
            continue
        kept_docs += 1

        evidence_parts = _extract_reason_texts(results, factor_source_field)
        abstract_text = ""
        abstract = data.get("abstract")
        if isinstance(abstract, str) and abstract.strip():
            abstract_text = abstract.strip()

        if not evidence_parts and factor_source_fallback_to_abstract and abstract_text:
            evidence_parts.append(abstract_text)

        evidence_text = "\n".join(evidence_parts)
        factors = _extract_factors_from_text(text=evidence_text, patterns=patterns)
        if (
            not factors
            and factor_source_fallback_to_abstract
            and abstract_text
            and abstract_text not in evidence_parts
        ):
            # When reason evidence exists but has no canonical hits, use abstract as a fallback source.
            factors = _extract_factors_from_text(text=abstract_text, patterns=patterns)
        if factors:
            docs_with_factor_hits += 1

        if doi not in doi_to_factors:
            doi_to_factors[doi] = set()
        doi_to_factors[doi].update(factors)

    stats = {
        "positive_docs": positive_docs,
        "kept_docs": kept_docs,
        "docs_with_factor_hits": docs_with_factor_hits,
        "unique_kept_dois": len(doi_to_factors),
    }
    return doi_to_factors, stats


def _extract_predicted_risk_factor_candidates(
    raw_answer: Any, *, include_aliases: bool = True
) -> list[str]:
    if not isinstance(raw_answer, dict):
        return []

    answer = raw_answer.get("answer")
    if not isinstance(answer, dict):
        return []

    risk_factors = answer.get("risk_factors")
    if not isinstance(risk_factors, list):
        return []

    out: list[str] = []
    for rf in risk_factors:
        if not isinstance(rf, dict):
            continue
        normalized_name = rf.get("normalized_name")
        if isinstance(normalized_name, str) and normalized_name.strip():
            out.append(normalized_name.strip())
        if include_aliases:
            aliases = rf.get("aliases")
            if isinstance(aliases, list):
                for alias in aliases:
                    if isinstance(alias, str) and alias.strip():
                        out.append(alias.strip())
    return out


def _extract_answer_summary(raw_answer: Any) -> str:
    if not isinstance(raw_answer, dict):
        return ""
    answer = raw_answer.get("answer")
    if not isinstance(answer, dict):
        return ""
    summary = answer.get("summary")
    return summary.strip() if isinstance(summary, str) else ""


def _extract_citation_snippets(raw_answer: Any) -> list[str]:
    if not isinstance(raw_answer, dict):
        return []
    citations = raw_answer.get("citations")
    if not isinstance(citations, list):
        return []
    out: list[str] = []
    for c in citations:
        if not isinstance(c, dict):
            continue
        for key in ("snippet", "text", "content"):
            value = c.get(key)
            if isinstance(value, str) and value.strip():
                out.append(value.strip())
                break
    return out


def _extract_cited_dois(raw_answer: Any) -> list[str]:
    if not isinstance(raw_answer, dict):
        return []

    citations = raw_answer.get("citations")
    if not isinstance(citations, list):
        return []

    out: list[str] = []
    seen: set[str] = set()
    for c in citations:
        if not isinstance(c, dict):
            continue
        doi = _norm_doi(c.get("doi"))
        if not doi or doi in seen:
            continue
        seen.add(doi)
        out.append(doi)
    return out


def _prf(*, tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = (float(tp) / float(tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (float(tp) / float(tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _infer_mode_from_input(path: Path) -> str:
    stem = path.stem.lower()
    if "rag_no_hyde" in stem:
        return "rag_no_hyde"
    if "rag_hyde" in stem:
        return "rag_hyde"
    if "llm_only" in stem:
        return "llm_only"
    return "unknown"


def run_phase6_extraction(
    *,
    input_jsonl: Path,
    tasks_clean_json: Path,
    out_dir: Path,
    reports_risk_field: str = "reports_risk_factors",
    reports_positive_value: str = "Yes",
    reports_confidence_field: str = "confidence_reports_risk_factors",
    min_confidence: int = 3,
    allow_missing_confidence: bool = True,
    factor_source_field: str = "reason_label",
    factor_source_fallback_to_abstract: bool = True,
    no_gold_policy: str = "skip",
    pred_closed_set_only: bool = False,
    pred_include_aliases: bool = True,
    pred_include_summary_factors: bool = False,
    pred_include_citation_snippets: bool = False,
    canonical_factors: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, float]:
    if no_gold_policy not in {"skip", "empty"}:
        raise ValueError("no_gold_policy must be one of: skip, empty")

    factors = canonical_factors if canonical_factors is not None else DEFAULT_CANONICAL_FACTORS
    alias_to_canonical, patterns = _build_alias_index(factors)
    canonical_vocabulary = set(patterns.keys())

    doi_to_gold_factors, label_stats = _load_doc_factor_map(
        tasks_clean_json=tasks_clean_json,
        reports_risk_field=reports_risk_field,
        reports_positive_value=reports_positive_value,
        reports_confidence_field=reports_confidence_field,
        min_confidence=min_confidence,
        allow_missing_confidence=allow_missing_confidence,
        factor_source_field=factor_source_field,
        factor_source_fallback_to_abstract=factor_source_fallback_to_abstract,
        patterns=patterns,
    )

    total_queries = 0
    skipped_no_gold = 0
    mode_from_path = _infer_mode_from_input(input_jsonl)
    rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []

    with input_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            rec = json.loads(line)
            if not isinstance(rec, dict):
                continue
            total_queries += 1

            query_id = str(rec.get("query_id") or "").strip() or f"row_{total_queries}"
            mode = str(rec.get("mode") or "").strip() or mode_from_path

            raw_answer = rec.get("answer_raw")
            pred_candidates = _extract_predicted_risk_factor_candidates(
                raw_answer,
                include_aliases=bool(pred_include_aliases),
            )
            pred_factors: set[str] = set()
            for candidate in pred_candidates:
                canon = _canonicalize_factor_name(
                    candidate,
                    alias_to_canonical=alias_to_canonical,
                    patterns=patterns,
                )
                if not canon:
                    continue
                if pred_closed_set_only and canon not in canonical_vocabulary:
                    continue
                pred_factors.add(canon)

            if pred_include_summary_factors:
                summary_text = _extract_answer_summary(raw_answer)
                if summary_text:
                    pred_factors.update(
                        _extract_factors_from_text(text=summary_text, patterns=patterns)
                    )

            if pred_include_citation_snippets:
                for snippet in _extract_citation_snippets(raw_answer):
                    pred_factors.update(_extract_factors_from_text(text=snippet, patterns=patterns))

            cited_dois = _extract_cited_dois(raw_answer)
            gold_factors: set[str] = set()
            gold_dois_used: list[str] = []
            for doi in cited_dois:
                factors_for_doi = doi_to_gold_factors.get(doi)
                if factors_for_doi is None:
                    continue
                gold_factors.update(factors_for_doi)
                gold_dois_used.append(doi)

            if not gold_factors and no_gold_policy == "skip":
                skipped_no_gold += 1
                excluded_rows.append(
                    {
                        "query_id": query_id,
                        "mode": mode,
                        "reason": "no_gold_factors",
                        "cited_doi_count": len(cited_dois),
                        "gold_doi_count": len(gold_dois_used),
                    }
                )
                continue

            tp = len(pred_factors.intersection(gold_factors))
            fp = len(pred_factors.difference(gold_factors))
            fn = len(gold_factors.difference(pred_factors))
            precision, recall, f1 = _prf(tp=tp, fp=fp, fn=fn)

            rows.append(
                {
                    "query_id": query_id,
                    "mode": mode,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "pred_factor_count": len(pred_factors),
                    "gold_factor_count": len(gold_factors),
                    "cited_doi_count": len(cited_dois),
                    "gold_doi_count": len(gold_dois_used),
                    "pred_factors": "|".join(sorted(pred_factors)),
                    "gold_factors": "|".join(sorted(gold_factors)),
                    "matched_factors": "|".join(sorted(pred_factors.intersection(gold_factors))),
                }
            )

    per_query_df = pd.DataFrame(rows)
    if not per_query_df.empty:
        per_query_df = per_query_df.sort_values(["query_id"]).reset_index(drop=True)
    else:
        per_query_df = pd.DataFrame(
            columns=[
                "query_id",
                "mode",
                "tp",
                "fp",
                "fn",
                "precision",
                "recall",
                "f1",
                "pred_factor_count",
                "gold_factor_count",
                "cited_doi_count",
                "gold_doi_count",
                "pred_factors",
                "gold_factors",
                "matched_factors",
            ]
        )

    tp_total = int(per_query_df["tp"].sum()) if not per_query_df.empty else 0
    fp_total = int(per_query_df["fp"].sum()) if not per_query_df.empty else 0
    fn_total = int(per_query_df["fn"].sum()) if not per_query_df.empty else 0
    micro_precision, micro_recall, micro_f1 = _prf(tp=tp_total, fp=fp_total, fn=fn_total)

    macro_precision = float(per_query_df["precision"].mean()) if not per_query_df.empty else 0.0
    macro_recall = float(per_query_df["recall"].mean()) if not per_query_df.empty else 0.0
    macro_f1 = float(per_query_df["f1"].mean()) if not per_query_df.empty else 0.0

    evaluated_queries = int(len(per_query_df))
    query_coverage_rate = (
        float(evaluated_queries) / float(total_queries) if total_queries > 0 else 0.0
    )
    skipped_no_gold_rate = (
        float(skipped_no_gold) / float(total_queries) if total_queries > 0 else 0.0
    )

    summary_row = {
        "mode": mode_from_path,
        "total_queries": total_queries,
        "evaluated_queries": evaluated_queries,
        "skipped_no_gold": skipped_no_gold,
        "query_coverage_rate": query_coverage_rate,
        "skipped_no_gold_rate": skipped_no_gold_rate,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "tp_total": tp_total,
        "fp_total": fp_total,
        "fn_total": fn_total,
        "label_positive_docs": int(label_stats["positive_docs"]),
        "label_kept_docs": int(label_stats["kept_docs"]),
        "label_docs_with_factor_hits": int(label_stats["docs_with_factor_hits"]),
        "label_unique_kept_dois": int(label_stats["unique_kept_dois"]),
        "pred_closed_set_only": bool(pred_closed_set_only),
        "pred_include_aliases": bool(pred_include_aliases),
        "pred_include_summary_factors": bool(pred_include_summary_factors),
        "pred_include_citation_snippets": bool(pred_include_citation_snippets),
    }
    summary_df = pd.DataFrame([summary_row])

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = input_jsonl.stem
    per_query_csv = out_dir / f"phase6_extraction_{stem}_per_query.csv"
    summary_csv = out_dir / f"phase6_extraction_{stem}_summary.csv"
    write_outputs(per_query_df, per_query_csv)
    write_outputs(summary_df, summary_csv)

    if excluded_rows:
        excluded_df = pd.DataFrame(excluded_rows).sort_values(["query_id"]).reset_index(drop=True)
        excluded_csv = out_dir / f"phase6_extraction_{stem}_excluded.csv"
        write_outputs(excluded_df, excluded_csv)

    log.info(
        "phase6 extraction done | input=%s | total=%d | evaluated=%d | micro_f1=%.4f | macro_f1=%.4f",
        input_jsonl,
        total_queries,
        evaluated_queries,
        micro_f1,
        macro_f1,
    )

    return {
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "query_coverage_rate": query_coverage_rate,
        "evaluated_queries": float(evaluated_queries),
        "skipped_no_gold": float(skipped_no_gold),
    }

import hashlib
from copy import deepcopy
from typing import Any, Iterable

import pytest

from biomed_platform.api.models.generated import schemas
from biomed_platform.api.models.generated.schemas import IngestItem
from biomed_platform.common.utils import (
    canonicalize_doi,
    compute_body_hash_from_items,
    compute_doc_id,
    normalize_doi,
)


class TestIngestHashingUtils:
    def _any_disease_value(self) -> Any:
        disease_enum = getattr(schemas, "Disease", None)
        if disease_enum is None:
            raise AssertionError("schemas.Disease is missing, update test to match generated schemas")
        return next(iter(disease_enum)).value

    def _any_source_type_value(self) -> Any:
        source_type_enum = getattr(schemas, "SourceType", None)
        if source_type_enum is None:
            raise AssertionError("schemas.SourceType is missing, update test to match generated schemas")
        return next(iter(source_type_enum)).value

    def _another_enum_value(self, enum_cls: Any, current_value: Any) -> Any | None:
        values = [m.value for m in enum_cls]
        for v in values:
            if v != current_value:
                return v
        return None

    def _author(self, name: str) -> Any:
        author_cls = getattr(schemas, "Author", None)
        if author_cls is None:
            raise AssertionError("schemas.Author is missing, update test to match generated schemas")
        return author_cls(root=name)

    def _base_item_payload(self) -> dict[str, Any]:
        doi = "10.1000/abc.def"
        return {
            "doi": doi,
            "disease": self._any_disease_value(),
            "year": 2024,
            "source_type": self._any_source_type_value(),
            "title": "A title",
            "journal": "A journal",
            "authors": [self._author("Alice"), self._author("Bob")],
            "content_text": "Some content",
        }

    def _make_item_validated(self, **overrides: Any) -> IngestItem:
        payload = {**self._base_item_payload(), **overrides}
        return IngestItem(**payload)

    def _make_item_for_hashing(self, **overrides: Any) -> IngestItem:
        payload = {**self._base_item_payload(), **overrides}
        doi_norm = normalize_doi(payload.get("doi", ""))
        item = self._make_item_validated(**payload)
        return item.model_copy(update={"doi_normalized": doi_norm})

    def test_canonicalize_doi_given_empty_when_canonicalize_then_returns_empty(self) -> None:
        given_raw = "   "

        when_canon = canonicalize_doi(given_raw)

        then_expected = ""
        assert when_canon == then_expected

    def test_canonicalize_doi_given_missing_pattern_when_canonicalize_then_returns_empty(self) -> None:
        given_raw = "not a doi"

        when_canon = canonicalize_doi(given_raw)

        then_expected = ""
        assert when_canon == then_expected

    def test_canonicalize_doi_given_doi_with_spaces_and_uppercase_when_canonicalize_then_extract_trim_and_lowercase(
        self,
    ) -> None:
        given_raw = " 10.1000/ABC.Def  "

        when_canon = canonicalize_doi(given_raw)

        then_expected = "10.1000/abc.def"
        assert when_canon == then_expected

    @pytest.mark.parametrize(
        "given_raw,then_expected",
        [
            ("doi:10.1000/ABC.Def", "10.1000/abc.def"),
            ("https://doi.org/10.1000/ABC.Def", "10.1000/abc.def"),
            ("DOI 10.1000/ABC.Def.", "10.1000/abc.def"),
            ("(10.1000/ABC.Def)", "10.1000/abc.def"),
            ('"10.1000/ABC.Def"', "10.1000/abc.def"),
            ("10.1000/ABC.Def];", "10.1000/abc.def"),
        ],
    )
    def test_canonicalize_doi_given_wrapped_or_trailed_when_canonicalize_then_extract_and_strip_punctuation(
        self,
        given_raw: str,
        then_expected: str,
    ) -> None:
        when_canon = canonicalize_doi(given_raw)

        assert when_canon == then_expected

    def test_normalize_doi_given_raw_when_normalize_then_matches_canonicalize(self) -> None:
        given_raw = " https://doi.org/10.1000/ABC.Def "

        when_norm = normalize_doi(given_raw)

        then_expected = canonicalize_doi(given_raw)
        assert when_norm == then_expected
        assert when_norm == "10.1000/abc.def"

    def test_compute_doc_id_given_normalized_doi_when_compute_then_returns_sha256_hex(self) -> None:
        given_doi_normalized = "10.1000/abc.def"

        when_doc_id = compute_doc_id(doi_normalized=given_doi_normalized)

        then_expected = hashlib.sha256(given_doi_normalized.encode("utf-8")).hexdigest()
        assert when_doc_id == then_expected
        assert len(when_doc_id) == 64

    def test_compute_body_hash_from_items_given_same_inputs_when_compute_twice_then_hash_is_stable(self) -> None:
        given_effective_embedding_model_id = "e5_large"
        given_items = [self._make_item_for_hashing()]

        when_hash_1 = compute_body_hash_from_items(
            effective_embedding_model_id=given_effective_embedding_model_id,
            items=given_items,
        )
        when_hash_2 = compute_body_hash_from_items(
            effective_embedding_model_id=given_effective_embedding_model_id,
            items=given_items,
        )

        then_hash_is_stable = when_hash_1 == when_hash_2
        assert then_hash_is_stable
        assert len(when_hash_1) == 64

    def test_compute_body_hash_from_items_given_optional_like_fields_when_compute_then_is_stable(self) -> None:
        given_effective_embedding_model_id = ""
        given_items_1 = [
            self._make_item_for_hashing(
                year=None,
                title="",
                journal="",
                authors=[],
                content_text="x",
            )
        ]
        given_items_2 = [
            self._make_item_for_hashing(
                year=None,
                title="",
                journal="",
                authors=[],
                content_text="x",
            )
        ]

        when_hash_1 = compute_body_hash_from_items(
            effective_embedding_model_id=given_effective_embedding_model_id,
            items=given_items_1,
        )
        when_hash_2 = compute_body_hash_from_items(
            effective_embedding_model_id=given_effective_embedding_model_id,
            items=given_items_2,
        )

        then_expected_same = when_hash_1 == when_hash_2
        assert then_expected_same

    def test_compute_body_hash_from_items_given_author_whitespace_and_empty_entries_when_compute_then_ignores_noise(
        self,
    ) -> None:
        given_effective_embedding_model_id = "e5_large"
        given_base = self._make_item_for_hashing(authors=[self._author("Alice"), self._author("Bob")])
        given_noisy = self._make_item_for_hashing(
            authors=[self._author(" Alice "), self._author(""), self._author("   "), self._author("Bob")]
        )

        when_hash_base = compute_body_hash_from_items(
            effective_embedding_model_id=given_effective_embedding_model_id,
            items=[given_base],
        )
        when_hash_noisy = compute_body_hash_from_items(
            effective_embedding_model_id=given_effective_embedding_model_id,
            items=[given_noisy],
        )

        then_expected_equal = when_hash_base == when_hash_noisy
        assert then_expected_equal

    def test_compute_body_hash_from_items_given_two_items_when_item_order_changes_then_hash_changes(self) -> None:
        given_effective_embedding_model_id = "e5"
        given_item_1 = self._make_item_for_hashing(
            doi="10.1000/aaa",
            title="T1",
            journal="J1",
            authors=[self._author("A")],
            content_text="C1",
        )
        given_item_2 = self._make_item_for_hashing(
            doi="10.1000/bbb",
            title="T2",
            journal="J2",
            authors=[self._author("B")],
            content_text="C2",
        )

        when_hash_1 = compute_body_hash_from_items(
            effective_embedding_model_id=given_effective_embedding_model_id,
            items=[given_item_1, given_item_2],
        )
        when_hash_2 = compute_body_hash_from_items(
            effective_embedding_model_id=given_effective_embedding_model_id,
            items=[given_item_2, given_item_1],
        )

        then_expected_different = when_hash_1 != when_hash_2
        assert then_expected_different

    @pytest.mark.parametrize(
        "change,new_value_factory",
        [
            ("effective_embedding_model_id", lambda self, it: "other"),
            ("doi", lambda self, it: "10.1000/changed"),
            ("year", lambda self, it: 2023),
            ("title", lambda self, it: "New title"),
            ("journal", lambda self, it: "New journal"),
            ("authors", lambda self, it: [self._author("Alice"), self._author("Eve")]),
            ("content_text", lambda self, it: "Changed content"),
        ],
    )
    def test_compute_body_hash_from_items_given_base_when_component_changes_then_hash_changes(
        self,
        change: str,
        new_value_factory,
    ) -> None:
        given_effective_embedding_model_id = "e5_large"
        given_item = self._make_item_for_hashing()

        given_base_hash = compute_body_hash_from_items(
            effective_embedding_model_id=given_effective_embedding_model_id,
            items=[given_item],
        )

        changed_effective_embedding_model_id = given_effective_embedding_model_id
        changed_payload = deepcopy(given_item.model_dump())

        if change == "effective_embedding_model_id":
            changed_effective_embedding_model_id = new_value_factory(self, given_item)
        else:
            changed_payload[change] = new_value_factory(self, given_item)

        if change == "doi":
            changed_payload["doi_normalized"] = normalize_doi(changed_payload["doi"])

        changed_item = IngestItem(**{k: v for k, v in changed_payload.items() if k != "doi_normalized"})
        changed_item = changed_item.model_copy(update={"doi_normalized": changed_payload.get("doi_normalized", "")})

        when_changed_hash = compute_body_hash_from_items(
            effective_embedding_model_id=changed_effective_embedding_model_id,
            items=[changed_item],
        )

        then_expected_different = given_base_hash != when_changed_hash
        assert then_expected_different

    def test_compute_body_hash_from_items_given_source_type_enum_has_alternative_when_changed_then_hash_changes(
        self,
    ) -> None:
        given_enum = getattr(schemas, "SourceType", None)
        assert given_enum is not None

        given_base = self._make_item_for_hashing()
        given_current = given_base.source_type.value if hasattr(given_base.source_type, "value") else given_base.source_type
        given_other = self._another_enum_value(given_enum, given_current)

        if given_other is None:
            pytest.skip("SourceType enum has only one value, cannot test change sensitivity")

        given_changed = self._make_item_for_hashing(source_type=given_other)

        when_hash_base = compute_body_hash_from_items(effective_embedding_model_id="e5_large", items=[given_base])
        when_hash_changed = compute_body_hash_from_items(effective_embedding_model_id="e5_large", items=[given_changed])

        then_expected_different = when_hash_base != when_hash_changed
        assert then_expected_different

    def test_compute_body_hash_from_items_given_disease_enum_has_alternative_when_changed_then_hash_changes(
        self,
    ) -> None:
        given_enum = getattr(schemas, "Disease", None)
        assert given_enum is not None

        given_base = self._make_item_for_hashing()
        given_current = given_base.disease.value if hasattr(given_base.disease, "value") else given_base.disease
        given_other = self._another_enum_value(given_enum, given_current)

        if given_other is None:
            pytest.skip("Disease enum has only one value, cannot test change sensitivity")

        given_changed = self._make_item_for_hashing(disease=given_other)

        when_hash_base = compute_body_hash_from_items(effective_embedding_model_id="e5_large", items=[given_base])
        when_hash_changed = compute_body_hash_from_items(effective_embedding_model_id="e5_large", items=[given_changed])

        then_expected_different = when_hash_base != when_hash_changed
        assert then_expected_different

    def test_compute_body_hash_from_items_given_items_iterable_generator_when_compute_then_is_stable(self) -> None:
        given_effective_embedding_model_id = "e5_large"
        given_item = self._make_item_for_hashing()

        def given_items() -> Iterable[IngestItem]:
            yield given_item

        when_hash_1 = compute_body_hash_from_items(
            effective_embedding_model_id=given_effective_embedding_model_id,
            items=given_items(),
        )
        when_hash_2 = compute_body_hash_from_items(
            effective_embedding_model_id=given_effective_embedding_model_id,
            items=given_items(),
        )

        then_expected_equal = when_hash_1 == when_hash_2
        assert then_expected_equal

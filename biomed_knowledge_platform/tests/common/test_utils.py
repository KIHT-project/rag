import hashlib
from copy import deepcopy
from typing import Any

import pytest

from biomed_platform.api.models.generated import schemas
from biomed_platform.common.utils import compute_body_hash, compute_doc_id, normalize_doi


class TestIngestHashingUtils:
    def _any_disease_value(self) -> str:
        disease_enum = getattr(schemas, "Disease", None)
        if disease_enum is None:
            raise AssertionError("schemas.Disease is missing, update test to match generated schemas")
        return next(iter(disease_enum)).value

    def _any_source_type_value(self) -> str:
        source_type_enum = getattr(schemas, "SourceType", None)
        if source_type_enum is None:
            raise AssertionError("schemas.SourceType is missing, update test to match generated schemas")
        return next(iter(source_type_enum)).value

    def _another_enum_value(self, enum_cls: Any, current_value: str) -> str | None:
        values = [m.value for m in enum_cls]
        alternatives = [v for v in values if v != current_value]
        return alternatives[0] if alternatives else None

    def _make_request(
        self,
        *,
        embedding_model_id: str | None,
        items: list[dict[str, Any]],
    ) -> schemas.IngestBatchRequest:
        return schemas.IngestBatchRequest(embedding_model_id=embedding_model_id, items=items)

    def _base_item_dict(self) -> dict[str, Any]:
        return {
            "doi": "10.1000/abc.def",
            "disease": self._any_disease_value(),
            "year": 2024,
            "source_type": self._any_source_type_value(),
            "title": "A title",
            "journal": "A journal",
            "authors": ["Alice", "Bob"],
            "content_text": "Some content",
        }

    def test_normalize_doi_given_doi_with_spaces_and_uppercase_when_normalize_then_trim_and_lowercase(self) -> None:
        given_doi = " 10.1000/ABC.Def  "

        when_normalized = normalize_doi(given_doi)

        then_expected = "10.1000/abc.def"
        assert when_normalized == then_expected

    def test_compute_doc_id_given_normalized_doi_when_compute_then_returns_sha256_hex(self) -> None:
        given_doi_normalized = "10.1000/abc.def"

        when_doc_id = compute_doc_id(doi_normalized=given_doi_normalized)

        then_expected = hashlib.sha256(given_doi_normalized.encode("utf-8")).hexdigest()
        assert when_doc_id == then_expected
        assert len(when_doc_id) == 64

    def test_compute_body_hash_given_same_request_values_when_compute_twice_then_hash_is_stable(self) -> None:
        given_request = self._make_request(
            embedding_model_id="e5_large",
            items=[self._base_item_dict()],
        )

        when_hash_1 = compute_body_hash(given_request)
        when_hash_2 = compute_body_hash(given_request)

        assert when_hash_1 == when_hash_2
        assert len(when_hash_1) == 64

    def test_compute_body_hash_given_optional_fields_none_when_compute_then_is_stable(self) -> None:
        base_item = self._base_item_dict()

        given_request_none_1 = self._make_request(
            embedding_model_id=None,
            items=[
                {
                    **base_item,
                    "year": None,
                    "title": None,
                    "journal": None,
                    "authors": None,
                }
            ],
        )
        given_request_none_2 = self._make_request(
            embedding_model_id=None,
            items=[
                {
                    **base_item,
                    "year": None,
                    "title": None,
                    "journal": None,
                    "authors": None,
                }
            ],
        )

        when_hash_1 = compute_body_hash(given_request_none_1)
        when_hash_2 = compute_body_hash(given_request_none_2)

        assert when_hash_1 == when_hash_2

    def test_compute_body_hash_given_two_items_when_item_order_changes_then_hash_changes(self) -> None:
        item_1 = {
            **self._base_item_dict(),
            "doi": "10.1000/aaa",
            "title": "T1",
            "journal": "J1",
            "authors": ["A"],
            "content_text": "C1",
        }
        item_2 = {
            **self._base_item_dict(),
            "doi": "10.1000/bbb",
            "title": "T2",
            "journal": "J2",
            "authors": ["B"],
            "content_text": "C2",
        }

        given_request_1 = self._make_request(embedding_model_id="e5", items=[item_1, item_2])
        given_request_2 = self._make_request(embedding_model_id="e5", items=[item_2, item_1])

        when_hash_1 = compute_body_hash(given_request_1)
        when_hash_2 = compute_body_hash(given_request_2)

        assert when_hash_1 != when_hash_2

    @pytest.mark.parametrize(
        "field_path,new_value_factory",
        [
            (("embedding_model_id",), lambda self, req: "other"),
            (("items", 0, "doi"), lambda self, req: "10.1000/changed"),
            (("items", 0, "year"), lambda self, req: 2023),
            (("items", 0, "title"), lambda self, req: "New title"),
            (("items", 0, "journal"), lambda self, req: "New journal"),
            (("items", 0, "authors"), lambda self, req: ["Alice", "Eve"]),
            (("items", 0, "content_text"), lambda self, req: "Changed content"),
        ],
    )
    def test_compute_body_hash_given_base_request_when_component_changes_then_hash_changes(
        self,
        field_path: tuple[Any, ...],
        new_value_factory,
    ) -> None:
        given_base = self._make_request(
            embedding_model_id="e5_large",
            items=[self._base_item_dict()],
        )
        base_hash = compute_body_hash(given_base)

        changed_payload = deepcopy(given_base.model_dump())

        cursor: Any = changed_payload
        for key in field_path[:-1]:
            cursor = cursor[key]
        cursor[field_path[-1]] = new_value_factory(self, given_base)

        given_changed = schemas.IngestBatchRequest(**changed_payload)
        changed_hash = compute_body_hash(given_changed)

        assert base_hash != changed_hash

    def test_compute_body_hash_given_source_type_enum_has_alternative_when_changed_then_hash_changes(self) -> None:
        source_type_enum = getattr(schemas, "SourceType", None)
        assert source_type_enum is not None

        given_base = self._make_request(
            embedding_model_id="e5_large",
            items=[self._base_item_dict()],
        )

        current = given_base.items[0].source_type.value
        other = self._another_enum_value(source_type_enum, current)

        if other is None:
            pytest.skip("SourceType enum has only one value, cannot test change sensitivity")

        changed_payload = deepcopy(given_base.model_dump())
        changed_payload["items"][0]["source_type"] = other

        assert compute_body_hash(given_base) != compute_body_hash(schemas.IngestBatchRequest(**changed_payload))

    def test_compute_body_hash_given_disease_enum_has_alternative_when_changed_then_hash_changes(self) -> None:
        disease_enum = getattr(schemas, "Disease", None)
        assert disease_enum is not None

        given_base = self._make_request(
            embedding_model_id="e5_large",
            items=[self._base_item_dict()],
        )

        current = given_base.items[0].disease.value
        other = self._another_enum_value(disease_enum, current)

        if other is None:
            pytest.skip("Disease enum has only one value, cannot test change sensitivity")

        changed_payload = deepcopy(given_base.model_dump())
        changed_payload["items"][0]["disease"] = other

        assert compute_body_hash(given_base) != compute_body_hash(schemas.IngestBatchRequest(**changed_payload))

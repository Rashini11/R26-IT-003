from __future__ import annotations

import copy
import json
import re
import sys
from types import ModuleType
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

from bson import ObjectId
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
from pymongo.errors import PyMongoError

import backend.vessel_inspections as inspections_api
from backend.vessel_inspections import CloudinaryStorage
from backend.auth import (
    AuthContext,
    get_auth_context,
    require_authenticated_request,
)


class FakeCollection:
    def __init__(self):
        self.documents = []
        self.indexes = []
        self.fail_reads = False
        self.last_pipeline = None
        self.aggregate_response = [{"records": [], "metadata": []}]

    def create_index(self, *args, **kwargs):
        self.indexes.append((args, kwargs))
        return "fake-index"

    def find_one(self, query, projection=None):
        if self.fail_reads:
            raise PyMongoError("database unavailable")
        for document in self.documents:
            if all(document.get(key) == value for key, value in query.items()):
                return copy.deepcopy(document)
        return None

    def insert_one(self, document):
        inserted = copy.deepcopy(document)
        inserted.setdefault("_id", ObjectId())
        self.documents.append(inserted)
        return SimpleNamespace(inserted_id=inserted["_id"])

    def find(self, query, projection=None):
        expression = re.compile(query["$or"][0]["vessel_name"]["$regex"], re.IGNORECASE)
        matching = []
        for document in self.documents:
            searchable = [
                document.get("vessel_name", ""),
                document.get("imo_number", ""),
                document.get("mmsi_number", ""),
                document.get("verified_data", {}).get("vessel_name", ""),
                document.get("verified_data", {}).get("imo_number", ""),
                document.get("verified_data", {}).get("mmsi_number", ""),
            ]
            if any(expression.search(str(value)) for value in searchable if value):
                matching.append(copy.deepcopy(document))
        return FakeCursor(matching)

    def update_one(self, query, update):
        for document in self.documents:
            if all(document.get(key) == value for key, value in query.items()):
                document.update(copy.deepcopy(update.get("$set", {})))
                return SimpleNamespace(matched_count=1)
        return SimpleNamespace(matched_count=0)

    def aggregate(self, pipeline):
        self.last_pipeline = pipeline
        return copy.deepcopy(self.aggregate_response)


class FakeCursor:
    def __init__(self, documents):
        self.documents = documents

    def sort(self, field, direction):
        self.documents.sort(
            key=lambda document: document.get(field) or "",
            reverse=direction < 0,
        )
        return self

    def limit(self, count):
        self.documents = self.documents[:count]
        return self

    def __iter__(self):
        return iter(self.documents)


class FakeDatabase:
    def __init__(self):
        self.collections = {}

    def __getitem__(self, name):
        self.collections.setdefault(name, FakeCollection())
        return self.collections[name]


def make_client(monkeypatch, cloudinary_upload=None):
    database = FakeDatabase()
    inspections_api.configure_database(database)
    fake_cloudinary = SimpleNamespace(
        upload=cloudinary_upload
        or (lambda submission_key, filename, content_type, content: {
            "public_id": f"oceaniq/vessel-inspections/{submission_key}",
            "asset_id": f"cloudinary-{submission_key}",
            "secure_url": "https://res.cloudinary.com/example/image/authenticated/example",
            "format": "png",
        }),
        download=lambda public_id, image_format: (b"image-bytes", "image/png"),
    )
    monkeypatch.setattr(inspections_api, "CloudinaryStorage", lambda: fake_cloudinary)

    app = FastAPI()
    app.include_router(inspections_api.router)
    app.dependency_overrides[require_authenticated_request] = lambda: {
        "id": "operator-1",
        "role": "user",
    }
    return TestClient(app), database


def make_payload(**updates):
    payload = {
        "submission_key": str(uuid4()),
        "raw_ais_text": "Vessel name: TEST VESSEL\nIMO: 9074729\nMMSI: 123456789",
        "ais_fields": {
            "vessel_name": "TEST VESSEL",
            "imo_number": "9074729",
            "mmsi_number": "123456789",
        },
        "ai_results": {
            "results": [{"label": "Local Ship", "confidence": 82.0}],
            "id_text": {"readings": [{"text": "IMO 9074729"}]},
        },
        "verified_data": {
            "vessel_name": "Verified Vessel",
            "vessel_type": "Ship",
            "imo_number": "9074729",
            "mmsi_number": "123456789",
            "flag_country": "Local",
        },
        "inspection_at": datetime.now(timezone.utc).isoformat(),
        "inspector_note": "Inspected at the outer harbour.",
        "threat_status": "not_threat",
    }
    payload.update(updates)
    return payload


def test_submission_index_matches_existing_unique_index_definition(monkeypatch):
    _, database = make_client(monkeypatch)

    submission_index = next(
        options
        for keys, options in database["vessel_inspections"].indexes
        if keys == ("submission_key",)
    )

    assert submission_index["unique"] is True
    assert submission_index["name"] == "vessel_inspections_unique_submission"
    assert "partialFilterExpression" not in submission_index


def post_inspection(client, payload):
    return client.post(
        "/vessel-inspections",
        data={"payload": json.dumps(payload)},
        files={"file": ("drone.png", b"image-bytes", "image/png")},
    )


def test_parse_ais_text_keeps_missing_fields_empty():
    fields = inspections_api.parse_ais_text(
        "Vessel name: Ocean Star\nMMSI: 123456789\nSOG: 12.4 kn\nCOG: 90"
    )

    assert fields["vessel_name"] == "Ocean Star"
    assert fields["mmsi_number"] == "123456789"
    assert fields["speed"] == "12.4"
    assert fields["heading"] == "90"
    assert "imo_number" not in fields
    assert "flag_country" not in fields


def test_parse_compact_ais_paste_format_into_vessel_fields():
    fields = inspections_api.parse_ais_text(
        "MSC ISCHIA\n"
        "Container Ship\n"
        "9253649\n"
        "LKCMB\n"
        "Oct 11, 05:30\n"
        "7.6 kn / 223.6°"
    )

    assert fields == {
        "vessel_name": "MSC ISCHIA",
        "vessel_type": "Container Ship",
        "imo_number": "9253649",
        "destination": "LKCMB",
        "reported_at": "Oct 11, 05:30",
        "speed": "7.6",
        "heading": "223.6",
    }


def test_imo_precedes_mmsi_and_mmsi_is_used_when_imo_unavailable(monkeypatch):
    client, database = make_client(monkeypatch)
    vessels = database["vessels"]
    imo_vessel = {"_id": ObjectId(), "imo_number": "9074729", "mmsi_number": "111111111"}
    mmsi_vessel = {"_id": ObjectId(), "mmsi_number": "123456789"}
    vessels.documents.extend([imo_vessel, mmsi_vessel])

    result = inspections_api.find_existing_vessel(
        {"imo_number": "9074729", "mmsi_number": "111111111"},
        vessels,
    )
    assert result["status"] == "existing"
    assert result["matched_by"] == "imo"

    result = inspections_api.find_existing_vessel(
        {"mmsi_number": "123456789"},
        vessels,
    )
    assert result["status"] == "existing"
    assert result["matched_by"] == "mmsi"


def test_identify_does_not_match_on_name_and_requests_manual_review(monkeypatch):
    client, database = make_client(monkeypatch)
    database["vessels"].documents.append({
        "_id": ObjectId(),
        "vessel_name": "Same Name",
        "imo_number": "9074729",
    })

    response = client.post(
        "/vessel-inspections/identify",
        json={"raw_ais_text": "Vessel name: Same Name"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "manual_confirmation_required"


def test_conflicting_imo_and_mmsi_candidates_require_manual_selection(monkeypatch):
    client, database = make_client(monkeypatch)
    imo_id = ObjectId()
    mmsi_id = ObjectId()
    database["vessels"].documents.extend(
        [
            {"_id": imo_id, "imo_number": "9074729", "mmsi_number": "111111111"},
            {"_id": mmsi_id, "mmsi_number": "123456789"},
        ]
    )

    response = client.post(
        "/vessel-inspections/identify",
        json={
            "raw_ais_text": "IMO: 9074729\nMMSI: 123456789",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "manual_review_required"
    assert body["vessel"]["id"] == str(imo_id)
    assert body["candidate_vessel"]["id"] == str(mmsi_id)


def test_manual_registry_search_returns_candidates_without_auto_matching(monkeypatch):
    client, database = make_client(monkeypatch)
    vessel_id = ObjectId()
    database["vessels"].documents.append(
        {
            "_id": vessel_id,
            "vessel_name": "Alpha",
            "imo_number": "9074729",
        }
    )

    response = client.get("/vessel-inspections/vessels/search", params={"search": "alpha"})

    assert response.status_code == 200
    assert response.json()["manual_review_required"] is True
    assert response.json()["candidates"][0]["id"] == str(vessel_id)


def test_new_vessel_and_first_inspection_keep_evidence_and_corrections(monkeypatch):
    client, database = make_client(monkeypatch)
    payload = make_payload(
        ais_fields={
            "vessel_name": "TEST VESSEL",
            "imo_number": "9074729",
            "mmsi_number": "123456789",
        },
        verified_data={
            "vessel_name": "Operator Vessel Name",
            "vessel_type": "Ship",
            "imo_number": "9074729",
            "mmsi_number": "123456789",
            "flag_country": "Local",
        },
        threat_status="threat",
    )

    response = post_inspection(client, payload)

    assert response.status_code == 200
    body = response.json()
    assert body["vessel_created"] is True
    assert len(database["vessels"].documents) == 1
    assert len(database["vessel_inspections"].documents) == 1
    record = database["vessel_inspections"].documents[0]
    assert record["threat_status"] == "threat"
    assert record["inspector_note"] == payload["inspector_note"]
    assert record["image"]["storage_provider"] == "cloudinary"
    assert record["image"]["cloudinary_public_id"] == (
        f"oceaniq/vessel-inspections/{payload['submission_key']}"
    )
    assert record["correction_history"]["vessel_name"] == {
        "ais_value": "TEST VESSEL",
        "ai_value": None,
        "verified_value": "Operator Vessel Name",
    }
    assert record["ai_results"] == payload["ai_results"]
    assert record["ais_snapshot"]["raw_text"] == payload["raw_ais_text"]

    duplicate = post_inspection(client, payload)
    assert duplicate.status_code == 200
    assert duplicate.json()["duplicate_submission"] is True
    assert len(database["vessel_inspections"].documents) == 1


def test_existing_vessel_gets_an_additional_inspection_without_duplicate(monkeypatch):
    client, database = make_client(monkeypatch)
    vessel_id = ObjectId()
    database["vessels"].documents.append({
        "_id": vessel_id,
        "imo_number": "9074729",
        "mmsi_number": "123456789",
        "vessel_name": "Registered Vessel",
    })

    first = post_inspection(client, make_payload())
    second_payload = make_payload(threat_status="threat")
    second = post_inspection(client, second_payload)

    assert first.status_code == second.status_code == 200
    assert first.json()["vessel_created"] is False
    assert second.json()["vessel_created"] is False
    assert len(database["vessels"].documents) == 1
    assert len(database["vessel_inspections"].documents) == 2


def test_missing_identifiers_require_manual_confirmation(monkeypatch):
    client, database = make_client(monkeypatch)
    payload = make_payload(
        raw_ais_text="Vessel name: Unidentified",
        ais_fields={"vessel_name": "Unidentified"},
        verified_data={"vessel_name": "Unidentified"},
    )

    blocked = post_inspection(client, payload)
    assert blocked.status_code == 409
    assert not database["vessels"].documents
    assert not database["vessel_inspections"].documents

    payload["manual_identity_confirmed"] = True
    confirmed = post_inspection(client, payload)
    assert confirmed.status_code == 200
    assert confirmed.json()["vessel_created"] is True


def test_cloudinary_failure_does_not_save_broken_vessel_or_inspection(monkeypatch):
    def fail_upload(*args, **kwargs):
        raise inspections_api.CloudinaryError("Cloudinary rejected the image upload")

    client, database = make_client(monkeypatch, cloudinary_upload=fail_upload)
    response = post_inspection(client, make_payload())

    assert response.status_code == 502
    assert response.json()["detail"] == "Cloudinary rejected the image upload"
    assert not database["vessels"].documents
    assert not database["vessel_inspections"].documents


def test_cloudinary_upload_uses_private_deterministic_asset(monkeypatch):
    class FakeApi:
        def __init__(self):
            self.resource_calls = []

        def resource(self, public_id, **options):
            self.resource_calls.append((public_id, options))
            raise NotFound("resource missing")

    class FakeUploader:
        def __init__(self):
            self.upload_calls = []

        def upload(self, content, **options):
            self.upload_calls.append((content, options))
            return {"public_id": options["public_id"], "asset_id": "asset-1"}

    api = FakeApi()
    uploader = FakeUploader()
    exceptions_module = ModuleType("cloudinary.exceptions")

    class NotFound(Exception):
        pass

    exceptions_module.NotFound = NotFound
    sdk = SimpleNamespace(
        config=lambda **options: None,
        api=api,
        uploader=uploader,
        utils=SimpleNamespace(),
        exceptions=exceptions_module,
    )
    monkeypatch.setitem(sys.modules, "cloudinary", sdk)
    monkeypatch.setitem(sys.modules, "cloudinary.exceptions", exceptions_module)
    monkeypatch.setenv("CLOUDINARY_CLOUD_NAME", "test-cloud")
    monkeypatch.setenv("CLOUDINARY_API_KEY", "test-key")
    monkeypatch.setenv("CLOUDINARY_API_SECRET", "test-secret")
    storage = CloudinaryStorage()

    asset = storage.upload("submission-123", "drone.png", "image/png", b"image-bytes")

    assert asset["public_id"] == "oceaniq/vessel-inspections/submission-123"
    assert api.resource_calls[0][1] == {
        "resource_type": "image",
        "type": "authenticated",
    }
    assert uploader.upload_calls[0][1]["overwrite"] is False
    assert uploader.upload_calls[0][1]["type"] == "authenticated"


def test_cloudinary_retry_reuses_existing_asset(monkeypatch):
    prior_asset = {"public_id": "oceaniq/vessel-inspections/submission-123"}
    api = SimpleNamespace(resource=lambda public_id, **options: prior_asset)
    uploader = SimpleNamespace(upload=lambda *args, **kwargs: pytest.fail("duplicate upload"))
    exceptions_module = ModuleType("cloudinary.exceptions")
    exceptions_module.NotFound = type("NotFound", (Exception,), {})
    sdk = SimpleNamespace(
        config=lambda **options: None,
        api=api,
        uploader=uploader,
        utils=SimpleNamespace(),
        exceptions=exceptions_module,
    )
    monkeypatch.setitem(sys.modules, "cloudinary", sdk)
    monkeypatch.setitem(sys.modules, "cloudinary.exceptions", exceptions_module)
    monkeypatch.setenv("CLOUDINARY_CLOUD_NAME", "test-cloud")
    monkeypatch.setenv("CLOUDINARY_API_KEY", "test-key")
    monkeypatch.setenv("CLOUDINARY_API_SECRET", "test-secret")

    asset = CloudinaryStorage().upload(
        "submission-123",
        "drone.png",
        "image/png",
        b"image-bytes",
    )

    assert asset is prior_asset


def test_unconfigured_cloudinary_fails_without_persisting_inspection(monkeypatch):
    client, database = make_client(monkeypatch)
    monkeypatch.delenv("CLOUDINARY_CLOUD_NAME", raising=False)
    monkeypatch.delenv("CLOUDINARY_API_KEY", raising=False)
    monkeypatch.delenv("CLOUDINARY_API_SECRET", raising=False)
    monkeypatch.setattr(
        inspections_api,
        "CloudinaryStorage",
        CloudinaryStorage,
    )

    response = post_inspection(client, make_payload())

    assert response.status_code == 503
    assert response.json()["detail"] == "Cloudinary is not configured"
    assert not database["vessels"].documents
    assert not database["vessel_inspections"].documents


def test_database_error_is_reported_as_service_unavailable(monkeypatch):
    client, database = make_client(monkeypatch)
    database["vessels"].fail_reads = True

    response = client.post(
        "/vessel-inspections/identify",
        json={"raw_ais_text": "IMO: 9074729"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Vessel inspection storage is temporarily unavailable"


def test_history_filters_and_secure_image_endpoint(monkeypatch):
    client, database = make_client(monkeypatch)
    record_id = ObjectId()
    database["vessel_inspections"].aggregate_response = [{
        "records": [{"_id": record_id, "threat_status": "threat"}],
        "metadata": [{"total": 1}],
    }]
    database["vessel_inspections"].documents.append({
        "_id": record_id,
        "image": {
            "storage_provider": "cloudinary",
            "cloudinary_public_id": "oceaniq/vessel-inspections/image-1",
            "content_type": "image/png",
        },
    })

    history = client.get(
        "/vessel-inspections",
        params={
            "search": "IMO-9074729",
            "threat_status": "threat",
            "date_from": "2026-01-01T00:00:00+00:00",
            "page": 2,
        },
    )
    assert history.status_code == 200
    assert history.json()["records"][0]["id"] == str(record_id)
    assert history.json()["page"] == 2
    pipeline_text = str(database["vessel_inspections"].last_pipeline)
    assert "threat_status" in pipeline_text
    assert "verified_data.imo_number" in pipeline_text

    image = client.get(f"/vessel-inspections/{record_id}/image")
    assert image.status_code == 200
    assert image.content == b"image-bytes"
    assert image.headers["cache-control"] == "private, no-store"


def test_invalid_threat_status_is_rejected(monkeypatch):
    client, _ = make_client(monkeypatch)
    payload = make_payload(threat_status="maybe")

    response = post_inspection(client, payload)

    assert response.status_code == 422


def test_authentication_allows_read_only_history_but_blocks_identification_write(monkeypatch):
    client, _ = make_client(monkeypatch)
    client.app.dependency_overrides.pop(require_authenticated_request)
    client.app.dependency_overrides[get_auth_context] = lambda: AuthContext(
        user_doc={
            "_id": ObjectId(),
            "role": "user",
            "approval_status": "approved",
            "access_level": "read_only",
            "is_active": True,
        },
        session_doc={"csrf_token": "valid-token"},
    )

    history = client.get("/vessel-inspections")
    identify = client.post(
        "/vessel-inspections/identify",
        json={"raw_ais_text": "IMO: 9074729"},
        headers={"X-CSRF-Token": "valid-token"},
    )

    assert history.status_code == 200
    assert identify.status_code == 403
    assert identify.json()["detail"] == "Read-only account: this action requires read-write access"

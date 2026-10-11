from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional
from urllib.parse import quote

from bson import ObjectId
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field, ValidationError, field_validator
from pymongo.errors import DuplicateKeyError, PyMongoError

from .auth import require_authenticated_request


router = APIRouter(
    prefix="/vessel-inspections",
    tags=["Vessel inspections"],
    dependencies=[Depends(require_authenticated_request)],
)

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_PAYLOAD_BYTES = 1_000_000
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
_vessels = None
_inspections = None


class VesselFields(BaseModel):
    vessel_name: Optional[str] = Field(default=None, max_length=120)
    vessel_type: Optional[str] = Field(default=None, max_length=80)
    imo_number: Optional[str] = Field(default=None, max_length=32)
    mmsi_number: Optional[str] = Field(default=None, max_length=32)
    flag_country: Optional[str] = Field(default=None, max_length=100)
    destination: Optional[str] = Field(default=None, max_length=100)
    reported_at: Optional[str] = Field(default=None, max_length=80)
    speed: Optional[str] = Field(default=None, max_length=32)
    heading: Optional[str] = Field(default=None, max_length=32)

    @field_validator("*", mode="before")
    @classmethod
    def clean_optional_text(cls, value):
        if value is None:
            return None
        cleaned = str(value).strip()
        return cleaned or None


class IdentifyRequest(BaseModel):
    raw_ais_text: str = Field(default="", max_length=20_000)
    vessel_fields: VesselFields = Field(default_factory=VesselFields)
    verified_data: Optional[VesselFields] = None
    ai_results: dict = Field(default_factory=dict)


class InspectionSubmission(BaseModel):
    submission_key: str = Field(min_length=16, max_length=80)
    raw_ais_text: str = Field(default="", max_length=20_000)
    ais_fields: VesselFields = Field(default_factory=VesselFields)
    ai_results: dict = Field(default_factory=dict)
    verified_data: VesselFields
    vessel_id: Optional[str] = None
    manual_identity_confirmed: bool = False
    inspection_at: datetime
    inspector_note: str = Field(default="", max_length=5_000)
    threat_status: Literal["threat", "not_threat"]

    @field_validator("inspection_at")
    @classmethod
    def normalize_inspection_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("Inspection date and time must include a timezone")
        return value.astimezone(timezone.utc)

    @field_validator("submission_key")
    @classmethod
    def validate_submission_key(cls, value: str) -> str:
        try:
            return str(uuid.UUID(value))
        except ValueError as exc:
            raise ValueError("A valid inspection submission key is required") from exc

    @field_validator("vessel_id")
    @classmethod
    def validate_vessel_id(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not ObjectId.is_valid(value):
            raise ValueError("Vessel reference is invalid")
        return value


def configure_database(database) -> None:
    global _vessels, _inspections
    if database is None:
        _vessels = None
        _inspections = None
        return
    _vessels = database["vessels"]
    _inspections = database["vessel_inspections"]
    _vessels.create_index(
        "imo_number",
        unique=True,
        partialFilterExpression={"imo_number": {"$type": "string"}},
        name="vessels_unique_imo",
    )
    _vessels.create_index(
        "mmsi_number",
        unique=True,
        partialFilterExpression={"mmsi_number": {"$type": "string"}},
        name="vessels_unique_mmsi",
    )
    _inspections.create_index(
        "submission_key",
        unique=True,
        name="vessel_inspections_unique_submission",
    )
    _inspections.create_index(
        [("inspection_at", -1)],
        name="vessel_inspections_by_date",
    )
    _inspections.create_index(
        [("vessel_id", 1), ("inspection_at", -1)],
        name="vessel_inspections_by_vessel_date",
    )


def _collections():
    if _vessels is None or _inspections is None:
        raise HTTPException(
            status_code=503,
            detail="Vessel inspection storage is not available",
        )
    return _vessels, _inspections


def is_valid_imo(value: Optional[str]) -> bool:
    digits = re.sub(r"\D", "", value or "")
    if len(digits) != 7:
        return False
    checksum = sum(int(digit) * weight for digit, weight in zip(digits[:6], range(7, 1, -1)))
    return checksum % 10 == int(digits[-1])


def _digits(value: Optional[str], length: int) -> Optional[str]:
    digits = re.sub(r"\D", "", value or "")
    return digits if len(digits) == length else None


def parse_ais_text(raw_text: str) -> dict:
    fields = {
        "vessel_name": None,
        "vessel_type": None,
        "imo_number": None,
        "mmsi_number": None,
        "flag_country": None,
        "destination": None,
        "reported_at": None,
        "speed": None,
        "heading": None,
    }
    aliases = {
        "vessel_name": r"(?:vessel\s*name|ship\s*name|name)",
        "vessel_type": r"(?:vessel\s*type|ship\s*type|type)",
        "imo_number": r"(?:imo(?:\s*(?:number|no|#))?)",
        "mmsi_number": r"(?:mmsi(?:\s*(?:number|no|#))?)",
        "flag_country": r"(?:flag(?:\s*country)?|country)",
        "destination": r"(?:destination|dest(?:ination)?\s*(?:port)?)",
        "reported_at": r"(?:reported(?:\s*(?:date|time|at))?|timestamp|date\s*(?:and|/)\s*time)",
        "speed": r"(?:speed(?:\s*(?:over\s*ground|sog))?)",
        "heading": r"(?:heading|course(?:\s*over\s*ground)?|cog)",
    }
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    for line in lines:
        match = re.match(r"^\s*([^:=]+)\s*[:=]\s*(.*?)\s*$", line)
        if not match:
            continue
        label, value = match.groups()
        for field, alias in aliases.items():
            if re.fullmatch(alias, label.strip(), re.IGNORECASE):
                fields[field] = value.strip() or None
                break

    if not fields["imo_number"]:
        match = re.search(r"\bIMO[\s:#-]*(\d{7})\b", raw_text, re.IGNORECASE)
        if match:
            fields["imo_number"] = match.group(1)
    if not fields["mmsi_number"]:
        match = re.search(r"\bMMSI[\s:#-]*(\d{9})\b", raw_text, re.IGNORECASE)
        if match:
            fields["mmsi_number"] = match.group(1)

    imo_line_index = next(
        (
            index
            for index, line in enumerate(lines)
            if re.fullmatch(r"\d{7}", line)
            or re.fullmatch(r"IMO[\s:#-]*\d{7}", line, re.IGNORECASE)
        ),
        None,
    )
    if imo_line_index is not None:
        preceding = [
            line
            for line in lines[:imo_line_index]
            if not re.match(r"^[^:=]+\s*[:=]", line)
        ]
        if preceding and not fields["vessel_name"]:
            fields["vessel_name"] = preceding[0]
        if len(preceding) > 1 and not fields["vessel_type"]:
            fields["vessel_type"] = preceding[1]
        if not fields["imo_number"]:
            imo_match = re.search(r"\d{7}", lines[imo_line_index])
            if imo_match:
                fields["imo_number"] = imo_match.group(0)
        for line in lines[imo_line_index + 1 :]:
            if (
                not fields["destination"]
                and re.fullmatch(r"[A-Z0-9]{3,10}", line)
                and not line.isdigit()
            ):
                fields["destination"] = line
                break

    if not fields["reported_at"]:
        reported_match = re.search(
            r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
            r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
            r"Dec(?:ember)?)\s+\d{1,2},?\s+\d{1,2}:\d{2}"
            r"(?:\s*(?:AM|PM))?(?:\s*(?:UTC|GMT|[+-]\d{2}:?\d{2}))?\b",
            raw_text,
            re.IGNORECASE,
        )
        if reported_match:
            fields["reported_at"] = reported_match.group(0).strip()

    movement_match = re.search(
        r"^\s*(\d+(?:\.\d+)?)\s*kn(?:ots?)?\s*/\s*"
        r"(\d{1,3}(?:\.\d+)?)(?:\s*[^\w\s])?\s*$",
        raw_text,
        re.IGNORECASE | re.MULTILINE,
    )
    speed_match = re.search(
        r"\b(?:SOG|speed)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(?:kn(?:ots?)?)?\b",
        raw_text,
        re.IGNORECASE,
    )
    heading_match = re.search(
        r"\b(?:COG|heading|course)\s*[:=]?\s*(\d{1,3}(?:\.\d+)?)\s*(?:°|deg(?:rees?)?)?\b",
        raw_text,
        re.IGNORECASE,
    )
    if movement_match:
        fields["speed"] = movement_match.group(1)
        fields["heading"] = movement_match.group(2)
    if not fields["speed"] and speed_match:
        fields["speed"] = speed_match.group(1)
    if not fields["heading"] and heading_match:
        fields["heading"] = heading_match.group(1)
    return VesselFields.model_validate(fields).model_dump(exclude_none=True)


def _merge_fields(parsed: dict, provided: VesselFields) -> dict:
    result = dict(parsed)
    for name, value in provided.model_dump(exclude_none=True).items():
        if value:
            result[name] = value
    return result


def _clean_fields(fields: VesselFields) -> dict:
    cleaned = fields.model_dump(exclude_none=True)
    imo = _digits(cleaned.get("imo_number"), 7)
    mmsi = _digits(cleaned.get("mmsi_number"), 9)
    if is_valid_imo(cleaned.get("imo_number")):
        cleaned["imo_number"] = imo
    else:
        cleaned.pop("imo_number", None)
    if mmsi:
        cleaned["mmsi_number"] = mmsi
    else:
        cleaned.pop("mmsi_number", None)
    return cleaned


def _public_vessel(vessel: Optional[dict]) -> Optional[dict]:
    if not vessel:
        return None
    result = dict(vessel)
    result["id"] = str(result.pop("_id", ""))
    return _serialize(result)


def _serialize(value):
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        result = {str(key): _serialize(item) for key, item in value.items() if key != "_id"}
        if "_id" in value:
            result["id"] = str(value["_id"])
        return result
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    return value


def find_existing_vessel(fields: dict, vessels) -> dict:
    imo = _digits(fields.get("imo_number"), 7)
    mmsi = _digits(fields.get("mmsi_number"), 9)
    valid_imo = imo if is_valid_imo(imo) else None
    imo_match = vessels.find_one({"imo_number": valid_imo}) if valid_imo else None
    mmsi_match = vessels.find_one({"mmsi_number": mmsi}) if mmsi else None

    if valid_imo and imo_match:
        if mmsi_match and mmsi_match.get("_id") != imo_match.get("_id"):
            return {
                "status": "manual_review_required",
                "reason": "IMO and MMSI identify different existing vessels",
                "vessel": _public_vessel(imo_match),
                "candidate_vessel": _public_vessel(mmsi_match),
                "matched_by": None,
            }
        return {
            "status": "existing",
            "vessel": _public_vessel(imo_match),
            "matched_by": "imo",
        }
    if valid_imo and mmsi_match:
        return {
            "status": "manual_review_required",
            "reason": "The supplied IMO and MMSI do not identify the same existing vessel",
            "vessel": None,
            "candidate_vessel": _public_vessel(mmsi_match),
            "matched_by": None,
        }
    if not valid_imo and mmsi_match:
        return {
            "status": "existing",
            "vessel": _public_vessel(mmsi_match),
            "matched_by": "mmsi",
        }
    if not valid_imo and not mmsi:
        return {
            "status": "manual_confirmation_required",
            "reason": "No valid IMO or MMSI is available; verify vessel identity manually.",
            "vessel": None,
            "matched_by": None,
        }
    return {"status": "new", "vessel": None, "matched_by": None}


def _ai_candidate_fields(ai_results: dict) -> dict:
    prediction = ai_results.get("prediction", ai_results)
    if not isinstance(prediction, dict):
        prediction = {}
    labels = [
        str(item.get("label", ""))
        for item in prediction.get("results", [])
        if isinstance(item, dict)
    ]
    labels.append(str(prediction.get("vessel_origin", "")))
    classification = " ".join(labels)
    fields = {}
    if re.search(r"\bship\b", classification, re.IGNORECASE):
        fields["vessel_type"] = "Ship"
    elif re.search(r"\bboat\b", classification, re.IGNORECASE):
        fields["vessel_type"] = "Boat"
    if re.search(r"\blocal\b", classification, re.IGNORECASE):
        fields["flag_country"] = "Local"
    elif re.search(r"\bforeign\b", classification, re.IGNORECASE):
        fields["flag_country"] = "Foreign"

    id_text = prediction.get("id_text", {})
    for reading in id_text.get("readings", []) if isinstance(id_text, dict) else []:
        if not isinstance(reading, dict):
            continue
        match = re.search(r"\b(?:IMO\s*)?(\d{7})\b", str(reading.get("text", "")), re.IGNORECASE)
        if match:
            fields["imo_number"] = match.group(1)
            break
    return fields


def _mongo_failure() -> HTTPException:
    return HTTPException(
        status_code=503,
        detail="Vessel inspection storage is temporarily unavailable",
    )


class GoogleDriveError(Exception):
    pass


def _drive_response_error(response, action: str) -> str:
    try:
        error = response.json().get("error", {})
    except (ValueError, AttributeError):
        error = {}
    if not isinstance(error, dict):
        error = {}
    reasons = error.get("errors", [])
    reason = reasons[0].get("reason") if reasons and isinstance(reasons[0], dict) else None
    message = error.get("message")
    details = [str(value).replace("\r", " ").replace("\n", " ") for value in (reason, message) if value]
    suffix = f" ({response.status_code})"
    if details:
        suffix += f": {' — '.join(details)[:300]}"
    return f"Google Drive could not {action}{suffix}"


class GoogleDriveStorage:
    def __init__(self):
        self.service_account_json = os.getenv("GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON", "").strip()
        self.service_account_file = os.getenv("GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE", "").strip()
        self.folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "").strip()

    def _session(self):
        if not self.service_account_json and not self.service_account_file:
            raise GoogleDriveError("Google Drive is not configured")
        try:
            from google.auth.transport.requests import AuthorizedSession
            from google.oauth2 import service_account

            if self.service_account_json:
                credentials_info = json.loads(self.service_account_json)
                credentials = service_account.Credentials.from_service_account_info(
                    credentials_info,
                    scopes=["https://www.googleapis.com/auth/drive.file"],
                )
            else:
                credentials = service_account.Credentials.from_service_account_file(
                    self.service_account_file,
                    scopes=["https://www.googleapis.com/auth/drive.file"],
                )
            return AuthorizedSession(credentials)
        except (ImportError, ValueError, OSError, KeyError) as exc:
            raise GoogleDriveError("Google Drive credentials are unavailable or invalid") from exc

    def _existing_file(self, session, submission_key: str) -> Optional[dict]:
        query = (
            "appProperties has { key='oceaniq_submission_id' "
            f"and value='{submission_key}' }} and trashed = false"
        )
        response = session.get(
            "https://www.googleapis.com/drive/v3/files",
            params={
                "q": query,
                "fields": "files(id,name,mimeType,webViewLink)",
                "pageSize": 1,
                "supportsAllDrives": "true",
                "includeItemsFromAllDrives": "true",
            },
            timeout=20,
        )
        if response.status_code != 200:
            raise GoogleDriveError(
                _drive_response_error(response, "check for an existing upload")
            )
        files = response.json().get("files", [])
        return files[0] if files else None

    def upload(self, submission_key: str, filename: str, content_type: str, content: bytes) -> dict:
        try:
            session = self._session()
            prior = self._existing_file(session, submission_key)
            if prior:
                return prior

            metadata = {
                "name": f"{submission_key}_{Path(filename).name}",
                "mimeType": content_type,
                "appProperties": {"oceaniq_submission_id": submission_key},
            }
            if self.folder_id:
                metadata["parents"] = [self.folder_id]

            boundary = f"oceaniq_{uuid.uuid4().hex}"
            body = (
                f"--{boundary}\r\n"
                "Content-Type: application/json; charset=UTF-8\r\n\r\n"
                f"{json.dumps(metadata)}\r\n"
                f"--{boundary}\r\n"
                f"Content-Type: {content_type}\r\n\r\n"
            ).encode("utf-8") + content + f"\r\n--{boundary}--".encode("ascii")
            response = session.post(
                "https://www.googleapis.com/upload/drive/v3/files",
                params={
                    "uploadType": "multipart",
                    "fields": "id,name,mimeType,webViewLink",
                    "supportsAllDrives": "true",
                },
                headers={"Content-Type": f"multipart/related; boundary={boundary}"},
                data=body,
                timeout=60,
            )
            if response.status_code not in {200, 201}:
                raise GoogleDriveError(
                    _drive_response_error(response, "upload the image")
                )
            uploaded = response.json()
            if not uploaded.get("id"):
                raise GoogleDriveError("Google Drive returned no image reference")
            return uploaded
        except GoogleDriveError:
            raise
        except Exception as exc:
            raise GoogleDriveError("Google Drive image storage is unavailable") from exc

    def download(self, file_id: str) -> tuple[bytes, str]:
        try:
            response = self._session().get(
                f"https://www.googleapis.com/drive/v3/files/{quote(file_id, safe='')}",
                params={"alt": "media", "supportsAllDrives": "true"},
                timeout=60,
            )
            if response.status_code != 200:
                raise GoogleDriveError("The stored image could not be retrieved")
            return response.content, response.headers.get("Content-Type", "application/octet-stream")
        except GoogleDriveError:
            raise
        except Exception as exc:
            raise GoogleDriveError("The stored image could not be retrieved") from exc


class CloudinaryError(Exception):
    pass


class CloudinaryStorage:
    def __init__(self):
        self.cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME", "").strip()
        self.api_key = os.getenv("CLOUDINARY_API_KEY", "").strip()
        self.api_secret = os.getenv("CLOUDINARY_API_SECRET", "").strip()

    def _configure(self):
        if not self.cloud_name or not self.api_key or not self.api_secret:
            raise CloudinaryError("Cloudinary is not configured")
        try:
            import cloudinary
            from cloudinary import api, uploader, utils
            from cloudinary.exceptions import NotFound

            cloudinary.config(
                cloud_name=self.cloud_name,
                api_key=self.api_key,
                api_secret=self.api_secret,
                secure=True,
            )
            return api, uploader, utils, NotFound
        except ImportError as exc:
            raise CloudinaryError("Cloudinary backend dependency is unavailable") from exc

    @staticmethod
    def _public_id(submission_key: str) -> str:
        return f"oceaniq/vessel-inspections/{submission_key}"

    def upload(
        self,
        submission_key: str,
        filename: str,
        content_type: str,
        content: bytes,
    ) -> dict:
        cloudinary_api, cloudinary_uploader, _, not_found_error = self._configure()
        public_id = self._public_id(submission_key)
        try:
            try:
                asset = cloudinary_api.resource(
                    public_id,
                    resource_type="image",
                    type="authenticated",
                )
            except Exception as exc:
                if not isinstance(exc, not_found_error):
                    raise
                asset = cloudinary_uploader.upload(
                    content,
                    resource_type="image",
                    type="authenticated",
                    public_id=public_id,
                    overwrite=False,
                    context={"filename": Path(filename).name},
                )
            if not asset.get("public_id"):
                raise CloudinaryError("Cloudinary returned no image reference")
            return asset
        except CloudinaryError:
            raise
        except Exception as exc:
            raise CloudinaryError(
                f"Cloudinary could not upload the image: {getattr(exc, 'message', str(exc))[:300]}"
            ) from exc

    def download(self, public_id: str, image_format: Optional[str]) -> tuple[bytes, str]:
        _, _, cloudinary_utils, _ = self._configure()
        try:
            import requests
            import time

            download_url = cloudinary_utils.private_download_url(
                public_id,
                image_format,
                resource_type="image",
                type="authenticated",
                expires_at=int(time.time()) + 60,
            )
            response = requests.get(download_url, timeout=60)
            if response.status_code != 200:
                raise CloudinaryError(
                    f"Cloudinary could not retrieve the inspection image ({response.status_code})"
                )
            return response.content, response.headers.get("Content-Type", "application/octet-stream")
        except CloudinaryError:
            raise
        except Exception as exc:
            raise CloudinaryError(
                f"Cloudinary could not retrieve the inspection image: "
                f"{getattr(exc, 'message', str(exc))[:300]}"
            ) from exc


@router.post("/identify")
def identify_vessel(request: IdentifyRequest):
    vessels, _ = _collections()
    parsed = _merge_fields(parse_ais_text(request.raw_ais_text), request.vessel_fields)
    match_fields = (
        _merge_fields(parsed, request.verified_data)
        if request.verified_data is not None
        else parsed
    )
    ai_fields = _ai_candidate_fields(request.ai_results)
    if not is_valid_imo(match_fields.get("imo_number")) and is_valid_imo(
        ai_fields.get("imo_number")
    ):
        match_fields["imo_number"] = ai_fields["imo_number"]
    if not _digits(match_fields.get("mmsi_number"), 9) and _digits(
        ai_fields.get("mmsi_number"), 9
    ):
        match_fields["mmsi_number"] = ai_fields["mmsi_number"]
    try:
        match = find_existing_vessel(match_fields, vessels)
    except PyMongoError as exc:
        raise _mongo_failure() from exc
    return {"ais_fields": parsed, **match}


@router.post("")
async def create_vessel_inspection(
    payload: str = Form(...),
    file: UploadFile = File(...),
    user: dict = Depends(require_authenticated_request),
):
    vessels, inspections = _collections()
    if len(payload.encode("utf-8")) > MAX_PAYLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Inspection details are too large")
    try:
        submission = InspectionSubmission.model_validate_json(payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

    try:
        prior_inspection = inspections.find_one({"submission_key": submission.submission_key})
    except PyMongoError as exc:
        raise _mongo_failure() from exc
    if prior_inspection:
        return {
            "message": "Inspection already saved",
            "inspection": _serialize(prior_inspection),
            "duplicate_submission": True,
        }

    content_type = (file.content_type or "").lower()
    if content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=415, detail="Upload a supported vessel image")
    if not file.filename:
        raise HTTPException(status_code=422, detail="The inspection image must have a filename")
    image_bytes = await file.read(MAX_IMAGE_BYTES + 1)
    await file.close()
    if not image_bytes:
        raise HTTPException(status_code=422, detail="The inspection image is empty")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Inspection images must be 20 MB or smaller")

    ais_fields = _merge_fields(
        parse_ais_text(submission.raw_ais_text),
        submission.ais_fields,
    )
    verified_data = _clean_fields(submission.verified_data)
    try:
        identity = find_existing_vessel(verified_data, vessels)
    except PyMongoError as exc:
        raise _mongo_failure() from exc
    manual_confirmation = submission.manual_identity_confirmed

    if identity["status"] == "manual_review_required" and not manual_confirmation:
        raise HTTPException(
            status_code=409,
            detail="IMO and MMSI are inconsistent; manually verify vessel identity",
        )
    if identity["status"] == "manual_review_required" and not submission.vessel_id:
        raise HTTPException(
            status_code=409,
            detail="Select one of the candidate vessels after manual review",
        )

    selected_vessel = None
    if submission.vessel_id:
        try:
            selected_vessel = vessels.find_one({"_id": ObjectId(submission.vessel_id)})
        except PyMongoError as exc:
            raise _mongo_failure() from exc
        if not selected_vessel:
            raise HTTPException(status_code=404, detail="Selected vessel no longer exists")
        identity_ids = {
            item.get("_id")
            for item in (identity.get("vessel"), identity.get("candidate_vessel"))
            if item
        }
        if identity_ids and selected_vessel.get("_id") not in identity_ids:
            raise HTTPException(
                status_code=409,
                detail="Selected vessel does not match the supplied identifiers",
            )
        if not identity_ids and not manual_confirmation:
            raise HTTPException(
                status_code=409,
                detail="Confirm vessel identity manually before linking this inspection",
            )
    elif identity["status"] == "existing":
        try:
            selected_vessel = vessels.find_one({"_id": ObjectId(identity["vessel"]["id"])})
        except PyMongoError as exc:
            raise _mongo_failure() from exc
    elif identity["status"] == "manual_review_required":
        if not manual_confirmation:
            raise HTTPException(
                status_code=409,
                detail="IMO and MMSI are inconsistent; manually verify vessel identity",
            )
        selected_id = submission.vessel_id
        chosen = next(
            (
                item
                for item in (identity.get("vessel"), identity.get("candidate_vessel"))
                if item and item.get("id") == selected_id
            ),
            None,
        )
        if chosen:
            try:
                selected_vessel = vessels.find_one({"_id": ObjectId(chosen["id"])})
            except PyMongoError as exc:
                raise _mongo_failure() from exc
        else:
            raise HTTPException(
                status_code=409,
                detail="Select one of the candidate vessels after manual review",
            )
    elif identity["status"] == "manual_confirmation_required" and not manual_confirmation:
        raise HTTPException(
            status_code=409,
            detail="Confirm vessel identity manually when IMO and MMSI are unavailable",
        )

    try:
        cloudinary_asset = CloudinaryStorage().upload(
            submission.submission_key,
            file.filename,
            content_type,
            image_bytes,
        )
    except CloudinaryError as exc:
        status_code = 503 if "not configured" in str(exc).lower() else 502
        raise HTTPException(
            status_code=status_code,
            detail=str(exc),
        ) from exc

    now = datetime.now(timezone.utc)
    vessel_set = {
        "verified_data": verified_data,
        "last_ais_data": {"raw_text": submission.raw_ais_text, "parsed": ais_fields},
        "last_ai_results": submission.ai_results,
        "updated_at": now,
    }
    vessel_set.update(
        {
            field: verified_data[field]
            for field in ("imo_number", "mmsi_number", "vessel_name")
            if verified_data.get(field)
        }
    )
    if selected_vessel:
        vessel_id = selected_vessel["_id"]
        try:
            vessels.update_one({"_id": vessel_id}, {"$set": vessel_set})
        except DuplicateKeyError as exc:
            raise HTTPException(
                status_code=409,
                detail="The corrected vessel identifier already belongs to another vessel",
            ) from exc
        except PyMongoError as exc:
            raise _mongo_failure() from exc
        created_vessel = False
    else:
        vessel_doc = {
            **vessel_set,
            "original_ais_data": {"raw_text": submission.raw_ais_text, "parsed": ais_fields},
            "ai_prediction_results": submission.ai_results,
            "imo_number": verified_data.get("imo_number"),
            "mmsi_number": verified_data.get("mmsi_number"),
            "vessel_name": verified_data.get("vessel_name"),
            "created_at": now,
        }
        vessel_doc = {key: value for key, value in vessel_doc.items() if value is not None}
        try:
            vessel_id = vessels.insert_one(vessel_doc).inserted_id
            selected_vessel = {"_id": vessel_id, **vessel_doc}
        except DuplicateKeyError as exc:
            try:
                refreshed = find_existing_vessel(verified_data, vessels)
                if refreshed.get("status") != "existing" or manual_confirmation:
                    raise HTTPException(
                        status_code=409,
                        detail="A vessel with these identifiers already exists; review its identity",
                    ) from exc
                vessel_id = ObjectId(refreshed["vessel"]["id"])
                selected_vessel = vessels.find_one({"_id": vessel_id})
                if not selected_vessel:
                    raise HTTPException(
                        status_code=409,
                        detail="The matching vessel changed during submission; check its identity again",
                    )
                vessels.update_one({"_id": vessel_id}, {"$set": vessel_set})
            except HTTPException:
                raise
            except PyMongoError as db_exc:
                raise _mongo_failure() from db_exc
            except Exception:
                raise HTTPException(
                    status_code=409,
                    detail="A vessel with these identifiers already exists; review its identity",
                ) from exc
        except PyMongoError as exc:
            raise _mongo_failure() from exc
        created_vessel = selected_vessel.get("created_at") == now

    ai_fields = _ai_candidate_fields(submission.ai_results)
    correction_history = {}
    for field in VesselFields.model_fields:
        ais_value = ais_fields.get(field)
        ai_value = ai_fields.get(field)
        verified_value = verified_data.get(field)
        if (
            (ais_value is not None and verified_value != ais_value)
            or (ai_value is not None and verified_value != ai_value)
            or (ais_value is None and ai_value is None and verified_value is not None)
        ):
            correction_history[field] = {
                "ais_value": ais_value,
                "ai_value": ai_value,
                "verified_value": verified_value,
            }
    inspection_doc = {
        "vessel_id": vessel_id,
        "submission_key": submission.submission_key,
        "inspection_at": submission.inspection_at,
        "inspector_note": submission.inspector_note.strip(),
        "threat_status": submission.threat_status,
        "ais_snapshot": {"raw_text": submission.raw_ais_text, "parsed": ais_fields},
        "ai_results": submission.ai_results,
        "verified_data": verified_data,
        "correction_history": correction_history,
        "image": {
            "storage_provider": "cloudinary",
            "cloudinary_public_id": cloudinary_asset["public_id"],
            "cloudinary_asset_id": cloudinary_asset.get("asset_id"),
            "secure_url": cloudinary_asset.get("secure_url"),
            "format": cloudinary_asset.get("format"),
            "image_url": "",
            "filename": Path(file.filename).name,
            "content_type": content_type,
        },
        "inspector_id": user.get("id"),
        "created_at": now,
    }
    try:
        inspection_doc["_id"] = ObjectId()
        inspection_doc["image"]["image_url"] = (
            f"/vessel-inspections/{inspection_doc['_id']}/image"
        )
        inspections.insert_one(inspection_doc)
    except DuplicateKeyError:
        try:
            existing = inspections.find_one({"submission_key": submission.submission_key})
        except PyMongoError as exc:
            raise _mongo_failure() from exc
        if existing:
            return {
                "message": "Inspection already saved",
                "inspection": _serialize(existing),
                "duplicate_submission": True,
            }
        raise HTTPException(status_code=409, detail="Inspection submission already exists")
    except PyMongoError as exc:
        raise _mongo_failure() from exc

    return {
        "message": "Vessel and inspection saved successfully",
        "vessel": _public_vessel(selected_vessel),
        "vessel_created": created_vessel,
        "inspection": _serialize(inspection_doc),
        "duplicate_submission": False,
    }


@router.get("")
def list_vessel_inspections(
    search: Optional[str] = Query(default=None, max_length=120),
    threat_status: Optional[Literal["threat", "not_threat"]] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=100),
):
    vessels, inspections = _collections()
    match: dict = {}
    if threat_status:
        match["threat_status"] = threat_status
    if date_from or date_to:
        date_query = {}
        if date_from:
            date_query["$gte"] = date_from
        if date_to:
            date_query["$lte"] = date_to
        match["inspection_at"] = date_query
    pipeline = [
        {"$match": match},
        {
            "$lookup": {
                "from": "vessels",
                "localField": "vessel_id",
                "foreignField": "_id",
                "as": "vessel",
            }
        },
        {"$unwind": {"path": "$vessel", "preserveNullAndEmptyArrays": True}},
    ]
    if search and search.strip():
        expression = re.escape(search.strip())
        pipeline.append(
            {
                "$match": {
                    "$or": [
                        {"verified_data.vessel_name": {"$regex": expression, "$options": "i"}},
                        {"verified_data.imo_number": {"$regex": expression, "$options": "i"}},
                        {"verified_data.mmsi_number": {"$regex": expression, "$options": "i"}},
                        {"vessel.vessel_name": {"$regex": expression, "$options": "i"}},
                        {"vessel.imo_number": {"$regex": expression, "$options": "i"}},
                        {"vessel.mmsi_number": {"$regex": expression, "$options": "i"}},
                    ]
                }
            }
        )
    pipeline.extend(
        [
            {"$sort": {"inspection_at": -1, "_id": -1}},
            {
                "$facet": {
                    "records": [
                        {"$skip": (page - 1) * page_size},
                        {"$limit": page_size},
                        {
                            "$project": {
                                "submission_key": 0,
                                "inspector_id": 0,
                                "vessel.last_ais_data": 0,
                                "vessel.last_ai_results": 0,
                            }
                        },
                    ],
                    "metadata": [{"$count": "total"}],
                }
            },
        ]
    )
    try:
        result = list(inspections.aggregate(pipeline))
    except PyMongoError as exc:
        raise _mongo_failure() from exc
    facet = result[0] if result else {"records": [], "metadata": []}
    total = facet["metadata"][0]["total"] if facet["metadata"] else 0
    return {
        "records": [_serialize(item) for item in facet["records"]],
        "page": page,
        "page_size": page_size,
        "total": total,
        "pages": (total + page_size - 1) // page_size,
    }


@router.get("/vessels/search")
def search_vessels_for_manual_review(
    search: str = Query(min_length=2, max_length=120),
):
    vessels, _ = _collections()
    search = search.strip()
    if len(search) < 2:
        raise HTTPException(status_code=422, detail="Enter at least two characters to search")
    expression = re.escape(search)
    query = {
        "$or": [
            {"vessel_name": {"$regex": expression, "$options": "i"}},
            {"imo_number": {"$regex": expression, "$options": "i"}},
            {"mmsi_number": {"$regex": expression, "$options": "i"}},
            {"verified_data.vessel_name": {"$regex": expression, "$options": "i"}},
            {"verified_data.imo_number": {"$regex": expression, "$options": "i"}},
            {"verified_data.mmsi_number": {"$regex": expression, "$options": "i"}},
        ]
    }
    try:
        candidates = list(
            vessels.find(
                query,
                {
                    "vessel_name": 1,
                    "imo_number": 1,
                    "mmsi_number": 1,
                    "verified_data": 1,
                },
            )
            .sort("vessel_name", 1)
            .limit(10)
        )
    except PyMongoError as exc:
        raise _mongo_failure() from exc
    return {
        "candidates": [_public_vessel(vessel) for vessel in candidates],
        "manual_review_required": True,
    }


@router.get("/{inspection_id}")
def get_vessel_inspection(inspection_id: str):
    _, inspections = _collections()
    if not ObjectId.is_valid(inspection_id):
        raise HTTPException(status_code=422, detail="Inspection reference is invalid")
    try:
        record = inspections.find_one({"_id": ObjectId(inspection_id)})
        if not record:
            raise HTTPException(status_code=404, detail="Vessel inspection was not found")
        vessel = _vessels.find_one({"_id": record["vessel_id"]})
    except PyMongoError as exc:
        raise _mongo_failure() from exc
    return {"inspection": _serialize(record), "vessel": _public_vessel(vessel)}


@router.get("/{inspection_id}/image")
def get_vessel_inspection_image(inspection_id: str):
    _, inspections = _collections()
    if not ObjectId.is_valid(inspection_id):
        raise HTTPException(status_code=422, detail="Inspection reference is invalid")
    try:
        record = inspections.find_one(
            {"_id": ObjectId(inspection_id)},
            {
                "image.storage_provider": 1,
                "image.cloudinary_public_id": 1,
                "image.format": 1,
                "image.drive_file_id": 1,
                "image.content_type": 1,
            },
        )
        if not record:
            raise HTTPException(status_code=404, detail="Vessel inspection was not found")
    except PyMongoError as exc:
        raise _mongo_failure() from exc
    image = record.get("image", {})
    if image.get("storage_provider") == "cloudinary":
        public_id = image.get("cloudinary_public_id")
        if not public_id:
            raise HTTPException(status_code=404, detail="Inspection image reference is missing")
        try:
            content, content_type = CloudinaryStorage().download(
                public_id,
                image.get("format"),
            )
        except CloudinaryError as exc:
            status_code = 503 if "not configured" in str(exc).lower() else 502
            raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    elif image.get("drive_file_id"):
        try:
            content, content_type = GoogleDriveStorage().download(image["drive_file_id"])
        except GoogleDriveError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
    else:
        raise HTTPException(status_code=404, detail="Inspection image reference is missing")
    return Response(
        content=content,
        media_type=image.get("content_type") or content_type,
        headers={"Cache-Control": "private, no-store"},
    )

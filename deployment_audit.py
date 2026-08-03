"""Print a deterministic runtime-state summary for local/Docker comparison."""

import hashlib
import json
import os
from datetime import date, datetime
from pathlib import Path

from app import app
from extensions import db
from models.database import ModelMetadata, ModelVersion
from services.model_version_service import ARTIFACT_FILES, MODELS_DIR
from services.report_data_service import get_report_data
from utils.timezone import JAKARTA_TZ, jakarta_now


PROJECT_ROOT = Path(__file__).resolve().parent
TABLES = (
    "patients",
    "prediction_history",
    "training_dataset_records",
    "model_versions",
    "model_metadata",
)
REPORTS = (
    "training_dataset",
    "patients",
    "prediction_history",
    "model_performance",
)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _tree_hash(directory):
    digest = hashlib.sha256()
    directory = Path(directory)
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _json_value(value):
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)


def _report_fingerprint(report_key):
    report = get_report_data(report_key)
    summary = [item for item in (report.get("summary") or []) if item[0] != "Tanggal Cetak"]
    rows = [row for row in report["rows"] if not row or row[0] != "Tanggal Cetak"]
    semantic_content = {
        "title": report["title"],
        "headers": report["headers"],
        "rows": rows,
        "total_records": report["total_records"],
        "summary": summary,
    }
    encoded = json.dumps(
        semantic_content,
        default=_json_value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "total_records": report["total_records"],
        "semantic_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def collect_state():
    metadata = db.session.get(ModelMetadata, 1)
    versions = ModelVersion.query.order_by(ModelVersion.id).all()
    counts = {
        table: db.session.execute(db.text(f"SELECT COUNT(*) FROM {table}")).scalar_one()
        for table in TABLES
    }
    artifacts = {
        filename: _sha256(MODELS_DIR / filename)
        for filename in ARTIFACT_FILES
        if (MODELS_DIR / filename).is_file()
    }
    state = {
        "database_path": app.config["SQLALCHEMY_DATABASE_URI"],
        "counts": counts,
        "active_model": {
            "version": metadata.model_version if metadata else None,
            "active_version_id": metadata.active_version_id if metadata else None,
            "last_training_date": metadata.last_training_date if metadata else None,
            "training_dataset_signature": metadata.training_dataset_signature if metadata else None,
        },
        "model_versions": [
            {
                "id": version.id,
                "version": version.version,
                "training_date": version.training_date,
                "training_record_count": version.training_record_count,
            }
            for version in versions
        ],
        "active_artifact_hashes": artifacts,
        "version_directories": sorted(path.name for path in (MODELS_DIR / "versions").glob("v*") if path.is_dir()),
        "reports": {report: _report_fingerprint(report) for report in REPORTS},
        "templates_sha256": _tree_hash(PROJECT_ROOT / "templates"),
        "static_sha256": _tree_hash(PROJECT_ROOT / "static"),
        "timezone": {
            "configured": str(JAKARTA_TZ),
            "jakarta_now": jakarta_now(),
            "environment_tz": os.environ.get("TZ"),
        },
    }
    comparable = {
        key: state[key]
        for key in (
            "counts",
            "active_model",
            "model_versions",
            "active_artifact_hashes",
            "version_directories",
            "reports",
            "templates_sha256",
            "static_sha256",
        )
    }
    encoded = json.dumps(
        comparable,
        default=_json_value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    state["deployment_fingerprint"] = hashlib.sha256(encoded).hexdigest()
    return state


if __name__ == "__main__":
    with app.app_context():
        print(json.dumps(collect_state(), default=_json_value, ensure_ascii=False, indent=2, sort_keys=True))

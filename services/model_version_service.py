"""Versioned storage, evaluation, and activation for trained model bundles."""

import hashlib
import json
import os
import shutil
import tempfile
import threading
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from extensions import db
from models.database import ModelMetadata, ModelVersion
from services.training_service import (
    normalize_training_dataset_dataframe,
    legacy_training_dataset_signature,
    restore_training_dataset_dataframe,
    train_models_from_database,
    training_dataset_content_signature,
    training_dataset_dataframe,
    training_dataset_signature,
)
from utils.preprocessing import preprocess_pipeline, split_data
from utils.timezone import jakarta_now

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
VERSION_ROOT = MODELS_DIR / "versions"
DATASET_FILENAME = "dataset.csv"
MANIFEST_FILENAME = "manifest.json"
ARTIFACT_FILES = (
    "xgboost_classifier.joblib",
    "xgboost_regressor.joblib",
    "scaler.joblib",
    "label_encoders.joblib",
    "feature_names.joblib",
)
_MODEL_OPERATION_LOCK = threading.Lock()


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _json_dump(value):
    return json.dumps(value, default=_json_default, sort_keys=True)


def _file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_hashes(directory):
    return {name: _file_hash(directory / name) for name in ARTIFACT_FILES}


def _manifest_path(directory):
    return Path(directory) / MANIFEST_FILENAME


def _read_manifest(directory):
    path = _manifest_path(directory)
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as manifest_file:
        return json.load(manifest_file)


def _write_manifest(directory, version, training_date, record_count, dataset_hash):
    """Write only the essential, human-readable restore-point metadata."""
    directory = Path(directory)
    manifest = {
        "version": version,
        "training_date": training_date.isoformat(),
        "training_record_count": int(record_count),
        "dataset_filename": DATASET_FILENAME,
        "dataset_sha256": dataset_hash,
    }
    staging = directory / f".{MANIFEST_FILENAME}.tmp"
    with staging.open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2, sort_keys=True)
        manifest_file.write("\n")
    os.replace(staging, _manifest_path(directory))


def _archive_dataset(directory, dataset, version, training_date):
    """Archive the exact normalized Training Dataset beside a version's artifacts."""
    directory = Path(directory)
    normalized = normalize_training_dataset_dataframe(dataset)
    dataset_path = directory / DATASET_FILENAME
    if dataset_path.exists():
        archived = normalize_training_dataset_dataframe(pd.read_csv(dataset_path))
        if training_dataset_content_signature(archived) != training_dataset_content_signature(normalized):
            raise FileExistsError(f"Dataset archive for {version} already contains different records.")
    else:
        staging = directory / f".{DATASET_FILENAME}.tmp"
        normalized.to_csv(staging, index=False, lineterminator="\n")
        os.replace(staging, dataset_path)

    dataset_hash = _file_hash(dataset_path)
    _write_manifest(directory, version, training_date, len(normalized), dataset_hash)
    return training_dataset_content_signature(normalized), dataset_hash


def _load_archived_dataset(version):
    directory = _resolve_artifact_path(version.artifact_path)
    manifest = _read_manifest(directory)
    filename = manifest.get("dataset_filename", DATASET_FILENAME)
    dataset_path = directory / filename
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Training Dataset archive is not available for {version.version}.")
    expected_hash = manifest.get("dataset_sha256")
    if expected_hash and _file_hash(dataset_path) != expected_hash:
        raise ValueError("Training Dataset archive integrity validation failed.")

    dataset = normalize_training_dataset_dataframe(pd.read_csv(dataset_path))
    if len(dataset) != version.training_record_count:
        raise ValueError("Training Dataset archive record count does not match the model version.")
    signature = training_dataset_content_signature(dataset)
    if signature != version.training_dataset_signature:
        raise ValueError("Training Dataset archive does not match the selected model version.")
    return dataset


def _validate_bundle(directory, expected_hashes=None):
    directory = Path(directory)
    missing = [name for name in ARTIFACT_FILES if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Model artifact bundle is incomplete: {', '.join(missing)}")

    hashes = _artifact_hashes(directory)
    if expected_hashes and hashes != expected_hashes:
        raise ValueError("Model artifact integrity validation failed.")

    classifier = joblib.load(directory / "xgboost_classifier.joblib")
    regressor = joblib.load(directory / "xgboost_regressor.joblib")
    scaler = joblib.load(directory / "scaler.joblib")
    encoders = joblib.load(directory / "label_encoders.joblib")
    feature_names = joblib.load(directory / "feature_names.joblib")
    if not hasattr(classifier, "predict_proba") or not hasattr(regressor, "predict"):
        raise ValueError("The selected artifact bundle does not contain valid XGBoost models.")
    if not hasattr(scaler, "transform") or not isinstance(encoders, dict) or not feature_names:
        raise ValueError("The selected artifact bundle contains invalid preprocessing artifacts.")
    return hashes


def _version_number(version):
    try:
        return int(str(version).lower().lstrip("v"))
    except (TypeError, ValueError):
        return 0


def _next_version(metadata):
    numbers = [_version_number(item.version) for item in ModelVersion.query.all()]
    if metadata:
        numbers.append(_version_number(metadata.model_version))
    return f"v{max(numbers or [0]) + 1}"


def _stored_artifact_path(directory):
    directory = Path(directory).resolve()
    try:
        return directory.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(directory)


def _resolve_artifact_path(stored_path):
    path = Path(stored_path)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _archive_active_bundle(version):
    """Copy the complete active bundle into an immutable version directory."""
    _validate_bundle(MODELS_DIR)
    VERSION_ROOT.mkdir(parents=True, exist_ok=True)
    destination = VERSION_ROOT / version
    source_hashes = _artifact_hashes(MODELS_DIR)
    if destination.exists():
        existing_hashes = _validate_bundle(destination)
        if existing_hashes != source_hashes:
            raise FileExistsError(f"Artifact directory for {version} already contains another bundle.")
        return destination, existing_hashes, False

    staging = Path(tempfile.mkdtemp(prefix=f".{version}-", dir=VERSION_ROOT))
    try:
        for name in ARTIFACT_FILES:
            shutil.copy2(MODELS_DIR / name, staging / name)
        hashes = _validate_bundle(staging, source_hashes)
        os.replace(staging, destination)
        return destination, hashes, True
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _replace_active_bundle(source_directory):
    """Validate and atomically replace every canonical active artifact file."""
    source_directory = Path(source_directory)
    _validate_bundle(source_directory)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".activation-", dir=MODELS_DIR))
    try:
        for name in ARTIFACT_FILES:
            shutil.copy2(source_directory / name, staging / name)
        _validate_bundle(staging)
        for name in ARTIFACT_FILES:
            os.replace(staging / name, MODELS_DIR / name)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _backup_active_bundle():
    backup = Path(tempfile.mkdtemp(prefix="sleep-model-backup-"))
    for name in ARTIFACT_FILES:
        shutil.copy2(MODELS_DIR / name, backup / name)
    _validate_bundle(backup)
    return backup


def evaluate_artifact_bundle(directory, dataset=None):
    """Evaluate existing artifacts on the unchanged deterministic pipeline split."""
    directory = Path(directory)
    _validate_bundle(directory)
    dataset = dataset if dataset is not None else training_dataset_dataframe()
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", encoding="utf-8", newline="", delete=False
        ) as temporary_file:
            temporary_path = temporary_file.name
            dataset.to_csv(temporary_file, index=False)
        processed = preprocess_pipeline(temporary_path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)

    if processed is None:
        raise ValueError("The active model could not be evaluated against the Training Dataset.")

    classifier = joblib.load(directory / "xgboost_classifier.joblib")
    regressor = joblib.load(directory / "xgboost_regressor.joblib")
    scaler = joblib.load(directory / "scaler.joblib")
    feature_names = joblib.load(directory / "feature_names.joblib")
    features = processed["features"]
    if list(features.columns) != list(feature_names):
        raise ValueError("Active model features do not match the Training Dataset features.")

    _, classification_test, _, classification_target = split_data(
        features, processed["target_classification"]
    )
    _, regression_test, _, regression_target = split_data(
        features, processed["target_regression"]
    )
    classification_prediction = classifier.predict(scaler.transform(classification_test))
    regression_prediction = regressor.predict(scaler.transform(regression_test))

    classification = {
        "accuracy": float(accuracy_score(classification_target, classification_prediction)),
        "precision": float(
            precision_score(
                classification_target,
                classification_prediction,
                average="weighted",
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                classification_target,
                classification_prediction,
                average="weighted",
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                classification_target,
                classification_prediction,
                average="weighted",
                zero_division=0,
            )
        ),
    }
    regression = {
        "mae": float(mean_absolute_error(regression_target, regression_prediction)),
        "rmse": float(np.sqrt(mean_squared_error(regression_target, regression_prediction))),
        "r2": float(r2_score(regression_target, regression_prediction)),
    }
    matrix = confusion_matrix(classification_target, classification_prediction).tolist()
    return classification, regression, matrix


def _version_record(
    version,
    training_date,
    dataset_signature,
    record_count,
    classification_metrics,
    regression_metrics,
    matrix,
    artifact_directory,
    hashes,
):
    classification = dict(classification_metrics)
    classification.pop("confusion_matrix", None)
    return ModelVersion(
        version=version,
        training_date=training_date,
        training_dataset_signature=dataset_signature,
        training_record_count=record_count,
        classification_metrics=_json_dump(classification),
        regression_metrics=_json_dump(regression_metrics),
        confusion_matrix=_json_dump(matrix),
        artifact_path=_stored_artifact_path(artifact_directory),
        artifact_hashes=_json_dump(hashes),
    )


def ensure_active_model_version():
    """Backfill the singleton's current active bundle into the version catalog once."""
    metadata = db.session.get(ModelMetadata, 1)
    if metadata is None:
        return None

    existing = ModelVersion.query.filter_by(version=metadata.model_version).first()
    if existing is not None:
        if metadata.active_version_id != existing.id:
            metadata.active_version_id = existing.id
            db.session.commit()
        return existing

    dataset = training_dataset_dataframe()
    dataset_signature = training_dataset_content_signature(dataset)
    classification, regression, matrix = evaluate_artifact_bundle(MODELS_DIR)
    artifact_directory, hashes, archive_created = _archive_active_bundle(metadata.model_version)
    try:
        version = _version_record(
            version=metadata.model_version,
            training_date=metadata.last_training_date or jakarta_now(),
            dataset_signature=dataset_signature,
            record_count=len(training_dataset_dataframe()),
            classification_metrics=classification,
            regression_metrics=regression,
            matrix=matrix,
            artifact_directory=artifact_directory,
            hashes=hashes,
        )
        db.session.add(version)
        db.session.flush()
        _archive_dataset(
            artifact_directory,
            dataset,
            version.version,
            version.training_date,
        )
        metadata.active_version_id = version.id
        metadata.training_dataset_signature = dataset_signature
        db.session.commit()
        return version
    except Exception:
        db.session.rollback()
        if archive_created:
            shutil.rmtree(artifact_directory, ignore_errors=True)
        raise


def register_archived_model_version(version_name):
    """Register a bundled version manifest without retraining the model."""
    existing = ModelVersion.query.filter_by(version=version_name).first()
    if existing is not None:
        return existing, False

    artifact_directory = VERSION_ROOT / version_name
    manifest_path = artifact_directory / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Model version manifest was not found for {version_name}.")

    with manifest_path.open("r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    if manifest.get("version") != version_name:
        raise ValueError(f"Model version manifest does not match {version_name}.")

    expected_hashes = manifest.get("artifact_hashes")
    hashes = _validate_bundle(artifact_directory, expected_hashes)
    dataset_path = artifact_directory / manifest.get("dataset_filename", DATASET_FILENAME)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Training Dataset archive was not found for {version_name}.")
    expected_dataset_hash = manifest.get("dataset_sha256")
    if expected_dataset_hash and _file_hash(dataset_path) != expected_dataset_hash:
        raise ValueError("Training Dataset archive integrity validation failed.")
    archived_dataset = normalize_training_dataset_dataframe(pd.read_csv(dataset_path))
    dataset_signature = training_dataset_content_signature(archived_dataset)
    if len(archived_dataset) != int(manifest["training_record_count"]):
        raise ValueError(f"Training Dataset archive count does not match {version_name}.")

    if all(key in manifest for key in ("classification_metrics", "regression_metrics", "confusion_matrix")):
        classification = manifest["classification_metrics"]
        regression = manifest["regression_metrics"]
        matrix = manifest["confusion_matrix"]
    else:
        classification, regression, matrix = evaluate_artifact_bundle(
            artifact_directory,
            archived_dataset,
        )
    classification = dict(classification)
    classification.pop("confusion_matrix", None)

    version = ModelVersion(
        version=version_name,
        training_date=datetime.fromisoformat(manifest["training_date"]),
        training_dataset_signature=dataset_signature,
        training_record_count=int(manifest["training_record_count"]),
        classification_metrics=_json_dump(classification),
        regression_metrics=_json_dump(regression),
        confusion_matrix=_json_dump(matrix),
        artifact_path=_stored_artifact_path(artifact_directory),
        artifact_hashes=_json_dump(hashes),
    )
    try:
        db.session.add(version)
        db.session.commit()
        return version, True
    except Exception:
        db.session.rollback()
        existing = ModelVersion.query.filter_by(version=version_name).first()
        if existing is not None:
            return existing, False
        raise


def activate_configured_model_version():
    """Bootstrap a bundled version when the database has no valid active catalog row."""
    version_name = os.environ.get("ACTIVE_MODEL_VERSION", "").strip()
    if not version_name:
        return None, False

    version, _ = register_archived_model_version(version_name)
    metadata = db.session.get(ModelMetadata, 1)
    if active_model_version(metadata) is not None:
        return active_model_version(metadata), False
    return activate_model_version(version.id)


def active_model_version(metadata=None):
    metadata = metadata or db.session.get(ModelMetadata, 1)
    if metadata is None:
        return None
    if metadata.active_version_id:
        version = db.session.get(ModelVersion, metadata.active_version_id)
        if version is not None:
            return version
    return ModelVersion.query.filter_by(version=metadata.model_version).first()


def ensure_version_dataset_archives():
    """Backfill recoverable historical CSV archives without adding database tables."""
    versions = ModelVersion.query.all()
    if not versions:
        return

    current_dataset = training_dataset_dataframe()
    current_count = len(current_dataset)
    current_signatures = {
        training_dataset_content_signature(current_dataset),
        legacy_training_dataset_signature(),
    }
    original_path = PROJECT_ROOT / "data" / "Sleep_health_and_lifestyle_dataset.csv"
    original_dataset = None
    if original_path.is_file():
        original_dataset = normalize_training_dataset_dataframe(pd.read_csv(original_path))

    changed = False
    for version in versions:
        directory = _resolve_artifact_path(version.artifact_path)
        dataset_path = directory / DATASET_FILENAME
        if dataset_path.is_file():
            candidate = normalize_training_dataset_dataframe(pd.read_csv(dataset_path))
        elif version.training_record_count == current_count and version.training_dataset_signature in current_signatures:
            candidate = current_dataset
        elif (
            original_dataset is not None
            and version.version in {"v7", "v9"}
            and version.training_record_count == len(original_dataset)
        ):
            candidate = original_dataset
        else:
            continue

        signature, _ = _archive_dataset(
            directory,
            candidate,
            version.version,
            version.training_date,
        )
        if version.training_dataset_signature != signature:
            version.training_dataset_signature = signature
            changed = True

    metadata = db.session.get(ModelMetadata, 1)
    active = active_model_version(metadata)
    if metadata and active and metadata.training_dataset_signature != active.training_dataset_signature:
        metadata.training_dataset_signature = active.training_dataset_signature
        changed = True
    if changed:
        db.session.commit()


def available_model_versions():
    return sorted(ModelVersion.query.all(), key=lambda item: _version_number(item.version), reverse=True)


def evaluation_for_version(version):
    if version is None:
        return None, None
    classification = version.classification_metrics_data
    classification["confusion_matrix"] = version.confusion_matrix_data
    return classification, version.regression_metrics_data


def train_and_register_version(dataset_signature=None):
    """Run the existing training adapter and persist the resulting version bundle."""
    with _MODEL_OPERATION_LOCK:
        metadata = db.session.get(ModelMetadata, 1)
        dataset = training_dataset_dataframe()
        dataset_signature = training_dataset_content_signature(dataset)
        version_name = _next_version(metadata)
        backup = _backup_active_bundle()
        artifact_directory = None
        archive_created = False
        try:
            (trainer, classification, regression), record_count = train_models_from_database()
            if trainer is None or classification is None or regression is None:
                raise RuntimeError(
                    "Model training did not return complete classification and regression metrics."
                )
            _validate_bundle(MODELS_DIR)
            training_date = jakarta_now()
            artifact_directory, hashes, archive_created = _archive_active_bundle(version_name)
            archived_signature, _ = _archive_dataset(
                artifact_directory,
                dataset,
                version_name,
                training_date,
            )
            if archived_signature != dataset_signature:
                raise ValueError("Archived Training Dataset does not match the trained dataset.")
            version = _version_record(
                version=version_name,
                training_date=training_date,
                dataset_signature=dataset_signature,
                record_count=record_count,
                classification_metrics=classification,
                regression_metrics=regression,
                matrix=classification.get("confusion_matrix", []),
                artifact_directory=artifact_directory,
                hashes=hashes,
            )
            db.session.add(version)
            db.session.flush()
            if metadata is None:
                metadata = ModelMetadata(
                    id=1,
                    active_model="XGBoost Classifier and Regressor",
                    model_version=version_name,
                )
                db.session.add(metadata)
            metadata.model_version = version_name
            metadata.last_training_date = training_date
            metadata.training_dataset_signature = dataset_signature
            metadata.active_version_id = version.id
            db.session.commit()
            return version, classification, regression, record_count
        except Exception:
            db.session.rollback()
            _replace_active_bundle(backup)
            if archive_created and artifact_directory:
                shutil.rmtree(artifact_directory, ignore_errors=True)
            raise
        finally:
            shutil.rmtree(backup, ignore_errors=True)


def activate_model_version(version_id):
    """Restore a complete model, dataset, metrics, and metadata restore point."""
    with _MODEL_OPERATION_LOCK:
        version = db.session.get(ModelVersion, version_id)
        if version is None:
            raise ValueError("The selected model version does not exist.")
        metadata = db.session.get(ModelMetadata, 1)
        if metadata is None:
            raise RuntimeError("Active model metadata is not available.")

        source_directory = _resolve_artifact_path(version.artifact_path)
        _validate_bundle(source_directory, version.artifact_hashes_data)
        archived_dataset = _load_archived_dataset(version)
        current_signature = training_dataset_signature()
        if metadata.active_version_id == version.id and current_signature == version.training_dataset_signature:
            _validate_bundle(MODELS_DIR, version.artifact_hashes_data)
            return version, False

        backup = _backup_active_bundle()
        try:
            restored_count = restore_training_dataset_dataframe(archived_dataset)
            if restored_count != version.training_record_count:
                raise ValueError("Restored Training Dataset record count is invalid.")
            _replace_active_bundle(source_directory)
            _validate_bundle(MODELS_DIR, version.artifact_hashes_data)
            metadata.active_version_id = version.id
            metadata.model_version = version.version
            metadata.last_training_date = version.training_date
            metadata.training_dataset_signature = version.training_dataset_signature
            db.session.commit()
            return version, True
        except Exception:
            db.session.rollback()
            _replace_active_bundle(backup)
            raise
        finally:
            shutil.rmtree(backup, ignore_errors=True)

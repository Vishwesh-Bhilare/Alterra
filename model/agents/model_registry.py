"""
Registry of imported RL checkpoints -- lets any trained model (plain
stable_baselines3 PPO, sb3-contrib RecurrentPPO, or sb3-contrib
MaskablePPO for the hybrid doctrine+ML scheduler) be dropped into the GUI
without editing code. Models are copied into
model/agents/checkpoints/imported/ and tracked in manifest.json there;
the GUI's "Import Model..." dialog is the only way entries get added.
"""
from __future__ import annotations

import json
import re
import shutil
import uuid
from pathlib import Path

VALID_ALGO_CLASSES = ("PPO", "RecurrentPPO", "MaskablePPO")


def _registry_dir(repo_root: str | Path) -> Path:
    return Path(repo_root) / "model" / "agents" / "checkpoints" / "imported"


def _manifest_path(repo_root: str | Path) -> Path:
    return _registry_dir(repo_root) / "manifest.json"


def _slugify(label: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", label.strip().lower()).strip("_")
    return slug or "model"


def _load_manifest(repo_root: str | Path) -> list[dict]:
    path = _manifest_path(repo_root)
    if not path.exists():
        return []
    with open(path, "r") as f:
        return json.load(f).get("models", [])


def _save_manifest(repo_root: str | Path, models: list[dict]) -> None:
    path = _manifest_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"models": models}, f, indent=2)


def list_models(repo_root: str | Path) -> list[dict]:
    return _load_manifest(repo_root)


def register_model(repo_root: str | Path, source_path: str, label: str, algo_class: str) -> dict:
    if algo_class not in VALID_ALGO_CLASSES:
        raise ValueError(f"Unknown algo_class {algo_class!r}, expected one of {VALID_ALGO_CLASSES}")
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"Model file not found: {source_path}")

    registry_dir = _registry_dir(repo_root)
    registry_dir.mkdir(parents=True, exist_ok=True)

    model_id = f"{_slugify(label)}_{uuid.uuid4().hex[:6]}"
    filename = f"{model_id}.zip"
    shutil.copy2(source, registry_dir / filename)

    entry = {"id": model_id, "label": label, "algo_class": algo_class, "filename": filename}
    models = _load_manifest(repo_root)
    models.append(entry)
    _save_manifest(repo_root, models)
    return entry


def _find(repo_root: str | Path, model_id: str) -> dict:
    for entry in _load_manifest(repo_root):
        if entry["id"] == model_id:
            return entry
    raise KeyError(f"No registered model with id {model_id!r}")


def resolve_model_path(repo_root: str | Path, model_id: str) -> str:
    return str(_registry_dir(repo_root) / _find(repo_root, model_id)["filename"])


def get_algo_class(repo_root: str | Path, model_id: str) -> str:
    return _find(repo_root, model_id)["algo_class"]


def get_label(repo_root: str | Path, model_id: str) -> str:
    return _find(repo_root, model_id)["label"]

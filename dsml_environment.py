"""Runtime checks for the Python environment used by the lessons."""

from __future__ import annotations

import os
import sys
from pathlib import Path

EXPECTED_CONDA_ENV = "dsml_py_adv"


def _path_name(value: str | None) -> str | None:
    if not value:
        return None
    return Path(value).name


def _environment_candidates() -> set[str]:
    candidates = {
        _path_name(sys.prefix),
        _path_name(os.environ.get("CONDA_PREFIX")),
        os.environ.get("CONDA_DEFAULT_ENV"),
    }
    return {candidate for candidate in candidates if candidate}


def ensure_conda_environment(expected: str = EXPECTED_CONDA_ENV) -> str:
    """Raise a clear error when a chunk is not using the expected Python env."""

    candidates = _environment_candidates()
    if expected not in candidates:
        detected = _path_name(sys.prefix) or "unknown"
        raise RuntimeError(
            "Este material debe ejecutarse con el ambiente conda "
            f"'{expected}'. Ambiente detectado: '{detected}'. "
            f"Activalo con 'conda activate {expected}' o configura reticulate/Jupyter "
            "para usar ese interprete antes de ejecutar chunks de Python. "
            f"Python actual: {sys.executable}"
        )
    return expected

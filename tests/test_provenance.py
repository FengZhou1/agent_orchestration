from __future__ import annotations

import json

import pytest

from agent_orch.data.provenance import (
    StaleArtifactError,
    assert_same_library,
    file_sha256,
)


def _library(tmp_path, payload: dict) -> object:
    path = tmp_path / "library.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def test_accepts_an_artifact_from_the_same_library(tmp_path):
    path = _library(tmp_path, {"entries": [{"index": 0}]})
    digest = assert_same_library(
        {"library_json_sha256": file_sha256(path)}, path
    )
    assert digest == file_sha256(path)


def test_rejects_an_artifact_from_a_rebuilt_library(tmp_path):
    path = _library(tmp_path, {"entries": [{"index": 0}]})
    # A rebuilt library whose index-to-deployment mapping need not hold any more.
    stale = {"library_json_sha256": file_sha256(path)}
    path.write_text(json.dumps({"entries": [{"index": 1}]}, sort_keys=True), encoding="utf-8")
    with pytest.raises(StaleArtifactError, match="regenerate"):
        assert_same_library(stale, path)


def test_rejects_an_artifact_with_no_library_hash(tmp_path):
    path = _library(tmp_path, {"entries": [{"index": 0}]})
    with pytest.raises(StaleArtifactError, match="records no deployment-library hash"):
        assert_same_library({"entries": {}}, path)

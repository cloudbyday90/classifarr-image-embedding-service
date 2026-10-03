# Classifarr Image Embedding Service - companion service for Classifarr
# Copyright (C) 2024-2026 Classifarr Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest

from image_embedder import config
from image_embedder.deadlines import embedding_deadline, positive_duration


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch):
    monkeypatch.delenv("EMBEDDING_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("REQUEST_TIMEOUT_SECONDS", raising=False)
    monkeypatch.setattr(config, "_TOML", {})


def test_default_embedding_budget_is_independent_of_remote_hop_budget():
    settings = config.Settings()
    assert settings.embedding_timeout_seconds == 45
    assert settings.request_timeout_seconds == 15


@pytest.mark.parametrize("legacy", [0.1, 1, 45])
def test_custom_legacy_constructor_timeout_is_preserved(legacy):
    settings = config.Settings(request_timeout_seconds=legacy)
    assert settings.embedding_timeout_seconds == legacy


@pytest.mark.parametrize("legacy", ["1", "15", "45"])
def test_explicit_legacy_env_timeout_is_preserved(monkeypatch, legacy):
    monkeypatch.setenv("REQUEST_TIMEOUT_SECONDS", legacy)
    settings = config.Settings()
    assert settings.embedding_timeout_seconds == float(legacy)


def test_new_env_deadline_overrides_toml_and_legacy_without_changing_downloads(
    monkeypatch,
):
    monkeypatch.setenv("EMBEDDING_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("REQUEST_TIMEOUT_SECONDS", "2")
    monkeypatch.setattr(config, "_TOML", {"queue": {"embedding_timeout_seconds": 40}})
    settings = config.Settings()
    assert settings.embedding_timeout_seconds == 12.5
    assert settings.request_timeout_seconds == 2


def test_explicit_constructor_and_toml_deadlines_are_supported(monkeypatch):
    monkeypatch.setattr(config, "_TOML", {"queue": {"embedding_timeout_seconds": 40}})
    assert config.Settings().embedding_timeout_seconds == 40
    assert config.Settings(embedding_timeout_seconds=9).embedding_timeout_seconds == 9


def test_custom_legacy_toml_deadline_is_preserved(monkeypatch):
    monkeypatch.setattr(config, "_TOML", {"image": {"request_timeout_seconds": 7}})
    assert config.Settings().embedding_timeout_seconds == 7


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        True,
        False,
        "",
        "bad",
        "nan",
        "inf",
        float("nan"),
        float("inf"),
        10**400,
        {},
    ],
)
def test_nonfinite_or_invalid_deadlines_are_refused(value):
    with pytest.raises(ValueError, match="positive finite"):
        positive_duration(value)


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "bad"])
def test_invalid_embedding_env_fails_settings_creation(monkeypatch, value):
    monkeypatch.setenv("EMBEDDING_TIMEOUT_SECONDS", value)
    with pytest.raises(ValueError, match="positive finite"):
        config.Settings()


def test_invalid_explicit_constructor_deadline_is_refused():
    with pytest.raises(ValueError, match="positive finite"):
        config.Settings(embedding_timeout_seconds=float("inf"))


def test_explicit_new_budget_wins_over_custom_legacy_policy():
    assert embedding_deadline(5, 100, legacy_env_set=True) == 5

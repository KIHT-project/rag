from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import biomed_platform.core.settings as settings_mod


class TestSettingsModuleBDD:
    def test_given_project_structure_when_project_root_then_returns_root_path(self) -> None:
        root = settings_mod.project_root()
        assert root.exists()
        assert root.is_dir()

    def test_given_missing_configs_dir_when_configs_dir_then_raises_file_not_found(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        fake_root = tmp_path / "fake_root"
        fake_root.mkdir()

        monkeypatch.setattr(settings_mod, "project_root", lambda: fake_root)

        with pytest.raises(FileNotFoundError, match="Configs directory not found"):
            settings_mod.configs_dir()

    def test_given_missing_yaml_file_when_load_yaml_then_raises_file_not_found(
        self, tmp_path: Path
    ) -> None:
        missing = tmp_path / "missing.yaml"

        with pytest.raises(FileNotFoundError, match="Config file not found"):
            settings_mod._load_yaml(missing)

    def test_given_yaml_null_when_load_yaml_then_returns_empty_dict(
        self, tmp_path: Path
    ) -> None:
        cfg = tmp_path / "empty.yaml"
        cfg.write_text("", encoding="utf-8")

        result = settings_mod._load_yaml(cfg)

        assert result == {}

    def test_given_yaml_list_when_load_yaml_then_raises_value_error(
        self, tmp_path: Path
    ) -> None:
        cfg = tmp_path / "bad.yaml"
        cfg.write_text("- a\n- b\n", encoding="utf-8")

        with pytest.raises(ValueError, match="YAML mapping"):
            settings_mod._load_yaml(cfg)

    def test_given_valid_yaml_mapping_when_load_yaml_then_returns_dict(
        self, tmp_path: Path
    ) -> None:
        cfg = tmp_path / "ok.yaml"
        cfg.write_text(yaml.safe_dump({"a": 1}), encoding="utf-8")

        result = settings_mod._load_yaml(cfg)

        assert result == {"a": 1}

    def test_given_missing_logging_yaml_when_load_settings_then_raises_file_not_found(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        cfg_dir = tmp_path / "configs"
        cfg_dir.mkdir()

        monkeypatch.setattr(settings_mod, "configs_dir", lambda: cfg_dir)

        with pytest.raises(FileNotFoundError, match="Missing required logging config"):
            settings_mod.load_settings()

    def test_given_multiple_yaml_files_when_load_settings_then_loads_non_logging_configs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        cfg_dir = tmp_path / "configs"
        cfg_dir.mkdir()

        (cfg_dir / "logging.yaml").write_text("version: 1\nroot: {}\n", encoding="utf-8")
        (cfg_dir / "api.yaml").write_text("title: Test API\n", encoding="utf-8")
        (cfg_dir / "retrieval.yaml").write_text("top_k: 5\n", encoding="utf-8")

        monkeypatch.setattr(settings_mod, "configs_dir", lambda: cfg_dir)

        settings = settings_mod.load_settings()

        assert isinstance(settings, settings_mod.AppSettings)
        assert settings.config_dir == cfg_dir
        assert "api" in settings.by_name
        assert "retrieval" in settings.by_name
        assert "logging" not in settings.by_name

    def test_given_appsettings_when_require_existing_then_returns_config(self) -> None:
        s = settings_mod.AppSettings(
            config_dir=Path("/tmp"),
            by_name={"api": {"title": "X"}},
        )

        result = s.require("api")

        assert result == {"title": "X"}

    def test_given_appsettings_when_require_missing_then_raises_key_error(self) -> None:
        s = settings_mod.AppSettings(config_dir=Path("/tmp"), by_name={})

        with pytest.raises(KeyError, match="Missing config"):
            s.require("api")

    def test_given_appsettings_when_get_missing_then_returns_default(self) -> None:
        s = settings_mod.AppSettings(config_dir=Path("/tmp"), by_name={})

        result = s.get("api", default={"x": 1})

        assert result == {"x": 1}

    def test_given_appsettings_when_logging_path_then_points_to_logging_yaml(self) -> None:
        cfg_dir = Path("/configs")
        s = settings_mod.AppSettings(config_dir=cfg_dir, by_name={})

        assert s.logging_path == cfg_dir / "logging.yaml"

    def test_given_existing_configs_dir_when_configs_dir_then_returns_configs_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        fake_root = tmp_path / "fake_root"
        cfg_dir = fake_root / "configs"
        cfg_dir.mkdir(parents=True)

        monkeypatch.setattr(settings_mod, "project_root", lambda: fake_root)

        result = settings_mod.configs_dir()

        assert result == cfg_dir
        assert result.exists()
        assert result.is_dir()

    def test_given_appsettings_when_get_missing_without_default_then_returns_empty_dict(
        self
    ) -> None:
        s = settings_mod.AppSettings(config_dir=Path("/tmp"), by_name={})

        result = s.get("api")

        assert result == {}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

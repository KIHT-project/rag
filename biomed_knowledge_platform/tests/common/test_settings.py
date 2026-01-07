from __future__ import annotations

from pathlib import Path

import pytest

from biomed_platform.common import settings as settings_mod


class TestSettings:
    def _write(self, path: Path, content: str) -> None:
        path.write_text(content, encoding="utf-8")

    def test_project_root_returns_existing_directory(self) -> None:
        # Given

        # When
        root = settings_mod.project_root()

        # Then
        assert root.exists()
        assert root.is_dir()

    def test_configs_dir_uses_env_dir_when_set(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Given
        cfg_dir = tmp_path / "cfg"
        cfg_dir.mkdir()
        monkeypatch.setenv("BIOMED_CONFIG_DIR", str(cfg_dir))

        # When
        resolved = settings_mod.configs_dir()

        # Then
        assert resolved == cfg_dir.resolve()

    def test_configs_dir_raises_when_env_dir_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Given
        missing = tmp_path / "missing"
        monkeypatch.setenv("BIOMED_CONFIG_DIR", str(missing))

        # When
        with pytest.raises(FileNotFoundError) as exc:
            settings_mod.configs_dir()

        # Then
        assert "Configs directory not found" in str(exc.value)

    def test_load_yaml_raises_when_file_missing(self, tmp_path: Path) -> None:
        # Given
        p = tmp_path / "nope.yaml"

        # When
        with pytest.raises(FileNotFoundError) as exc:
            settings_mod._load_yaml(p)  # noqa: SLF001

        # Then
        assert "Config file not found" in str(exc.value)

    def test_load_yaml_returns_empty_dict_when_yaml_is_empty(self, tmp_path: Path) -> None:
        # Given
        p = tmp_path / "empty.yaml"
        self._write(p, "")

        # When
        data = settings_mod._load_yaml(p)  # noqa: SLF001

        # Then
        assert data == {}

    def test_load_yaml_raises_when_yaml_is_not_mapping(self, tmp_path: Path) -> None:
        # Given
        p = tmp_path / "bad.yaml"
        self._write(p, "- a\n- b\n")

        # When
        with pytest.raises(ValueError) as exc:
            settings_mod._load_yaml(p)  # noqa: SLF001

        # Then
        assert "Config must be a YAML mapping" in str(exc.value)

    def test_appsettings_logging_path_points_to_logging_yaml(self, tmp_path: Path) -> None:
        # Given
        s = settings_mod.AppSettings(config_dir=tmp_path, by_name={})

        # When
        p = s.logging_path

        # Then
        assert p == tmp_path / "logging.yaml"

    def test_appsettings_require_returns_config_when_present(self) -> None:
        # Given
        s = settings_mod.AppSettings(config_dir=Path("../../src"), by_name={"api": {"title": "x"}})

        # When
        cfg = s.require("api")

        # Then
        assert cfg == {"title": "x"}

    def test_appsettings_require_raises_when_missing(self) -> None:
        # Given
        s = settings_mod.AppSettings(config_dir=Path("../../src"), by_name={})

        # When
        with pytest.raises(KeyError) as exc:
            s.require("api")

        # Then
        assert "Missing config: api.yaml" in str(exc.value)

    def test_validate_required_configs_passes_when_all_present(self) -> None:
        # Given
        s = settings_mod.AppSettings(
            config_dir=Path("../../src"),
            by_name={
                "api": {"title": "x"},
                "rag": {"default_embedding_model_id": "e"},
                "qdrant": {"url": "http://localhost:6333"},
                "llm": {"ollama_base_url": "http://localhost:11434"},
            },
        )

        # When
        settings_mod._validate_required_configs(s)  # noqa: SLF001

        # Then
        assert True

    def test_validate_required_configs_raises_when_any_required_missing(self) -> None:
        # Given
        s = settings_mod.AppSettings(
            config_dir=Path("../../src"),
            by_name={
                "api": {"title": "x"},
                "rag": {"default_embedding_model_id": "e"},
                "qdrant": {"url": "http://localhost:6333"},
            },
        )

        # When
        with pytest.raises(KeyError) as exc:
            settings_mod._validate_required_configs(s)  # noqa: SLF001

        # Then
        assert "Missing config: llm.yaml" in str(exc.value)

    def test_load_settings_reads_all_yaml_and_skips_logging_yaml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Given
        cfg_dir = tmp_path / "configs"
        cfg_dir.mkdir()

        self._write(cfg_dir / "logging.yaml", "version: 1\n")
        self._write(cfg_dir / "api.yaml", "title: test\n")
        self._write(cfg_dir / "rag.yaml", "default_embedding_model_id: e\n")
        self._write(cfg_dir / "qdrant.yaml", "url: http://localhost:6333\n")
        self._write(cfg_dir / "llm.yaml", "ollama_base_url: http://localhost:11434\n")
        self._write(cfg_dir / "extra.yaml", "x: 1\n")

        monkeypatch.setenv("BIOMED_CONFIG_DIR", str(cfg_dir))

        # When
        s = settings_mod.load_settings()

        # Then
        assert s.config_dir == cfg_dir.resolve()
        assert s.logging_path == cfg_dir / "logging.yaml"
        assert "api" in s.by_name
        assert "rag" in s.by_name
        assert "qdrant" in s.by_name
        assert "llm" in s.by_name
        assert "extra" in s.by_name
        assert "logging" not in s.by_name

    def test_load_settings_raises_when_logging_yaml_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Given
        cfg_dir = tmp_path / "configs"
        cfg_dir.mkdir()

        self._write(cfg_dir / "api.yaml", "title: test\n")
        self._write(cfg_dir / "rag.yaml", "default_embedding_model_id: e\n")
        self._write(cfg_dir / "qdrant.yaml", "url: http://localhost:6333\n")
        self._write(cfg_dir / "llm.yaml", "ollama_base_url: http://localhost:11434\n")

        monkeypatch.setenv("BIOMED_CONFIG_DIR", str(cfg_dir))

        # When
        with pytest.raises(FileNotFoundError) as exc:
            settings_mod.load_settings()

        # Then
        assert "Missing required logging config" in str(exc.value)

    def test_given_env_var_not_set_when_configs_dir_then_uses_project_root_configs_dir(
            self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Given
        monkeypatch.delenv("BIOMED_CONFIG_DIR", raising=False)

        fake_root = Path("/tmp/fake-root")
        fake_configs = fake_root / "configs"

        monkeypatch.setattr(settings_mod, "project_root", lambda: fake_root)
        monkeypatch.setattr(Path, "exists", lambda self: self == fake_configs)

        # When
        resolved = settings_mod.configs_dir()

        # Then
        assert resolved == fake_configs

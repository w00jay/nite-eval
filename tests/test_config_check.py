"""Pre-run existence check for paths named in the llama-swap config.

llama-swap only discovers a missing binary or GGUF when it tries to swap that
model in, which is mid-run: the sweep is already going, earlier models have
already been scored, and the failure arrives as a load error rather than a
setup problem.

The case that motivated this is `bonsai2-27b-ptq1`, the one entry that does not
use the stock llama.cpp build. Its binary lives in a separate PrismML fork
checkout that nothing else in the repo references, so a rebuild or a clean
there breaks exactly one model and nothing warns first.
"""

from pathlib import Path

import pytest

from nite_eval.config_check import check_referenced_files

CONFIG = """\
models:
  "stock-model":
    cmd: >
      env CUDA_VISIBLE_DEVICES=GPU-1111
      {bin_dir}/llama-server
      -m {model_dir}/present.gguf
      --port ${{PORT}} -ngl 999
    group: "target-gpu"

  "fork-model":
    cmd: >
      env CUDA_VISIBLE_DEVICES=GPU-1111
      {bin_dir}/fork/llama-server
      -m {model_dir}/missing.gguf
      --port ${{PORT}} -ngl 999
    group: "target-gpu"

groups:
  "target-gpu":
    exclusive: true
"""


@pytest.fixture
def config(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "bin"
    (bin_dir / "fork").mkdir(parents=True)
    (bin_dir / "llama-server").touch()
    # The fork binary and one GGUF are deliberately absent.
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "present.gguf").touch()

    path = tmp_path / "llama_swap_config.yaml"
    path.write_text(CONFIG.format(bin_dir=bin_dir, model_dir=model_dir))
    return path


def test_reports_the_missing_binary_and_gguf(config: Path):
    problems = check_referenced_files(config)
    joined = "\n".join(problems)
    assert len(problems) == 2
    assert "fork/llama-server" in joined
    assert "missing.gguf" in joined


def test_says_nothing_about_paths_that_exist(config: Path):
    problems = check_referenced_files(config, model_names={"stock-model"})
    assert problems == []


def test_only_checks_the_models_being_run(config: Path):
    """A sweep of one model must not fail on another entry's missing file."""
    assert check_referenced_files(config, model_names={"stock-model"}) == []
    assert len(check_referenced_files(config, model_names={"fork-model"})) == 2


def test_unknown_model_name_checks_nothing(config: Path):
    assert check_referenced_files(config, model_names={"not-in-config"}) == []


def test_missing_config_is_reported_not_raised(tmp_path: Path):
    problems = check_referenced_files(tmp_path / "absent.yaml")
    assert len(problems) == 1
    assert "not found" in problems[0]


def test_malformed_yaml_is_reported_not_raised(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text("models: [unclosed\n")
    problems = check_referenced_files(path)
    assert len(problems) == 1
    assert "not valid YAML" in problems[0]


def test_relative_paths_are_ignored(tmp_path: Path):
    """Only absolute paths are checkable; a bare `llama-server` is on PATH."""
    path = tmp_path / "rel.yaml"
    path.write_text('models:\n  "m":\n    cmd: "llama-server -m model.gguf"\n')
    assert check_referenced_files(path) == []

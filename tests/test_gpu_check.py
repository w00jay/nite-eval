"""GPU placement verification tests."""

from pathlib import Path
from unittest.mock import patch

import pytest

from nite_eval.gpu_check import (
    Gpu,
    GpuPlacementError,
    Process,
    check_config_pinning,
    preflight,
    resolve_expected_uuids,
    verify_runtime_placement,
)

TARGET = "GPU-aaaaaaaa-0000-0000-0000-000000000001"
JUDGE = "GPU-bbbbbbbb-0000-0000-0000-000000000002"

GPUS = [
    Gpu(index=0, uuid=JUDGE, name="RTX 3060", total_mib=12288, used_mib=4000),
    Gpu(index=1, uuid=TARGET, name="RTX 3090", total_mib=24576, used_mib=18000),
]


def _proc(pid, uuid, port=None, used=1000):
    cmd = f"/bin/llama-server -m model.gguf --port {port}" if port else "/bin/llama-swap --config c.yaml"
    return Process(pid=pid, gpu_uuid=uuid, used_mib=used, cmdline=cmd)


def test_correct_placement_has_no_errors():
    procs = [_proc(1, TARGET), _proc(2, JUDGE, 9091), _proc(3, JUDGE, 9092)]
    with (
        patch("nite_eval.gpu_check.list_gpus", return_value=GPUS),
        patch("nite_eval.gpu_check.list_compute_processes", return_value=procs),
    ):
        errors, warnings = verify_runtime_placement(TARGET, JUDGE)
    assert errors == []
    assert warnings == []


def test_judge_on_target_gpu_is_an_error():
    """The failure that silently ruins a run: judge contending with the target."""
    procs = [_proc(1, TARGET), _proc(2, TARGET, 9091), _proc(3, JUDGE, 9092)]
    with (
        patch("nite_eval.gpu_check.list_gpus", return_value=GPUS),
        patch("nite_eval.gpu_check.list_compute_processes", return_value=procs),
    ):
        errors, _ = verify_runtime_placement(TARGET, JUDGE)
    assert len(errors) == 1
    assert "judge:reward-anything" in errors[0]
    assert "RTX 3090" in errors[0]  # reports where it actually is


def test_target_on_judge_gpu_is_an_error():
    procs = [_proc(1, JUDGE), _proc(2, JUDGE, 9091)]
    with (
        patch("nite_eval.gpu_check.list_gpus", return_value=GPUS),
        patch("nite_eval.gpu_check.list_compute_processes", return_value=procs),
    ):
        errors, _ = verify_runtime_placement(TARGET, JUDGE)
    assert any("target" in e for e in errors)


def test_low_headroom_is_a_warning_not_an_error():
    tight = [
        Gpu(index=0, uuid=JUDGE, name="RTX 3060", total_mib=12288, used_mib=11800),
        Gpu(index=1, uuid=TARGET, name="RTX 3090", total_mib=24576, used_mib=18000),
    ]
    procs = [_proc(1, TARGET), _proc(2, JUDGE, 9091)]
    with (
        patch("nite_eval.gpu_check.list_gpus", return_value=tight),
        patch("nite_eval.gpu_check.list_compute_processes", return_value=procs),
    ):
        errors, warnings = verify_runtime_placement(TARGET, JUDGE)
    assert errors == []
    assert len(warnings) == 1
    assert "488 MiB free" in warnings[0]


def test_missing_gpu_is_an_error():
    with (
        patch("nite_eval.gpu_check.list_gpus", return_value=[GPUS[0]]),
        patch("nite_eval.gpu_check.list_compute_processes", return_value=[]),
    ):
        errors, _ = verify_runtime_placement(TARGET, JUDGE)
    assert any("not present on this host" in e for e in errors)


def test_config_pinning_detects_wrong_uuid(tmp_path: Path):
    cfg = tmp_path / "swap.yaml"
    cfg.write_text(
        "models:\n"
        '  "a":\n'
        f"    cmd: env CUDA_VISIBLE_DEVICES={TARGET} llama-server\n"
        '  "b":\n'
        f"    cmd: env CUDA_VISIBLE_DEVICES={JUDGE} llama-server\n"
    )
    problems = check_config_pinning(cfg, TARGET)
    assert len(problems) == 1
    assert "swap.yaml:5" in problems[0]


def test_config_pinning_ignores_comments(tmp_path: Path):
    cfg = tmp_path / "swap.yaml"
    cfg.write_text(f"# CUDA_VISIBLE_DEVICES={JUDGE} is not the target\ncmd: env CUDA_VISIBLE_DEVICES={TARGET} x\n")
    assert check_config_pinning(cfg, TARGET) == []


def test_same_gpu_for_target_and_judge_is_rejected(monkeypatch):
    monkeypatch.setenv("TARGET_GPU_UUID", TARGET)
    monkeypatch.setenv("JUDGE_GPU_UUID", TARGET)
    with pytest.raises(GpuPlacementError, match="same GPU"):
        resolve_expected_uuids()


def test_missing_env_vars_are_rejected(monkeypatch):
    monkeypatch.delenv("TARGET_GPU_UUID", raising=False)
    monkeypatch.delenv("JUDGE_GPU_UUID", raising=False)
    with pytest.raises(GpuPlacementError, match="TARGET_GPU_UUID"):
        resolve_expected_uuids()


def test_preflight_raises_on_placement_error(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("TARGET_GPU_UUID", TARGET)
    monkeypatch.setenv("JUDGE_GPU_UUID", JUDGE)
    cfg = tmp_path / "swap.yaml"
    cfg.write_text(f"cmd: env CUDA_VISIBLE_DEVICES={TARGET} llama-server\n")
    procs = [_proc(2, TARGET, 9091)]  # judge on the target GPU
    with (
        patch("nite_eval.gpu_check.list_gpus", return_value=GPUS),
        patch("nite_eval.gpu_check.list_compute_processes", return_value=procs),
        pytest.raises(GpuPlacementError, match="GPU placement problems"),
    ):
        preflight(cfg, strict=True)

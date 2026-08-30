"""GPU placement verification for target and judge models.

Placement errors are silent corrupters. If a judge lands on the same GPU as the
model under evaluation they contend for VRAM and the target's layers spill to
host memory — the eval still completes and still writes scores, but latency
figures become meaningless and a large model can fail to load at all. Nothing
in the pipeline noticed this before; the only signal was a slow run.

Two checks, run at different times:

* `check_config_pinning` — static. Reads the llama-swap config and confirms
  every target model is pinned to the target GPU. Runs before servers start,
  so a misconfiguration costs nothing to catch.
* `verify_runtime_placement` — dynamic. Asks the driver which GPU each live
  llama-server process actually occupies. Catches the cases static config
  cannot: a stale server from a previous run, a manually started process, an
  ignored CUDA_VISIBLE_DEVICES.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

CUDA_ENV_RE = re.compile(r"CUDA_VISIBLE_DEVICES=(\S+)")

# Below this much free VRAM a co-resident process is likely to push the next
# allocation into host memory. Judges are small and static; the target is not.
LOW_HEADROOM_MIB = 1024


class GpuPlacementError(RuntimeError):
    """Raised when a model is on the wrong GPU, or placement cannot be verified."""


@dataclass(frozen=True)
class Gpu:
    index: int
    uuid: str
    name: str
    total_mib: int
    used_mib: int

    @property
    def free_mib(self) -> int:
        return self.total_mib - self.used_mib

    def __str__(self) -> str:
        return f"GPU {self.index} ({self.name}, {self.used_mib}/{self.total_mib} MiB)"


@dataclass(frozen=True)
class Process:
    pid: int
    gpu_uuid: str
    used_mib: int
    cmdline: str

    @property
    def role(self) -> str:
        """Classify by listening port — judges own REWARD_PORT and FLOW_PORT."""
        reward = os.environ.get("REWARD_PORT", "9091")
        flow = os.environ.get("FLOW_PORT", "9092")
        if f"--port {reward}" in self.cmdline:
            return "judge:reward-anything"
        if f"--port {flow}" in self.cmdline:
            return "judge:flow-judge"
        if "llama-server" in self.cmdline or "llama-swap" in self.cmdline:
            return "target"
        return "other"


def _nvidia_smi(query: str, entity: str = "gpu") -> list[list[str]]:
    flag = f"--query-{entity}={query}"
    try:
        out = subprocess.run(
            ["nvidia-smi", flag, "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        ).stdout
    except FileNotFoundError as e:
        raise GpuPlacementError("nvidia-smi not found — cannot verify GPU placement") from e
    except subprocess.CalledProcessError as e:
        raise GpuPlacementError(f"nvidia-smi failed: {e.stderr.strip()}") from e
    except subprocess.TimeoutExpired as e:
        raise GpuPlacementError("nvidia-smi timed out — driver may be wedged") from e

    return [[c.strip() for c in line.split(",")] for line in out.strip().splitlines() if line.strip()]


def list_gpus() -> list[Gpu]:
    rows = _nvidia_smi("index,uuid,name,memory.total,memory.used")
    return [Gpu(index=int(r[0]), uuid=r[1], name=r[2], total_mib=int(r[3]), used_mib=int(r[4])) for r in rows]


def _read_cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode(errors="replace").strip()
    except OSError:
        return ""


def list_compute_processes() -> list[Process]:
    rows = _nvidia_smi("pid,gpu_uuid,used_memory", entity="compute-apps")
    procs = []
    for r in rows:
        try:
            pid = int(r[0])
        except ValueError:
            continue
        procs.append(Process(pid=pid, gpu_uuid=r[1], used_mib=int(r[2]), cmdline=_read_cmdline(pid)))
    return procs


def _gpu_by_uuid(gpus: list[Gpu], uuid: str) -> Gpu | None:
    return next((g for g in gpus if g.uuid == uuid), None)


def resolve_expected_uuids() -> tuple[str, str]:
    """Return (target_uuid, judge_uuid) from the environment.

    run_nightly.sh sources .env before invoking the orchestrator, so these are
    the same values that pin the servers.
    """
    target = os.environ.get("TARGET_GPU_UUID", "")
    judge = os.environ.get("JUDGE_GPU_UUID", "")
    missing = [n for n, v in (("TARGET_GPU_UUID", target), ("JUDGE_GPU_UUID", judge)) if not v]
    if missing:
        raise GpuPlacementError(
            f"{' and '.join(missing)} not set — GPU placement cannot be verified. "
            "Set them in .env (see .env.example); nvidia-smi -L lists UUIDs."
        )
    if target == judge:
        raise GpuPlacementError(
            f"TARGET_GPU_UUID and JUDGE_GPU_UUID are the same GPU ({target}). "
            "Judges would contend with the model under evaluation for VRAM, "
            "inflating latency and risking a failed load. Assign separate GPUs."
        )
    return target, judge


def check_config_pinning(config_path: Path, target_uuid: str) -> list[str]:
    """Confirm every model in the llama-swap config pins to the target GPU.

    Returns a list of human-readable problems; empty means the config is clean.
    """
    if not config_path.exists():
        return [f"{config_path} not found — cannot verify target GPU pinning"]

    problems = []
    for lineno, line in enumerate(config_path.read_text().splitlines(), start=1):
        if "CUDA_VISIBLE_DEVICES" not in line or line.lstrip().startswith("#"):
            continue
        m = CUDA_ENV_RE.search(line)
        if not m:
            continue
        pinned = m.group(1)
        if pinned != target_uuid:
            problems.append(
                f"{config_path}:{lineno} pins CUDA_VISIBLE_DEVICES={pinned} but TARGET_GPU_UUID={target_uuid}"
            )
    return problems


REMEDIATION = (
    "Fix by pinning CUDA_VISIBLE_DEVICES to the intended GPU UUID in "
    "config/llama_swap_config.yaml (target) and JUDGE_GPU_UUID in .env (judges), "
    "then kill any stale llama-server processes before rerunning. "
    "`nvidia-smi -L` lists UUIDs."
)


def verify_runtime_placement(target_uuid: str, judge_uuid: str) -> tuple[list[str], list[str]]:
    """Confirm live llama-server processes sit on their assigned GPUs.

    Returns (errors, warnings). Errors mean a model is demonstrably on the
    wrong GPU and the run's numbers cannot be trusted. Warnings mean placement
    is correct but conditions are marginal.
    """
    gpus = list_gpus()
    errors: list[str] = []
    warnings: list[str] = []

    for label, uuid in (("target", target_uuid), ("judge", judge_uuid)):
        if _gpu_by_uuid(gpus, uuid) is None:
            errors.append(
                f"{label} GPU {uuid} is not present on this host. "
                f"Available: {', '.join(f'{g.index}={g.uuid}' for g in gpus)}"
            )
    if errors:
        return errors, warnings

    expected = {"target": target_uuid, "judge:reward-anything": judge_uuid, "judge:flow-judge": judge_uuid}
    for proc in list_compute_processes():
        want = expected.get(proc.role)
        if want is None:
            continue
        if proc.gpu_uuid != want:
            actual = _gpu_by_uuid(gpus, proc.gpu_uuid)
            intended = _gpu_by_uuid(gpus, want)
            errors.append(f"{proc.role} (pid {proc.pid}, {proc.used_mib} MiB) is on {actual}, expected {intended}")

    for label, uuid in (("target", target_uuid), ("judge", judge_uuid)):
        gpu = _gpu_by_uuid(gpus, uuid)
        if gpu and gpu.free_mib < LOW_HEADROOM_MIB:
            warnings.append(f"{label} {gpu} has only {gpu.free_mib} MiB free — allocations may spill to host memory")

    return errors, warnings


def preflight(config_path: Path, *, strict: bool = True) -> tuple[list[str], list[str]]:
    """Full placement check. Call before an eval run and after servers start.

    Returns (errors, warnings). Raises GpuPlacementError on errors when strict.
    """
    target_uuid, judge_uuid = resolve_expected_uuids()
    errors = check_config_pinning(config_path, target_uuid)
    runtime_errors, warnings = verify_runtime_placement(target_uuid, judge_uuid)
    errors += runtime_errors

    gpus = list_gpus()
    for label, uuid in (("target", target_uuid), ("judge", judge_uuid)):
        gpu = _gpu_by_uuid(gpus, uuid)
        if gpu:
            logger.info("GPU placement OK: %s -> %s", label, gpu)

    for w in warnings:
        logger.warning("GPU placement: %s", w)

    if errors and strict:
        raise GpuPlacementError("GPU placement problems detected:\n  - " + "\n  - ".join(errors) + "\n" + REMEDIATION)
    for e in errors:
        logger.error("GPU placement: %s", e)
    return errors, warnings

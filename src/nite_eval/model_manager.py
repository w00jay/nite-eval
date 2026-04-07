"""Manage llama-server and llama-swap processes for evaluation.

Handles:
- Starting the judge model on GPU 1 (persistent)
- Starting llama-swap on GPU 0 (target model hot-swapping)
- Health checking via /health endpoint
- Graceful shutdown
"""

import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

# Resolve from env (set in `.env` or shell). Falls back to PATH lookup.
LLAMA_SERVER_PATH = Path(os.environ.get("LLAMA_SERVER_BIN", "llama-server"))
HEALTH_POLL_INTERVAL = 1.0
HEALTH_POLL_TIMEOUT = 120.0


@dataclass
class ServerConfig:
    model_path: str
    port: int
    gpu_id: int
    ctx_size: int = 8192
    n_gpu_layers: int = 999
    extra_args: list[str] = field(default_factory=list)


@dataclass
class ManagedServer:
    config: ServerConfig
    process: subprocess.Popen | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.config.port}"

    @property
    def health_url(self) -> str:
        return f"{self.base_url}/health"

    @property
    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None


def start_server(config: ServerConfig, llama_server_path: Path = LLAMA_SERVER_PATH) -> ManagedServer:
    """Start a llama-server process with GPU isolation."""
    cmd = [
        str(llama_server_path),
        "-m",
        config.model_path,
        "--port",
        str(config.port),
        "-ngl",
        str(config.n_gpu_layers),
        "--ctx-size",
        str(config.ctx_size),
        "-fa",
        "on",
        "--cache-type-k",
        "q8_0",
        "--cache-type-v",
        "q8_0",
        "--no-webui",
        "--metrics",
        *config.extra_args,
    ]

    env = {"CUDA_VISIBLE_DEVICES": str(config.gpu_id)}
    logger.info("Starting llama-server: gpu=%d port=%d model=%s", config.gpu_id, config.port, config.model_path)

    process = subprocess.Popen(
        cmd,
        env={**subprocess.os.environ, **env},  # type: ignore[arg-type]
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    server = ManagedServer(config=config, process=process)
    return server


def wait_until_ready(server: ManagedServer, timeout: float = HEALTH_POLL_TIMEOUT) -> bool:
    """Poll /health until the server is ready or timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not server.is_running:
            logger.error("Server process died before becoming ready (pid=%s)", server.process and server.process.pid)
            return False
        try:
            resp = httpx.get(server.health_url, timeout=2.0)
            if resp.status_code == 200:
                logger.info("Server ready: %s", server.base_url)
                return True
        except httpx.ConnectError:
            pass
        time.sleep(HEALTH_POLL_INTERVAL)

    logger.error("Server failed to become ready within %.0fs", timeout)
    return False


def stop_server(server: ManagedServer, timeout: float = 10.0) -> None:
    """Gracefully stop a llama-server process."""
    if not server.is_running:
        return

    assert server.process is not None
    pid = server.process.pid
    logger.info("Stopping server pid=%d", pid)

    server.process.terminate()
    try:
        server.process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.warning("Server pid=%d did not stop gracefully, killing", pid)
        server.process.kill()
        server.process.wait(timeout=5.0)

    server.process = None
    logger.info("Server pid=%d stopped", pid)


def check_health(base_url: str, timeout: float = 5.0) -> bool:
    """Check if a server at base_url is healthy."""
    try:
        resp = httpx.get(f"{base_url}/health", timeout=timeout)
        return resp.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


def check_vllm_ready(base_url: str, timeout: float = 5.0) -> bool:
    """Check if a vLLM server is ready (model loaded, not just process up)."""
    try:
        resp = httpx.get(f"{base_url}/v1/models", timeout=timeout)
        if resp.status_code != 200:
            return False
        data = resp.json()
        return len(data.get("data", [])) > 0
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


@dataclass
class LlamaSwapConfig:
    """Configuration for llama-swap proxy."""

    config_path: str
    port: int = 9070
    process: subprocess.Popen | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None


def start_llama_swap(config: LlamaSwapConfig, binary_path: str = "llama-swap") -> LlamaSwapConfig:
    """Start the llama-swap proxy."""
    cmd = [binary_path, "--config", config.config_path, "--listen", f":{config.port}"]
    logger.info("Starting llama-swap: port=%d config=%s", config.port, config.config_path)

    config.process = subprocess.Popen(
        cmd,
        env={**subprocess.os.environ, "CUDA_VISIBLE_DEVICES": "0"},  # type: ignore[arg-type]
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return config


def warm_up_model(base_url: str, model_name: str, timeout: float = 60.0) -> bool:
    """Send a warm-up prompt to a model via llama-swap to trigger loading.

    Returns True if the model responded successfully.
    """
    try:
        resp = httpx.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": "Say OK."}],
                "max_tokens": 4,
                "temperature": 0.0,
            },
            timeout=timeout,
        )
        if resp.status_code == 200:
            logger.info("Model %s warmed up successfully", model_name)
            return True
        logger.error("Warm-up failed for %s: status=%d", model_name, resp.status_code)
        return False
    except (httpx.ConnectError, httpx.TimeoutException) as e:
        logger.error("Warm-up failed for %s: %s", model_name, e)
        return False

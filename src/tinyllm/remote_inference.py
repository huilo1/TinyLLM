from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shlex
import subprocess
import threading
import time


@dataclass(slots=True)
class RemoteGenerationConfig:
    max_new_tokens: int
    temperature: float
    top_k: int


@dataclass(slots=True)
class RemoteDemoConfig:
    generation: RemoteGenerationConfig
    run_dir: Path
    is_chat_model: bool = True


class RemoteWorkerUnavailable(RuntimeError):
    pass


class RemoteInferenceError(RuntimeError):
    pass


class HFRemoteSSHGenerator:
    def __init__(
        self,
        ssh_host: str,
        ssh_port: int,
        ssh_user: str,
        remote_workdir: str,
        remote_activate_path: str,
        remote_config_path: str,
        remote_adapter_path: str,
        report_dir: str | Path,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float = 0.9,
        startup_timeout_s: float = 180.0,
        startup_retry_interval_s: float = 3.0,
    ) -> None:
        self.ssh_host = ssh_host
        self.ssh_port = ssh_port
        self.ssh_user = ssh_user
        self.remote_workdir = remote_workdir
        self.remote_activate_path = remote_activate_path
        self.remote_config_path = remote_config_path
        self.remote_adapter_path = remote_adapter_path
        self.top_p = top_p
        self.startup_timeout_s = startup_timeout_s
        self.startup_retry_interval_s = startup_retry_interval_s
        self.config = RemoteDemoConfig(
            generation=RemoteGenerationConfig(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            ),
            run_dir=Path(report_dir),
        )
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None

    def _spawn_process(self) -> subprocess.Popen[str]:
        pythonpath = f"{self.remote_workdir}/src"
        worker_command = " ".join(
            [
                "python -u -m tinyllm.hf_worker",
                f"--config {shlex.quote(self.remote_config_path)}",
                f"--adapter {shlex.quote(self.remote_adapter_path)}",
            ]
        )
        remote_command = " && ".join(
            [
                f"cd {shlex.quote(self.remote_workdir)}",
                f"source {shlex.quote(self.remote_activate_path)}",
                f"export PYTHONPATH={shlex.quote(pythonpath)}",
                worker_command,
            ]
        )
        return subprocess.Popen(
            [
                "ssh",
                "-T",
                "-p",
                str(self.ssh_port),
                "-o",
                "BatchMode=yes",
                "-o",
                "ServerAliveInterval=30",
                "-o",
                "ServerAliveCountMax=3",
                "-o",
                "ConnectTimeout=10",
                "-o",
                "StrictHostKeyChecking=accept-new",
                f"{self.ssh_user}@{self.ssh_host}",
                f"bash -lc {shlex.quote(remote_command)}",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

    def _ssh_base_command(self) -> list[str]:
        return [
            "ssh",
            "-T",
            "-p",
            str(self.ssh_port),
            "-o",
            "BatchMode=yes",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=3",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "StrictHostKeyChecking=accept-new",
            f"{self.ssh_user}@{self.ssh_host}",
        ]

    def _read_json_message(self) -> dict:
        if self._process is None or self._process.stdout is None:
            raise RemoteWorkerUnavailable("Remote worker is not running.")
        last_line = ""
        while True:
            line = self._process.stdout.readline()
            if line == "":
                detail = f" Last output: {last_line}" if last_line else ""
                raise RemoteWorkerUnavailable(f"Remote worker closed the connection.{detail}")
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                last_line = line
                continue

    def _stop_process(self) -> None:
        if self._process is None:
            return
        if self._process.stdin is not None:
            try:
                self._process.stdin.close()
            except OSError:
                pass
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2)
        self._process = None

    def _cleanup_remote_workers(self) -> None:
        match = f"python -u -m tinyllm.hf_worker --config {self.remote_config_path} --adapter {self.remote_adapter_path}"
        cleanup_command = f"pkill -f -- {shlex.quote(match)} || true"
        subprocess.run(
            [
                *self._ssh_base_command(),
                f"bash -lc {shlex.quote(cleanup_command)}",
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )

    def _start_worker_once(self) -> None:
        self._process = self._spawn_process()
        message = self._read_json_message()
        if message.get("status") != "ready":
            raise RemoteWorkerUnavailable(f"Remote worker failed to start: {message}")

    def _start_worker(self) -> None:
        deadline = time.monotonic() + self.startup_timeout_s
        last_error: Exception | None = None
        while True:
            try:
                self._start_worker_once()
                return
            except Exception as exc:
                last_error = exc
                self._stop_process()
                if time.monotonic() >= deadline:
                    raise RemoteWorkerUnavailable(
                        f"Remote GPU worker did not become ready within {self.startup_timeout_s:.0f}s. "
                        f"Last error: {exc}"
                    ) from exc
                time.sleep(self.startup_retry_interval_s)

    def _ensure_process(self) -> None:
        if self._process is None or self._process.poll() is not None:
            self._start_worker()

    def _request(self, payload: dict) -> dict:
        with self._lock:
            last_error: Exception | None = None
            for _ in range(2):
                try:
                    self._ensure_process()
                    assert self._process is not None and self._process.stdin is not None
                    self._process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    self._process.stdin.flush()
                    response = self._read_json_message()
                except (BrokenPipeError, OSError, RemoteWorkerUnavailable) as exc:
                    last_error = exc
                    self._stop_process()
                    continue
                if not response.get("ok"):
                    raise RemoteInferenceError(response.get("error", "Remote inference failed."))
                return response
            raise RemoteWorkerUnavailable(f"Remote worker is unavailable: {last_error}")

    def unload(self) -> str:
        with self._lock:
            was_running = self._process is not None and self._process.poll() is None
            self._stop_process()
            self._cleanup_remote_workers()
        if was_running:
            return "Модель выгружена из памяти GPU. Следующий запрос поднимет worker заново."
        return "Модель уже не была загружена. Следующий запрос поднимет worker заново."

    def complete(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
    ) -> tuple[str, str]:
        response = self._request(
            {
                "kind": "raw_prompt",
                "raw_prompt": prompt,
                "max_new_tokens": max_new_tokens or self.config.generation.max_new_tokens,
                "temperature": temperature if temperature is not None else self.config.generation.temperature,
                "top_k": top_k if top_k is not None else self.config.generation.top_k,
                "top_p": self.top_p,
            }
        )
        return response["completion"], response["full_text"]

    def chat(
        self,
        user_message: str,
        system_prompt: str = "",
        history: list[dict] | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
    ) -> tuple[str, str, str]:
        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        for item in history or []:
            role = str(item.get("role", "")).strip()
            content = str(item.get("content", "")).strip()
            if role and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_message.strip()})
        response = self._request(
            {
                "kind": "messages",
                "messages": messages,
                "max_new_tokens": max_new_tokens or self.config.generation.max_new_tokens,
                "temperature": temperature if temperature is not None else self.config.generation.temperature,
                "top_k": top_k if top_k is not None else self.config.generation.top_k,
                "top_p": self.top_p,
            }
        )
        return response["completion"], response["prompt_text"], response["full_text"]

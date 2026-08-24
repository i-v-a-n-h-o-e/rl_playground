#!/usr/bin/env python3
"""End-to-end smoke test for the running development container."""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANAGER = ROOT / "c"
ENV_FILE = ROOT / ".docker" / ".env"


def run(*command: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(command))
    return subprocess.run(
        command,
        cwd=ROOT,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def manager(*arguments: str) -> subprocess.CompletedProcess[str]:
    return run(str(MANAGER), *arguments)


def compose(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run(
        "docker",
        "compose",
        "--project-directory",
        str(ROOT),
        "--env-file",
        str(ENV_FILE),
        "--file",
        str(ROOT / ".docker" / "compose.yml"),
        *arguments,
        check=check,
    )


def container_exec(*arguments: str) -> subprocess.CompletedProcess[str]:
    return compose("exec", "-T", "dev", *arguments)


def parse_env() -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key] = value.strip().strip("'\"")
    return values


def wait_for_health(container_name: str, timeout: float = 90) -> None:
    deadline = time.monotonic() + timeout
    last_status = "unknown"
    while time.monotonic() < deadline:
        result = run(
            "docker",
            "inspect",
            "--format",
            "{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}",
            container_name,
            check=False,
        )
        last_status = result.stdout.strip()
        if last_status == "healthy":
            return
        if last_status in {"exited", "dead"}:
            break
        time.sleep(2)
    logs = compose("logs", "--tail=200", "dev", check=False).stdout
    raise AssertionError(f"container did not become healthy (last={last_status})\n{logs}")


def find_private_key() -> Path:
    authorized = (ROOT / ".docker" / ".runtime" / "authorized_keys").read_text(encoding="utf-8")
    authorized_parts = authorized.split()
    if len(authorized_parts) < 2:
        raise AssertionError("invalid generated authorized_keys")
    fingerprint_material = authorized_parts[:2]
    for public_key in (Path.home() / ".ssh").glob("*.pub"):
        parts = public_key.read_text(encoding="utf-8").split()
        private_key = public_key.with_suffix("")
        if parts[:2] == fingerprint_material and private_key.is_file():
            return private_key
    raise AssertionError("private key matching .docker/.runtime/authorized_keys was not found")


def assert_contains(output: str, expected: str) -> None:
    if expected not in output:
        raise AssertionError(f"expected {expected!r} in output:\n{output}")


def main() -> None:
    values = parse_env()
    container_name = values["CONTAINER_NAME"]

    manager("start")
    wait_for_health(container_name)

    status = container_exec("supervisorctl", "status").stdout
    for service in ("app", "jupyter", "sshd"):
        assert_contains(status, service)
    if status.count("RUNNING") != 3:
        raise AssertionError(f"not all supervisor services are running:\n{status}")

    identity = container_exec("id", "-u", "developer").stdout.strip()
    assert identity == values["CONTAINER_UID"], (identity, values["CONTAINER_UID"])
    assert_contains(container_exec("python", "--version").stdout, "Python 3.12")
    assert_contains(container_exec("uv", "--version").stdout, "uv 0.11.32")
    uv_project = container_exec("bash", "-lc", 'printf %s "$UV_PROJECT"').stdout
    assert uv_project == "/workspace/.python_env", uv_project
    container_exec("uv", "lock", "--check")
    assert_contains(container_exec("cmake", "--version").stdout, "cmake version")
    assert_contains(container_exec("ninja", "--version").stdout, "1.")
    ml_smoke = container_exec("python", "examples/check_acceleration.py").stdout
    assert_contains(ml_smoke, "torchvision=")
    assert_contains(ml_smoke, "gym-maze=ok")
    assert_contains(ml_smoke, "accelerator=cpu")
    assert_contains(ml_smoke, "smoke=ok")

    container_exec(
        "bash",
        "-lc",
        "printf '#include <iostream>\\nint main(){std::cout << \"cpp-ok\";}\\n' "
        "> /tmp/smoke.cpp && g++ /tmp/smoke.cpp -o /tmp/smoke && /tmp/smoke",
    )

    mounts = run(
        "docker",
        "inspect",
        "--format",
        '{{range .Mounts}}{{println .Source "|" .Destination}}{{end}}',
        container_name,
    ).stdout
    assert_contains(mounts, f"{ROOT} | /workspace")

    sshd_settings = container_exec("sshd", "-T").stdout
    assert_contains(sshd_settings, "passwordauthentication no")
    assert_contains(sshd_settings, "permitrootlogin no")
    assert_contains(sshd_settings, "authenticationmethods publickey")

    with tempfile.NamedTemporaryFile() as known_hosts:
        ssh = run(
            "ssh",
            "-i",
            str(find_private_key()),
            "-p",
            values["SSH_PORT"],
            "-o",
            "BatchMode=yes",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            f"UserKnownHostsFile={known_hosts.name}",
            "developer@127.0.0.1",
            "printf ssh-ok",
        )
        assert ssh.stdout.rstrip().endswith("ssh-ok"), ssh.stdout

    query = urllib.parse.urlencode({"token": values["JUPYTER_TOKEN"]})
    with urllib.request.urlopen(
        f"http://127.0.0.1:{values['JUPYTER_PORT']}/api/status?{query}", timeout=10
    ) as response:
        assert response.status == 200
        assert isinstance(json.load(response), dict)

    try:
        urllib.request.urlopen(f"http://127.0.0.1:{values['JUPYTER_PORT']}/api/status", timeout=10)
    except urllib.error.HTTPError as error:
        assert error.code in {401, 403}, error.code
    else:
        raise AssertionError("Jupyter API accepted a request without its token")

    logs = compose("logs", "--tail=200", "dev").stdout
    assert_contains(logs, "worker: started")

    manager(
        "session",
        "integration-smoke",
        "--",
        "python",
        "-u",
        "-c",
        "import time; print('tmux-ok'); time.sleep(30)",
    )
    time.sleep(1)
    capture = manager("session", "integration-smoke", "--capture").stdout
    assert_contains(capture, "tmux-ok")
    container_exec("tmux", "kill-session", "-t", "integration-smoke")

    manager("restart")
    wait_for_health(container_name)
    manager("stop")
    remaining = run("docker", "inspect", container_name, check=False)
    if remaining.returncode == 0:
        raise AssertionError("./c stop did not remove the container")

    print("integration smoke test passed")


if __name__ == "__main__":
    main()

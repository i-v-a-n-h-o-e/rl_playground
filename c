#!/usr/bin/env python3
"""Manage the local Docker Compose development environment."""

from __future__ import annotations

import argparse
import os
import platform
import re
import secrets
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent
DOCKER_DIR = ROOT / ".docker"
ENV_FILE = DOCKER_DIR / ".env"
ENV_EXAMPLE = DOCKER_DIR / "env.example"
RUNTIME_DIR = DOCKER_DIR / ".runtime"
AUTHORIZED_KEYS = RUNTIME_DIR / "authorized_keys"
COMPOSE_FILE = DOCKER_DIR / "compose.yml"
COMPOSE_OVERRIDE_FILE = DOCKER_DIR / "compose.override.yml"
SERVICE = "dev"

ALIASES = {
    "i": "init",
    "b": "build",
    "u": "start",
    "s": "stop",
    "r": "restart",
    "a": "attach",
    "l": "logs",
    "p": "status",
    "j": "jupyter",
    "t": "session",
    "e": "exec",
    "h": "help",
}


class CommandError(RuntimeError):
    """A user-facing command error."""


def run(
    command: list[str],
    *,
    check: bool = True,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=ROOT,
            check=check,
            text=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
        )
    except FileNotFoundError as error:
        raise CommandError(f"command not found: {command[0]}") from error
    except subprocess.CalledProcessError as error:
        if capture and error.stderr:
            print(error.stderr.rstrip(), file=sys.stderr)
        raise CommandError(f"command failed with exit code {error.returncode}") from error


def compose(
    *arguments: str, check: bool = True, capture: bool = False
) -> subprocess.CompletedProcess[str]:
    command = [
        "docker",
        "compose",
        "--project-directory",
        str(ROOT),
        "--env-file",
        str(ENV_FILE),
        "--file",
        str(COMPOSE_FILE),
    ]
    if COMPOSE_OVERRIDE_FILE.exists():
        command.extend(["--file", str(COMPOSE_OVERRIDE_FILE)])
    command.extend(arguments)
    return run(command, check=check, capture=capture)


def parse_env(path: Path = ENV_FILE) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def replace_env_values(text: str, replacements: dict[str, str]) -> str:
    output: list[str] = []
    for line in text.splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key = line.split("=", 1)[0].strip()
            if key in replacements:
                line = f"{key}={replacements[key]}"
        output.append(line)
    return "\n".join(output) + "\n"


def project_slug() -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", ROOT.name.lower()).strip("-")
    return slug or "pet-project"


def default_public_key() -> Optional[Path]:
    ssh_dir = Path.home() / ".ssh"
    preferred = [
        ssh_dir / "id_ed25519.pub",
        ssh_dir / "id_ecdsa.pub",
        ssh_dir / "id_rsa.pub",
    ]
    return next((path for path in preferred if path.is_file()), None)


def validate_public_keys(contents: str) -> None:
    keys = [
        line.strip() for line in contents.splitlines() if line.strip() and not line.startswith("#")
    ]
    allowed_prefixes = ("ssh-ed25519 ", "ssh-rsa ", "ecdsa-sha2-", "sk-ssh-")
    if not keys or any(not key.startswith(allowed_prefixes) for key in keys):
        raise CommandError("the authorized keys file does not contain valid OpenSSH public keys")


def initialize(key_path: Optional[str], force: bool = False) -> None:
    RUNTIME_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)

    if force or not ENV_FILE.exists():
        if not ENV_EXAMPLE.exists():
            raise CommandError(f"missing template: {ENV_EXAMPLE}")
        slug = project_slug()
        if platform.system() == "Linux":
            uid, gid = os.getuid(), os.getgid()
        else:
            uid, gid = 1000, 1000
        contents = replace_env_values(
            ENV_EXAMPLE.read_text(encoding="utf-8"),
            {
                "COMPOSE_PROJECT_NAME": slug,
                "IMAGE_NAME": f"{slug}-dev",
                "CONTAINER_NAME": f"{slug}-dev",
                "CONTAINER_HOSTNAME": f"{slug}-dev",
                "JUPYTER_TOKEN": secrets.token_urlsafe(32),
                "CONTAINER_UID": str(uid),
                "CONTAINER_GID": str(gid),
            },
        )
        ENV_FILE.write_text(contents, encoding="utf-8")
        ENV_FILE.chmod(0o600)
        print(f"created {ENV_FILE.relative_to(ROOT)}")
    else:
        print(f"kept existing {ENV_FILE.relative_to(ROOT)}")

    if force or not AUTHORIZED_KEYS.exists():
        public_key = Path(key_path).expanduser() if key_path else default_public_key()
        if public_key is None or not public_key.is_file():
            raise CommandError(
                "no SSH public key found; generate one with 'ssh-keygen -t ed25519' "
                "or run './c init --key /path/to/key.pub'"
            )
        contents = public_key.read_text(encoding="utf-8")
        validate_public_keys(contents)
        AUTHORIZED_KEYS.write_text(contents.rstrip() + "\n", encoding="utf-8")
        AUTHORIZED_KEYS.chmod(0o600)
        print(f"copied {public_key} to {AUTHORIZED_KEYS.relative_to(ROOT)}")
    else:
        validate_public_keys(AUTHORIZED_KEYS.read_text(encoding="utf-8"))
        print(f"kept existing {AUTHORIZED_KEYS.relative_to(ROOT)}")


def ensure_initialized() -> None:
    if not ENV_FILE.exists() or not AUTHORIZED_KEYS.exists():
        print("initializing local configuration...")
        initialize(None)
    values = parse_env()
    if not values.get("JUPYTER_TOKEN") or values.get("JUPYTER_TOKEN") == "change-me":
        raise CommandError(
            "JUPYTER_TOKEN is not configured in .docker/.env; run './c init --force'"
        )
    validate_public_keys(AUTHORIZED_KEYS.read_text(encoding="utf-8"))


def ensure_docker() -> None:
    if shutil.which("docker") is None:
        raise CommandError("Docker is not installed or is not on PATH")
    run(["docker", "compose", "version"], capture=True)


def ensure_running() -> None:
    result = compose("ps", "--services", "--status", "running", capture=True)
    if SERVICE not in result.stdout.split():
        raise CommandError("container is not running; start it with './c start'")


def command_init(arguments: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="./c init")
    parser.add_argument("--key", help="path to an OpenSSH public key")
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace local .docker/.env and authorized_keys",
    )
    options = parser.parse_args(arguments)
    initialize(options.key, options.force)


def command_build(arguments: list[str]) -> None:
    ensure_initialized()
    ensure_docker()
    compose("build", *arguments)


def command_start(arguments: list[str]) -> None:
    ensure_initialized()
    ensure_docker()
    compose("up", "--detach", *arguments)
    compose("ps")
    print_access()


def command_stop(arguments: list[str]) -> None:
    ensure_initialized()
    ensure_docker()
    compose("down", *arguments)


def command_restart(arguments: list[str]) -> None:
    ensure_initialized()
    ensure_docker()
    compose("restart", *arguments)
    compose("ps")


def command_attach(arguments: list[str]) -> None:
    if arguments:
        raise CommandError("attach does not accept arguments; use './c exec COMMAND ...'")
    ensure_initialized()
    ensure_docker()
    ensure_running()
    compose("exec", SERVICE, "bash", "-l")


def command_exec(arguments: list[str]) -> None:
    ensure_initialized()
    ensure_docker()
    ensure_running()
    compose("exec", SERVICE, *(arguments or ["bash", "-l"]))


def command_logs(arguments: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="./c logs")
    parser.add_argument("--tail", default="200")
    parser.add_argument("--no-follow", action="store_true")
    options = parser.parse_args(arguments)
    ensure_initialized()
    ensure_docker()
    command = ["logs", f"--tail={options.tail}"]
    if not options.no_follow:
        command.append("--follow")
    command.append(SERVICE)
    compose(*command)


def command_status(arguments: list[str]) -> None:
    if arguments:
        raise CommandError("status does not accept arguments")
    ensure_initialized()
    ensure_docker()
    compose("ps")
    if compose("ps", "--services", "--status", "running", check=False, capture=True).stdout.strip():
        compose("exec", "-T", SERVICE, "supervisorctl", "status", check=False)


def command_ssh(arguments: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="./c ssh")
    parser.add_argument("--host", default="localhost")
    options = parser.parse_args(arguments)
    ensure_initialized()
    ensure_docker()
    if options.host == "localhost":
        ensure_running()
    port = parse_env().get("SSH_PORT", "2222")
    run(["ssh", "-p", port, f"developer@{options.host}"])


def command_jupyter(arguments: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="./c jupyter")
    parser.add_argument("--host", default="localhost")
    options = parser.parse_args(arguments)
    ensure_initialized()
    values = parse_env()
    port = values.get("JUPYTER_PORT", "8888")
    token = values["JUPYTER_TOKEN"]
    print(f"http://{options.host}:{port}/lab?token={token}")


def validate_session_name(name: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise CommandError(
            "session name may contain only letters, digits, dot, underscore and dash"
        )


def command_session(arguments: list[str]) -> None:
    ensure_initialized()
    ensure_docker()
    ensure_running()
    if not arguments:
        compose("exec", SERVICE, "tmux", "list-sessions", check=False)
        return

    name, *rest = arguments
    validate_session_name(name)
    if rest == ["--capture"]:
        compose("exec", "-T", SERVICE, "tmux", "capture-pane", "-p", "-S", "-", "-t", name)
        return
    if rest:
        if rest[0] != "--" or len(rest) == 1:
            raise CommandError("use './c session NAME -- COMMAND ...' to create a session")
        exists = compose(
            "exec", "-T", SERVICE, "tmux", "has-session", "-t", name, check=False, capture=True
        )
        if exists.returncode == 0:
            raise CommandError(f"tmux session already exists: {name}")
        shell_command = shlex.join(rest[1:])
        compose("exec", "-T", SERVICE, "tmux", "new-session", "-d", "-s", name, shell_command)
        print(f"started tmux session {name}; attach with './c session {name}'")
        return
    compose("exec", SERVICE, "tmux", "attach-session", "-t", name)


def command_pull(arguments: list[str]) -> None:
    ensure_initialized()
    ensure_docker()
    compose("pull", SERVICE, *arguments)


def command_config(arguments: list[str]) -> None:
    if arguments:
        raise CommandError("config does not accept arguments")
    ensure_initialized()
    ensure_docker()
    compose("config")


def command_doctor(arguments: list[str]) -> None:
    if arguments:
        raise CommandError("doctor does not accept arguments")
    ensure_initialized()
    ensure_docker()
    compose("config", "--quiet")
    print("configuration is valid")


def command_clean(arguments: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="./c clean")
    parser.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    options = parser.parse_args(arguments)
    ensure_initialized()
    ensure_docker()
    if not options.yes:
        answer = input("Remove containers and named volumes? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            print("cancelled")
            return
    compose("down", "--volumes", "--remove-orphans")
    print("containers and named volumes removed; .docker/.env and .docker/.runtime were preserved")


def print_access() -> None:
    values = parse_env()
    print(f"SSH:     ssh -p {values.get('SSH_PORT', '2222')} developer@localhost")
    print("Jupyter: ./c jupyter")
    print("Logs:    ./c logs")


def print_help() -> None:
    print(
        """Universal pet project container

Usage: ./c COMMAND [OPTIONS]

Commands:
  init, i       create .docker/.env and copy an SSH public key
  build, b      build the image
  start, u      start services in the background
  stop, s       stop and remove the container
  restart, r    restart services
  attach, a     open a login shell in the container
  exec, e       execute a command in the container
  ssh           connect through the container SSH server
  logs, l       follow SSH, Jupyter and supervised app logs
  status, p     show Compose and supervisor status
  jupyter, j    print the authenticated JupyterLab URL
  session, t    list, create, attach to, or capture tmux sessions
  pull          pull IMAGE_NAME:IMAGE_TAG from a registry
  config        print the resolved Compose configuration
  doctor        validate local prerequisites and Compose configuration
  clean         remove containers and named volumes (keeps local secrets)
  help, h       show this help

Examples:
  ./c init --key ~/.ssh/id_ed25519.pub
  ./c build
  ./c start
  ./c attach
  ./c session worker -- python examples/worker.py
  ./c session worker              # Ctrl-b d detaches
  ./c session worker --capture
  ./c logs
  ./c stop
"""
    )


COMMANDS = {
    "init": command_init,
    "build": command_build,
    "start": command_start,
    "stop": command_stop,
    "restart": command_restart,
    "attach": command_attach,
    "exec": command_exec,
    "logs": command_logs,
    "status": command_status,
    "ssh": command_ssh,
    "jupyter": command_jupyter,
    "session": command_session,
    "pull": command_pull,
    "config": command_config,
    "doctor": command_doctor,
    "clean": command_clean,
    "help": lambda _arguments: print_help(),
}


def main(arguments: Optional[list[str]] = None) -> int:
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    if not arguments:
        print_help()
        return 0
    name = ALIASES.get(arguments[0], arguments[0])
    command = COMMANDS.get(name)
    if command is None:
        print(f"unknown command: {arguments[0]}\n", file=sys.stderr)
        print_help()
        return 2
    try:
        command(arguments[1:])
    except CommandError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# RL playground

A reproducible reinforcement-learning environment with PyTorch, TorchVision, Gymnasium
(including classic control environments), OpenCV, Plotly, Matplotlib, NumPy, and JupyterLab. It runs in Ubuntu
24.04 containers on `linux/arm64` and `linux/amd64`, or natively on Apple Silicon when Metal GPU
acceleration is required.

The repository is mounted at `/workspace`; source code, application configuration, notebooks,
and project logs therefore remain ordinary files on the host. SSH, JupyterLab, and an optional
application process start automatically under `supervisord`.

## Supported machines

- Apple Silicon Mac: the local Linux container uses `arm64`.
- Raspberry Pi 3 Model B+ and Raspberry Pi 4 Model B: use a **64-bit** operating system and the
  `linux/arm64` image. The Pi 3 B+ has only 1 GB RAM, so enable swap and avoid large parallel
  builds or heavy ML workloads.
- Intel/AMD NAS or Linux server: use `linux/amd64`.

The template intentionally does not target 32-bit Raspberry Pi operating systems (`arm/v7`).
Check a Linux host with `uname -m`: `aarch64` is the expected value for `arm64`.

## Prerequisites

- Docker with Compose v2 (`docker compose version`);
- Python 3.9 or newer on the host for the `./c` manager;
- [uv](https://docs.astral.sh/uv/getting-started/installation/) for the native macOS environment;
- an OpenSSH public key, preferably `~/.ssh/id_ed25519.pub`.

## Quick start

### Reproducible CPU environment in Docker

```bash
./c init
./c build
./c start
./c status
./c exec python examples/check_acceleration.py
```

Docker uses all CPU cores made available in Docker Desktop and gets 2 GB of shared memory by
default. Increase Docker Desktop's VM CPU/memory limits if a workload needs more. The Linux
image intentionally installs the smaller CPU-only PyTorch build.

### Apple GPU acceleration with PyTorch MPS

Docker Desktop runs Linux in a VM and cannot expose the Mac's Metal GPU or Apple Neural Engine
to PyTorch. Run training natively when acceleration matters:

```bash
./c native-sync
./c native-check
uv run --project .python_env jupyter lab
```

The check should print `accelerator=mps` on a supported Apple Silicon Mac. Training code can
reuse `best_torch_device()` from `examples/check_acceleration.py`, or select the device directly:

```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)
```

The Neural Engine is not a general PyTorch training device. It is available through native
Apple frameworks such as Core ML, primarily for supported converted models and inference. See
[`docs/hardware.md`](docs/hardware.md) for the full decision guide.

`./c init` copies the first supported public key from `~/.ssh`, generates a random Jupyter
token, and writes both into ignored local files. To select a key explicitly:

```bash
./c init --key ~/.ssh/id_ed25519.pub
```

Do not commit `.docker/.env` or `.docker/.runtime/`; both are ignored.

The repository root intentionally stays small:

```text
.
├── .docker/       # image, Compose, services, local env and SSH state
├── .python_env/   # pyproject.toml, uv.lock and Python version
├── c              # container manager
├── examples/
├── tests/
└── README.md
```

## Manager commands

| Command | Alias | Purpose |
|---|---:|---|
| `./c init` | `i` | Create local secrets and configuration |
| `./c build` | `b` | Build the current architecture image |
| `./c start` | `u` | Start the container in the background |
| `./c stop` | `s` | Stop and remove the container |
| `./c restart` | `r` | Restart all services |
| `./c attach` | `a` | Open Bash through `docker compose exec` |
| `./c exec COMMAND` | `e` | Run a command in the container |
| `./c native-sync` | — | Create/update the host environment for Apple MPS |
| `./c native-check` | — | Smoke-test the ML stack and report acceleration |
| `./c ssh` | — | Open an SSH session to the local container |
| `./c logs` | `l` | Follow container and supervised application logs |
| `./c status` | `p` | Show Compose and supervisor status |
| `./c jupyter` | `j` | Print an authenticated JupyterLab URL |
| `./c session` | `t` | Manage detachable tmux sessions |
| `./c pull` | — | Pull the configured image from a registry |
| `./c doctor` | — | Validate prerequisites and Compose configuration |
| `./c clean` | — | Remove containers and named volumes |

Run `./c help` for examples. Extra build and Compose-up flags are forwarded, for example
`./c build --no-cache` and `./c start --build`.

## VS Code through Remote SSH

No Dev Containers extension is required. Install VS Code's **Remote - SSH** extension and add
the local container to `~/.ssh/config`:

```sshconfig
Host pet-project-local
    HostName localhost
    Port 2222
    User developer
    IdentityFile ~/.ssh/id_ed25519
```

Start the container, connect to `pet-project-local`, and open `/workspace`. VS Code installs
its server in a named volume, so extensions survive `./c stop` and the editor can debug with
`/opt/venv/bin/python` inside the container.

For a container running on a Raspberry Pi or NAS, change `HostName` to that host's LAN/VPN IP.
The SSH port is controlled by `SSH_PORT` in `.docker/.env`.

The container's SSH host keys persist in `.docker/.runtime/ssh`. Do not delete that directory
unless you expect clients to report a changed host key.

## JupyterLab

JupyterLab listens inside the container on port 8888 and uses the token generated in
`.docker/.env`.

```bash
./c jupyter
./c jupyter --host 192.168.1.42
```

Use the second form to print the URL for a Raspberry Pi/NAS and open it from an iPad. The
default `BIND_ADDRESS=0.0.0.0` exposes SSH and Jupyter to the host network; use it only on a
trusted LAN/VPN. Set `BIND_ADDRESS=127.0.0.1` for local-only access. Never forward these ports
directly from an internet router.

VS Code connected over SSH can select `/opt/venv/bin/python` as a notebook kernel without using
the browser endpoint.

## Python dependencies with uv

Runtime dependencies live in `.python_env/pyproject.toml`; development tools use the `dev`
dependency group. `.python_env/uv.lock` makes image builds reproducible.

The included runtime stack is:

- `torch` and `torchvision` (CPU-only wheels in Linux containers; native MPS-capable wheels on
  macOS);
- `gymnasium[classic-control]`;
- `opencv-python-headless` (the `cv2` API without GUI dependencies, suitable for notebooks and
  containers);
- `plotly`, `matplotlib`, and `numpy`.

Inside the container:

```bash
uv add numpy scipy pandas
uv add --dev pytest-cov
uv remove pandas
```

The image sets `UV_PROJECT=/workspace/.python_env`, so these commands work from `/workspace`
without extra flags. When using uv directly on the host, pass the project explicitly, for
example `uv --project .python_env lock --check`.

`/opt/venv` is writable by `developer`, so changes are usable immediately. Rebuild after a
dependency change to make it part of the image:

```bash
./c build
./c start --force-recreate
```

System packages are listed one per line in `.docker/system-packages.txt`. CMake, Ninja, GCC,
Clang, GDB, LLDB, pkg-config, and ccache are installed in the base image.

## Long-running processes

### Supervised application

`APP_COMMAND` in `.docker/.env` starts with the container, restarts after an unexpected
failure, and writes to standard output. The default demonstrates this with
`examples/worker.py`:

```bash
./c logs
./c status
./c restart
```

Set `APP_COMMAND=` to disable it or replace it with the project's bot/service command. Keep
secrets such as Telegram tokens in `.docker/.env`, not in `APP_COMMAND` or committed files.

### Detachable terminal with tmux

For experiments and manually managed programs:

```bash
./c session experiment -- python examples/worker.py
./c session experiment
# Press Ctrl-b, then d to detach.
./c session experiment --capture
```

The process survives SSH and terminal disconnections but not container recreation. Use the
supervised `APP_COMMAND` for a service that must start again after a reboot.

## Raspberry Pi or NAS workflow

1. Install a 64-bit OS, Docker Engine, Compose v2, Git, and Python 3 on the host.
2. Clone the fork on the target host.
3. Run `./c init --key /path/to/macbook-key.pub` (or place the MacBook key in
   `.docker/.runtime/authorized_keys`).
4. Run `./c build && ./c start`, or configure the GHCR image and use `./c pull && ./c start`.
5. Connect VS Code Remote SSH to the host IP and `SSH_PORT`.

Because the repository is a bind mount, host tools can inspect project files and any logs the
application writes under `/workspace/logs`. Container stdout remains available through
`./c logs`.

## Multi-architecture image in GHCR

`.github/workflows/container.yml` builds `linux/amd64` and `linux/arm64` with Buildx. Pull
requests validate the build; pushes to `main` and `v*` tags publish to:

```text
ghcr.io/OWNER/REPOSITORY:latest
```

The workflow authenticates with the repository `GITHUB_TOKEN`. After forking, ensure GitHub
Actions has permission to write packages. GHCR packages are private on first publication unless
their visibility is changed.

To run a published image, set these in `.docker/.env`:

```dotenv
IMAGE_NAME=ghcr.io/OWNER/REPOSITORY
IMAGE_TAG=latest
```

Then run `./c pull && ./c start`. Compose keeps the local `build` definition, so `./c build`
still creates a project-specific image when needed.

## Extending for hardware

USB/serial devices, GPIO, Bluetooth, and host networking are deliberately not enabled by
default. See [`docs/hardware.md`](docs/hardware.md) for scoped Compose override examples.

## Validation

Fast host-side checks do not require a running container or the ML environment:

```bash
python3 -m unittest discover -s tests -v
./c doctor
```

After `./c native-sync`, verify every requested library and the active accelerator with:

```bash
./c native-check
```

After building the image, the integration test exercises the health check, supervisor, C++
compiler, ML/RL imports and computations, bind mount, public-key SSH, authenticated Jupyter,
worker logs, tmux, restart, and stop:

```bash
python3 tests/integration.py
```

## Security notes

- SSH accepts public keys only; root login and passwords are disabled.
- Jupyter refuses to start with an empty/default token.
- `.docker/.env`, authorized keys, and persistent SSH host keys are local ignored files.
- The `developer` user has passwordless sudo because this is a development container. Remove
  `/etc/sudoers.d/developer` in a project that requires a stricter runtime.
- A source bind mount means code changes on the host immediately affect the running service.

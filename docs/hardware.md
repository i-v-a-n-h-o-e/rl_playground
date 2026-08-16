# Hardware and host networking extensions

The base Compose file deliberately does not grant access to host hardware. Add only the
permissions required by a concrete project in a local `.docker/compose.override.yml`. The
`./c` manager detects and loads this ignored project-specific file automatically.

## Apple Silicon: what can actually be accelerated

Docker Desktop runs the project's Linux container inside a virtual machine. macOS Metal/MPS
and Core ML are host frameworks, so neither the Apple GPU nor the Neural Engine appears as a
PyTorch device inside that Linux VM. `privileged: true`, a Compose `devices` mapping, and
`--gpus all` do not change this.

Choose the execution mode by workload:

| Workload | Recommended mode | Device |
|---|---|---|
| Reproducible development, tests, CPU rollouts | Docker | CPU |
| PyTorch training on a MacBook | Native macOS | `mps` GPU |
| Core ML model inference/deployment | Native macOS app/tools | CPU, GPU, or Neural Engine selected by Core ML |
| NVIDIA CUDA training | Linux/Windows host with an NVIDIA GPU | `cuda` |

For Apple GPU training, install the exact locked environment on the host and check it:

```bash
./c native-sync
./c native-check
```

Keep tensors and the model on the same device:

```python
from examples.check_acceleration import best_torch_device

device = best_torch_device()
model = model.to(device)
batch = batch.to(device)
```

Some PyTorch operations may not have an MPS kernel. During early development, the optional
environment variable below lets unsupported operations fall back to CPU (usually more slowly):

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 \
  uv run --project .python_env python your_training_script.py
```

Do not set this silently in the project: an unnoticed fallback can make performance confusing.
Profile first, reduce host-to-device transfers, and compare MPS against CPU for small networks;
GPU dispatch overhead can dominate tiny RL batches.

The Apple Neural Engine is not exposed as a PyTorch training backend. Core ML can schedule
supported model operations across CPU, GPU, and Neural Engine, but that is a separate native
deployment/inference workflow rather than a Docker or ordinary PyTorch RL training accelerator.

## Docker CPU resources

Compose does not cap CPU count, so the container can use every core assigned to Docker Desktop.
Memory and CPU allocation are controlled in Docker Desktop settings. `/dev/shm` defaults to
`2gb` in this repository to support PyTorch `DataLoader` workers and vectorized Gymnasium
environments; change `SHM_SIZE` in `.docker/.env` if needed.

For CPU-heavy experiments, tune thread and worker counts rather than multiplying both without
bounds. For example, many Gymnasium worker processes combined with PyTorch using every thread
per process can oversubscribe the Mac.

## Serial or USB device

```yaml
services:
  dev:
    devices:
      - /dev/ttyUSB0:/dev/ttyUSB0
    group_add:
      - dialout
```

On Linux, check the actual device and group with `ls -l /dev/ttyUSB0`. Device paths on macOS
are not passed directly through Docker Desktop in the same way as native Linux devices.

## Raspberry Pi GPIO

GPIO libraries differ in how they access the hardware. Prefer mapping the smallest required
devices, for example `/dev/gpiomem`, rather than enabling `privileged: true` for the whole
container. Add the matching group ID with `group_add` when the library supports it.

## Bluetooth

Bluetooth normally requires the host D-Bus socket and additional capabilities. Treat this as
a project-specific integration and review the security implications before adding the socket.

## Host networking

Linux-only services that rely on multicast discovery may need:

```yaml
services:
  dev:
    network_mode: host
    ports: []
```

When host networking is enabled, Compose port mappings are ignored. This behaves differently
under Docker Desktop, so keep it in a Linux-specific override rather than the base template.

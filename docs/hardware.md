# Hardware and host networking extensions

The base Compose file deliberately does not grant access to host hardware. Add only the
permissions required by a concrete project in a local `.docker/compose.override.yml`. The
`./c` manager detects and loads this ignored project-specific file automatically.

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

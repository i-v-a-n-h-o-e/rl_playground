# Review of the previous attempts

The canonical template was assembled after comparing `docker_prj` and
`docker_expirimental`.

## Reused ideas

- one short executable for build, start, shell access, and stop;
- a non-root development user with optional sudo;
- SSH and JupyterLab available in the same long-running environment;
- the complete repository mounted at `/workspace`;
- Compose-managed ports and persistent VS Code state;
- a preinstalled C/C++ build toolchain and common shell utilities.

## Replaced implementations

- fixed container names and ports became `.docker/.env` settings;
- password SSH became public-key-only authentication;
- unauthenticated Jupyter became a generated token configuration;
- background shell processes and `sleep infinity` became supervised services with signal
  handling and restart policies;
- duplicated and inconsistent Bash/Python launchers became one tested Python executable;
- invalid `vscode` users, `start-notebook.sh` calls, and mismatched mount paths were removed;
- obsolete VS Code settings and the unnecessary Dev Container definition were removed in
  favour of Remote SSH;
- architecture-specific local builds became an `amd64`/`arm64` Buildx workflow for GHCR.

# syntax=docker/dockerfile:1.7

FROM ghcr.io/astral-sh/uv:0.11.32 AS uv

FROM ubuntu:24.04

ARG DEVELOPER_USER=developer
ARG DEVELOPER_UID=1000
ARG DEVELOPER_GID=1000
ARG SOURCE_URL=""

LABEL org.opencontainers.image.title="Universal pet project environment" \
      org.opencontainers.image.description="Ubuntu-based Python, C++, JupyterLab and SSH development environment" \
      org.opencontainers.image.source="${SOURCE_URL}"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

COPY .docker/system-packages.txt /tmp/system-packages.txt

RUN apt-get update \
    && sed -e '/^[[:space:]]*#/d' -e '/^[[:space:]]*$/d' /tmp/system-packages.txt \
       | xargs apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/* /tmp/system-packages.txt \
    && rm -f /etc/ssh/ssh_host_*_key /etc/ssh/ssh_host_*_key.pub

COPY --from=uv /uv /uvx /usr/local/bin/

RUN existing_group="$(getent group "${DEVELOPER_GID}" | cut -d: -f1)" \
    && if [ -n "${existing_group}" ]; then \
         groupmod --new-name "${DEVELOPER_USER}" "${existing_group}"; \
       else \
         groupadd --gid "${DEVELOPER_GID}" "${DEVELOPER_USER}"; \
       fi \
    && existing_user="$(getent passwd "${DEVELOPER_UID}" | cut -d: -f1)" \
    && if [ -n "${existing_user}" ]; then \
         usermod --login "${DEVELOPER_USER}" \
           --home "/home/${DEVELOPER_USER}" --move-home \
           --shell /bin/bash "${existing_user}"; \
       else \
         useradd --uid "${DEVELOPER_UID}" --gid "${DEVELOPER_GID}" \
           --create-home --shell /bin/bash "${DEVELOPER_USER}"; \
       fi \
    && usermod --gid "${DEVELOPER_GID}" "${DEVELOPER_USER}" \
    && printf '%s ALL=(ALL) NOPASSWD:ALL\n' "${DEVELOPER_USER}" \
       > "/etc/sudoers.d/${DEVELOPER_USER}" \
    && chmod 0440 "/etc/sudoers.d/${DEVELOPER_USER}" \
    && mkdir -p /workspace /opt/venv /run/sshd \
    && chown -R "${DEVELOPER_USER}:${DEVELOPER_USER}" \
       /workspace /opt/venv "/home/${DEVELOPER_USER}"

WORKDIR /tmp/project
COPY .python_env/pyproject.toml .python_env/uv.lock .python_env/.python-version ./
RUN uv sync --locked --no-install-project \
    && chown -R "${DEVELOPER_USER}:${DEVELOPER_USER}" /opt/venv

COPY .docker/sshd_config /etc/ssh/sshd_config
COPY .docker/supervisord.conf /etc/supervisor/supervisord.conf
COPY .docker/entrypoint.sh /usr/local/bin/container-entrypoint
COPY .docker/start-app.sh /usr/local/bin/start-app
COPY .docker/start-jupyter.sh /usr/local/bin/start-jupyter

RUN chmod 0755 \
      /usr/local/bin/container-entrypoint \
      /usr/local/bin/start-app \
      /usr/local/bin/start-jupyter

WORKDIR /workspace

# Keep uv commands convenient from /workspace while storing their metadata out of the way.
ENV UV_PROJECT=/workspace/.python_env

EXPOSE 22 8888

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD supervisorctl -c /etc/supervisor/supervisord.conf status sshd \
      | grep -qE '^sshd[[:space:]]+RUNNING' \
      && supervisorctl -c /etc/supervisor/supervisord.conf status jupyter \
      | grep -qE '^jupyter[[:space:]]+RUNNING' || exit 1

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/container-entrypoint"]

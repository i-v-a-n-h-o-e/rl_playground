#!/usr/bin/env bash
set -Eeuo pipefail

developer_user="developer"
developer_home="/home/${developer_user}"
runtime_dir="/workspace/.docker/.runtime"
authorized_keys_source="${runtime_dir}/authorized_keys"
ssh_runtime_dir="${runtime_dir}/ssh"

fail() {
  echo "container-entrypoint: $*" >&2
  exit 1
}

if [[ ! -s "${authorized_keys_source}" ]]; then
  fail "${authorized_keys_source} is missing or empty; run ./c init on the host"
fi

requested_uid="${HOST_UID:-1000}"
requested_gid="${HOST_GID:-1000}"

if [[ ! "${requested_uid}" =~ ^[0-9]+$ || ! "${requested_gid}" =~ ^[0-9]+$ ]]; then
  fail "HOST_UID and HOST_GID must be numeric"
fi

current_uid="$(id -u "${developer_user}")"
current_gid="$(id -g "${developer_user}")"
identity_changed=false

if [[ "${requested_gid}" != "${current_gid}" ]]; then
  if getent group "${requested_gid}" >/dev/null; then
    echo "container-entrypoint: GID ${requested_gid} already exists; keeping ${current_gid}" >&2
  else
    groupmod --gid "${requested_gid}" "${developer_user}"
    current_gid="${requested_gid}"
    identity_changed=true
  fi
fi

if [[ "${requested_uid}" != "${current_uid}" ]]; then
  if getent passwd "${requested_uid}" >/dev/null; then
    echo "container-entrypoint: UID ${requested_uid} already exists; keeping ${current_uid}" >&2
  else
    usermod --uid "${requested_uid}" "${developer_user}"
    current_uid="${requested_uid}"
    identity_changed=true
  fi
fi

mkdir -p \
  "${developer_home}/.ssh" \
  "${developer_home}/.cache/uv" \
  "${developer_home}/.vscode-server" \
  "${ssh_runtime_dir}" \
  /run/sshd
if [[ "${identity_changed}" == true ]]; then
  chown -R "${developer_user}:${developer_user}" "${developer_home}" /opt/venv
else
  chown "${developer_user}:${developer_user}" \
    "${developer_home}/.cache/uv" "${developer_home}/.vscode-server"
fi
install -m 0600 -o "${developer_user}" -g "${developer_user}" \
  "${authorized_keys_source}" "${developer_home}/.ssh/authorized_keys"
chown "${developer_user}:${developer_user}" "${developer_home}/.ssh"
chmod 0700 "${developer_home}/.ssh"

if [[ ! -s "${ssh_runtime_dir}/ssh_host_ed25519_key" ]]; then
  ssh-keygen -q -t ed25519 -N '' -f "${ssh_runtime_dir}/ssh_host_ed25519_key"
fi
if [[ ! -s "${ssh_runtime_dir}/ssh_host_rsa_key" ]]; then
  ssh-keygen -q -t rsa -b 3072 -N '' -f "${ssh_runtime_dir}/ssh_host_rsa_key"
fi
chmod 0600 "${ssh_runtime_dir}"/ssh_host_*_key
chmod 0644 "${ssh_runtime_dir}"/ssh_host_*_key.pub
chown -R "${current_uid}:${current_gid}" "${runtime_dir}" 2>/dev/null || true

/usr/sbin/sshd -t -f /etc/ssh/sshd_config

if [[ -z "${JUPYTER_TOKEN:-}" || "${JUPYTER_TOKEN}" == "change-me" ]]; then
  fail "JUPYTER_TOKEN is not configured; run ./c init on the host"
fi

exec /usr/bin/supervisord -c /etc/supervisor/supervisord.conf

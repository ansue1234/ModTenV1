FROM ros:humble-ros-base

# Raspberry Pi 5 (arm64) or x86_64 host. Build with:  docker compose build
ARG UID=1000
ARG GID=1000
ARG USERNAME=ros

USER root

RUN set -eux; \
    if ! getent group "${USERNAME}" >/dev/null; then \
      groupadd -g "${GID}" "${USERNAME}"; \
    fi; \
    if ! id -u "${USERNAME}" >/dev/null 2>&1; then \
      useradd -m -u "${UID}" -g "${GID}" -s /bin/bash "${USERNAME}"; \
    fi; \
    usermod -aG dialout "${USERNAME}"; \
    chown -R "${UID}:${GID}" /home/"${USERNAME}"

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-venv \
      ros-humble-tf2-ros ros-humble-tf2-tools \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

USER ${USERNAME}
WORKDIR /ws

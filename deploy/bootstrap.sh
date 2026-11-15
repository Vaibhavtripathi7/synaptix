#!/usr/bin/env bash
# One-time setup for a fresh Ubuntu 24.04 server. Idempotent: safe to run again.
# Usage, on the server:  bash deploy/bootstrap.sh
# Inbound traffic is filtered by the cloud's network security group, not on the host, so this
# opens no ports; see docs/deploy.md for the rules.
set -euo pipefail

# Docker Engine and the compose plugin, from Docker's own apt repository (Ubuntu's is too old
# for the compose features this project uses).
if ! command -v docker >/dev/null; then
  sudo apt-get update
  sudo apt-get install -y ca-certificates curl
  sudo install -m 0755 -d /etc/apt/keyrings
  sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
    | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
  sudo apt-get update
  sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
  sudo usermod -aG docker "$USER"
  echo "Docker installed. Log out and back in once so the docker group applies."
fi

# A 2 GiB swap file as a safety net on a 4 GiB machine: a burst (two large uploads during two
# parses) slows down instead of being killed. Low swappiness keeps it a last resort, so the
# working set, Postgres's buffers above all, stays in RAM.
if [ -z "$(swapon --show --noheadings)" ]; then
  if [ ! -f /swapfile ]; then
    sudo fallocate -l 2G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
  fi
  sudo swapon /swapfile
  grep -q '^/swapfile ' /etc/fstab \
    || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
fi
echo 'vm.swappiness = 10' | sudo tee /etc/sysctl.d/99-synaptix.conf >/dev/null
sudo sysctl -q --system

# Security patches applied automatically.
sudo apt-get install -y unattended-upgrades >/dev/null
echo "Bootstrap complete."

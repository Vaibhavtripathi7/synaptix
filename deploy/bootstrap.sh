#!/usr/bin/env bash
# One-time setup for a fresh Ubuntu server (written for Oracle Cloud's Ampere A1 image).
# Idempotent: safe to run again. Usage, on the server:  bash deploy/bootstrap.sh
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

# Oracle's Ubuntu images reject every inbound connection except SSH in the host firewall, in
# addition to the cloud's security list. Accept HTTP and HTTPS (TCP, and UDP for HTTP/3).
open() {
  sudo iptables -C INPUT -p "$1" --dport "$2" -j ACCEPT 2>/dev/null \
    || sudo iptables -I INPUT 1 -p "$1" --dport "$2" -j ACCEPT
}
open tcp 80
open tcp 443
open udp 443
if command -v netfilter-persistent >/dev/null; then sudo netfilter-persistent save; fi

# Security patches applied automatically.
sudo apt-get install -y unattended-upgrades >/dev/null
echo "Bootstrap complete."

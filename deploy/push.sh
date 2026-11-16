#!/usr/bin/env bash
# Ship the working tree to the server and (re)build and start the stack.
#   deploy/push.sh azureuser@<hostname>
# Sends exactly the files git tracks or would track, so everything .gitignore excludes - .env,
# data/, .venv - never leaves this machine.
set -euo pipefail
host=${1:?usage: deploy/push.sh user@host}
dir=synaptix
compose="docker compose -f docker-compose.yml -f docker-compose.prod.yml --profile app"

git ls-files -z --cached --others --exclude-standard \
  | rsync -az --from0 --files-from=- ./ "$host:$dir/"
ssh "$host" "cd $dir && test -f .env || { echo 'missing $dir/.env on the server'; exit 1; }
  $compose up -d --build --remove-orphans && $compose ps"

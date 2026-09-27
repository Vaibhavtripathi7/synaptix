#!/usr/bin/env bash
# Nightly database backup, kept 14 days. On the server, from the repository root, add to cron:
#   15 3 * * *  cd ~/synaptix && bash deploy/backup.sh >> ~/backups/backup.log 2>&1
# Uploaded PDFs live in the synaptix-storage volume; documents can be re-ingested from them, so
# the database is what must not be lost. Copy ~/backups off the machine for real safety.
set -euo pipefail
out=~/backups
mkdir -p "$out"
stamp=$(date +%Y%m%d-%H%M%S)
docker compose -f docker-compose.yml -f docker-compose.prod.yml exec -T db \
  pg_dump -U synaptix -d synaptix --format=custom > "$out/synaptix-$stamp.dump"
find "$out" -name 'synaptix-*.dump' -mtime +14 -delete
echo "$(date -Is) backup ok: $out/synaptix-$stamp.dump"

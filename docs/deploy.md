# Deploying to a free VM

The whole stack — Caddy (HTTPS), two API replicas, a worker, Postgres with pgvector, the queue
Redis and the cache Redis — on one Oracle Cloud *Always Free* virtual machine: 4 ARM cores,
24 GB of memory, no charge. Budget about 45 minutes the first time.

What you get: `https://<your-name>.duckdns.org`, where anyone can use Synaptix with their own
free Gemini key. Your cost is zero, because users' keys pay for their own model calls.

## 1. Create the account and the VM

1. Sign up at [cloud.oracle.com](https://www.oracle.com/cloud/free/). A card is used to verify
   identity; Always Free resources are never charged.
   **Choose your home region carefully — it cannot be changed later, and the free ARM machines
   are only available in your home region.** Pick one near your users; if capacity turns out to
   be scarce there, see *Troubleshooting*.
2. *Compute → Instances → Create instance*:
   - **Image:** Canonical Ubuntu 24.04.
   - **Shape:** *Ampere* → `VM.Standard.A1.Flex`, **4 OCPUs, 24 GB memory** (the whole free
     allowance).
   - **Networking:** keep the default VCN and a public subnet; *Assign a public IPv4 address*.
   - **SSH keys:** upload your public key (`~/.ssh/id_ed25519.pub`; create one with
     `ssh-keygen -t ed25519` if you have none).
   - **Boot volume:** 100 GB is within the free 200 GB.
3. Note the instance's **public IP address**.

## 2. Open ports 80 and 443 in the cloud firewall

*Networking → Virtual cloud networks →* your VCN *→ Security Lists → Default security list →
Add ingress rules*:

| source CIDR | protocol | destination port |
|---|---|---|
| `0.0.0.0/0` | TCP | 80 |
| `0.0.0.0/0` | TCP | 443 |
| `0.0.0.0/0` | UDP | 443 (optional, enables HTTP/3) |

Port 80 must be open even though the site is HTTPS-only: Let's Encrypt proves you control the
domain by connecting to it.

## 3. A free hostname

1. Sign in at [duckdns.org](https://www.duckdns.org), create a subdomain (for example
   `synaptix-yourname`), and set its IP to the instance's public IP.
2. Check: `ping synaptix-yourname.duckdns.org` answers from that IP.

Any domain you own works too: point an `A` record at the IP.

## 4. Prepare the server

```sh
ssh ubuntu@<public-ip>
mkdir -p synaptix && exit
```

From your machine, in the repository root, send the code (only files git tracks; `.env`,
`data/` and `.venv` stay home):

```sh
git ls-files -z --cached --others --exclude-standard \
  | rsync -az --from0 --files-from=- ./ ubuntu@<public-ip>:synaptix/
```

On the server:

```sh
ssh ubuntu@<public-ip>
cd synaptix
bash deploy/bootstrap.sh        # Docker, host firewall for 80/443, automatic security updates
exit                            # log out and back in once, so the docker group applies
```

## 5. Configure and start

```sh
ssh ubuntu@<public-ip>
cd synaptix
cp deploy/env.production.example .env
nano .env                       # DOMAIN=synaptix-yourname.duckdns.org
                                # POSTGRES_PASSWORD=<output of: openssl rand -base64 32>
docker compose -f docker-compose.yml -f docker-compose.prod.yml --profile app up -d --build
```

The first build compiles nothing (every dependency ships ARM wheels) but downloads them; allow
a few minutes. Startup order is enforced: the databases become healthy, migrations run once,
then the worker, the API replicas and Caddy start. Caddy obtains the certificate on the first
request, which takes a few seconds.

## 6. Check it

```sh
curl https://synaptix-yourname.duckdns.org/readyz          # {"status":"ready"}
curl -I http://synaptix-yourname.duckdns.org/               # 308 → https
docker compose -f docker-compose.yml -f docker-compose.prod.yml ps
```

Open the site, paste a Gemini key under *Keys* (from
[aistudio.google.com/apikey](https://aistudio.google.com/apikey)), upload a PDF.

## Day two

**Deploy a change** — from your machine: `deploy/push.sh ubuntu@<public-ip>`. It sends the
code, rebuilds, runs migrations, then recreates changed containers. Expect a few seconds of
interruption while the API replicas restart: Compose recreates them together, it does not roll
them one by one. Zero-downtime deploys need an orchestrator that does (Kubernetes, Nomad, or a
blue/green pair behind Caddy) — worth it once people depend on the site.

**Logs** — `docker compose -f docker-compose.yml -f docker-compose.prod.yml logs -f api worker`.
They are JSON; every line of a request carries its `request_id`, including the lines the worker
writes for jobs that request queued. Logs rotate at 20 MB × 5 per container.

**Backups** — `crontab -e` on the server and add:
```
15 3 * * *  cd ~/synaptix && bash deploy/backup.sh >> ~/backups/backup.log 2>&1
```
Dumps are kept 14 days in `~/backups`. Copy them off the machine now and then: a backup on the
same disk does not survive losing the disk. Restore with
`docker compose ... exec -T db pg_restore -U synaptix -d synaptix --clean < <file>.dump`.

**Scale** — `deploy.replicas` of `api` in `docker-compose.prod.yml`; Caddy discovers replicas
through Docker's DNS, so no proxy change is needed. See [`capacity.md`](capacity.md) for when
one machine stops being enough and what changes then.

## What is and isn't reachable

Only Caddy publishes ports (80, 443). Postgres, both Redis instances, the API replicas and the
metrics endpoints are on the private compose network. Caddy adds HSTS, `nosniff`,
`no-referrer` and frame denial. The server stores no user API keys (see *Hosting it for other
people* in the README).

## Troubleshooting

- **"Out of host capacity" creating the A1 instance** — free ARM capacity is shared. Retry later
  (often within hours), or try a different availability domain in the same region.
- **The site doesn't load, `curl` times out** — a firewall. Check both layers: the security list
  (step 2) and the host firewall (`sudo iptables -L INPUT -n | head`; 80 and 443 must be
  `ACCEPT` above any `REJECT`; `bootstrap.sh` does this).
- **Certificate errors** — Caddy's log explains: `docker compose ... logs caddy`. Usually the
  DuckDNS record doesn't point at this IP yet, or port 80 is closed.
- **`missing POSTGRES_PASSWORD`** — `.env` is missing or incomplete; compose refuses to start
  rather than fall back to a default password.

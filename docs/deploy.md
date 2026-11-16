# Deploying to an Azure VM

The whole stack - Caddy (HTTPS), the API, a worker, Postgres with pgvector, the queue Redis and
the cache Redis - on one Azure virtual machine with 2 vCPUs and 4 GiB of memory, paid from Azure
for Students credit (no card needed). Budget about 45 minutes the first time.

What you get: `https://<label>.<region>.cloudapp.azure.com`, where anyone can use Synaptix with
their own free Gemini key. Users' keys pay for their own model calls; the VM is the only cost.

## 1. The account and its regions

1. Sign up at [azure.microsoft.com/free/students](https://azure.microsoft.com/free/students)
   with your college email. The offer is a credit valid for 12 months, renewable while you are a
   student. When the credit runs out, the subscription is disabled and the VM with it.
2. Student subscriptions may deploy only to a few regions, and the list differs per account.
   Find yours before creating anything: search **Policy** -> **Assignments** -> *Allowed resource
   deployment regions* -> the allowed locations. A VM in any other region fails with
   `RequestDisallowedByAzure`.
3. **Cost Management -> Budgets**: set the budget to the credit, with alerts at 50% and 80%. The
   credit cannot be overspent, but the alert is what warns you before the site goes dark.

## 2. Create the VM

*Virtual machines -> Create*:

| setting | value |
|---|---|
| Resource group | new: `synaptix` |
| Region | one of the allowed regions from step 1 |
| Image | Ubuntu Server 24.04 LTS - x64 Gen2 |
| Size | **Standard_B2als_v2** (2 vCPU, 4 GiB); Standard_B2s (2 vCPU, 4 GiB) if it is unavailable |
| Authentication | SSH public key, username `azureuser`, your `~/.ssh/id_ed25519.pub` (`ssh-keygen -t ed25519` if you have none) |
| Inbound ports | SSH (22), HTTP (80), HTTPS (443) |
| OS disk | 64 GiB Standard SSD, delete with VM |
| Public IP | new; delete it and the NIC with the VM |
| Auto-shutdown | **off**, or the site goes down every evening |

The size picker shows the monthly price; check it against the credit. Don't pick the sizes
listed as free (B1s, B2ats v2, B2pts v2): they have 1 GiB, and the worker alone peaks at about
1.2 GiB while parsing.

Port 80 must stay open even though the site is HTTPS-only: Let's Encrypt proves you control the
name by connecting to it. For HTTP/3, add an inbound rule for UDP 443 under *VM -> Networking*.

## 3. A hostname

VM -> **Overview** -> *DNS name: Not configured* -> set a label, for example `synaptix-yourname`.
The VM is now `synaptix-yourname.<region>.cloudapp.azure.com`. A Standard public IP is static,
so the name keeps pointing at the VM across restarts. Any domain you own works too: point an
`A` record at the public IP.

## 4. Prepare the server

```sh
ssh azureuser@<hostname>
mkdir -p synaptix && exit
```

From your machine, in the repository root, send the code (only files git tracks; `.env`,
`data/` and `.venv` stay home):

```sh
git ls-files -z --cached --others --exclude-standard \
  | rsync -az --from0 --files-from=- ./ azureuser@<hostname>:synaptix/
```

On the server:

```sh
ssh azureuser@<hostname>
cd synaptix
bash deploy/bootstrap.sh        # Docker, a 2 GiB swap file, automatic security updates
exit                            # log out and back in once, so the docker group applies
```

## 5. Configure and start

```sh
ssh azureuser@<hostname>
cd synaptix
cp deploy/env.production.example .env
nano .env                       # DOMAIN=<hostname>
                                # POSTGRES_PASSWORD=<output of: openssl rand -base64 32>
docker compose -f docker-compose.yml -f docker-compose.prod.yml --profile app up -d --build
```

The first build compiles nothing (every dependency ships prebuilt wheels) but downloads them;
allow a few minutes. Startup order is enforced: the databases become healthy, migrations run
once, then the worker and the API start, then Caddy. Caddy obtains the certificate on the first
request, which takes a few seconds.

## 6. Check it

```sh
curl https://<hostname>/readyz          # {"status":"ready"}
curl -I http://<hostname>/               # 308 -> https
docker compose -f docker-compose.yml -f docker-compose.prod.yml ps
```

Open the site, paste a Gemini key under *Keys* (from
[aistudio.google.com/apikey](https://aistudio.google.com/apikey)), upload a PDF.

## Day two

**Deploy a change** - from your machine: `deploy/push.sh azureuser@<hostname>`. It sends the
code, rebuilds, runs migrations, then recreates changed containers. Expect a few seconds of
interruption while the API restarts: Compose replaces containers, it does not roll them.
Zero-downtime deploys need an orchestrator that does (Kubernetes, Nomad, or a blue/green pair
behind Caddy) - worth it once people depend on the site.

**Logs** - `docker compose -f docker-compose.yml -f docker-compose.prod.yml logs -f api worker`.
They are JSON; every line of a request carries its `request_id`, including the lines the worker
writes for jobs that request queued. Logs rotate at 20 MB x 5 per container.

**Backups** - `crontab -e` on the server and add:
```
15 3 * * *  cd ~/synaptix && bash deploy/backup.sh >> ~/backups/backup.log 2>&1
```
Dumps are kept 14 days in `~/backups`. Copy them off the machine now and then
(`scp 'azureuser@<hostname>:backups/*.dump' .`): a backup on the same disk does not survive
losing the disk, and the VM is deleted with the subscription when the credit ends. Restore with
`docker compose ... exec -T db pg_restore -U synaptix -d synaptix --clean < <file>.dump`.

**Metrics and dashboards** - set `GRAFANA_ADMIN_PASSWORD` in `.env`, then start the stack with
the observability profile as well:
```sh
docker compose -f docker-compose.yml -f docker-compose.prod.yml \
  --profile app --profile observability up -d
```
Prometheus and Grafana listen on the server's localhost only, never publicly. From your machine,
tunnel to them: `ssh -L 3000:localhost:3000 -L 9090:localhost:9090 azureuser@<hostname>`, then
open `http://localhost:3000` (dashboard *Synaptix*) or `http://localhost:9090/alerts`.
Prometheus finds every API replica and the worker on its own; nothing to configure when you
scale.

**Scale** - resize the VM first, then raise `deploy.replicas` of `api` and the Postgres memory
settings in `docker-compose.prod.yml`. Caddy discovers replicas through Docker's DNS, so no
proxy change is needed. See [`capacity.md`](capacity.md) for when one machine stops being enough
and what changes then.

## What is and isn't reachable

Only Caddy publishes ports (80, 443). Postgres, both Redis instances, the API and the metrics
endpoints are on the private compose network. Caddy adds HSTS, `nosniff`, `no-referrer`, frame
denial, and a Content-Security-Policy that blocks inline script and any request to another
origin - the browser holds each user's Gemini key and workspace token, and that policy is what
keeps an injected script from sending them anywhere. The server stores no user API keys (see
*Hosting it for other people* in the README).

## Troubleshooting

- **`RequestDisallowedByAzure` creating the VM** - the region is not in your subscription's
  allowed list (step 1).
- **The site doesn't load, `curl` times out** - the network security group. *VM -> Networking*
  must show inbound rules allowing TCP 80 and 443 from any source.
- **Certificate errors** - Caddy's log explains: `docker compose ... logs caddy`. Usually the
  DNS label isn't set yet, `DOMAIN` in `.env` doesn't match it, or port 80 is closed. If the log
  shows a Let's Encrypt rate-limit error, use a name from [duckdns.org](https://www.duckdns.org)
  or your own domain instead.
- **Containers restarting** - `docker stats` shows memory against each limit. Starting the
  observability profile on 4 GiB leaves little headroom; the swap file absorbs bursts.
- **`missing POSTGRES_PASSWORD`** - `.env` is missing or incomplete; compose refuses to start
  rather than fall back to a default password.

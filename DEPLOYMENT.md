# Deploying NexWealth AI to Hugging Face Spaces

This documents the steps used to Dockerize the app and deploy it to Hugging
Face Spaces (Docker SDK), including the gotchas hit along the way.

## 1. Prerequisites

- Docker Desktop installed and running locally.
- A Hugging Face account.
- `huggingface_hub`'s `hf` CLI (ships with the `huggingface_hub` package
  already in `requirements.txt`; the older `huggingface-cli` binary is
  deprecated — use `hf`).
- `git-lfs` installed (`git lfs version` to check).

## 2. Dockerfile

[Dockerfile](Dockerfile) is a single-stage `python:3.11-slim` build:

- Installs `build-essential` + `curl` (curl needed for the `HEALTHCHECK`).
- Installs `requirements.txt`.
- Copies the full repo in.
- **Bakes the RAG index and seeded customer DB into the image at build
  time**: `RUN python scripts/build_rag_index.py && python
  scripts/seed_customers.py`. This means the deployed image is fully
  self-contained — no persistent volume needed, and it survives Space
  restarts/redeploys, since `corpus/` and `data/seed/customers.json` are
  static. (Trade-off: refreshing the corpus or reseeding customers requires
  a new image build, not a live re-index.)
- Serves on **port 7860** (`ENV PORT=7860`, `EXPOSE 7860`) — this is the
  port Hugging Face Spaces' Docker SDK expects by default.
- `HEALTHCHECK` hits `/_stcore/health` (Streamlit's built-in health
  endpoint).

## 3. docker-compose.yml (local dev only)

[docker-compose.yml](docker-compose.yml) maps host `8501` → container
`7860` so local `docker compose up --build` still opens on the familiar
`localhost:8501`. No volume mounts — the image is self-contained per §2.

## 4. .dockerignore

[.dockerignore](.dockerignore) excludes `.venv`, `.git`, caches, and the
*generated* data dirs (`data/chroma/`, `data/raw/`, `data/processed/`,
`data/profile.db`) — these get created fresh inside the image by the
build-time `RUN` steps, not copied from the host. `corpus/` and
`data/seed/` are **not** excluded since the build needs them.

## 5. README.md metadata block

Hugging Face Spaces requires a YAML frontmatter block at the very top of
`README.md` to recognize the Space config:

```yaml
---
title: NexWealth AI
emoji: 💬
colorFrom: indigo
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
---
```

## 6. Local verification (before deploying)

```powershell
docker compose up -d --build
# wait a few seconds, then:
Invoke-WebRequest http://localhost:8501/_stcore/health -UseBasicParsing
# expect: StatusCode 200, Content "ok"
docker logs financial-advisor-llm-app-1 --tail 60   # sanity-check no errors
docker compose down
```

## 7. Authenticate with Hugging Face

```powershell
hf auth login
# paste a write-scoped token from https://huggingface.co/settings/tokens
hf auth whoami   # confirms the logged-in username
```

## 8. Create the Space

```powershell
hf repo create <username>/<space-name> --type space --sdk docker
```

## 9. Push the code to the Space

Spaces are plain git repos. Two gotchas came up pushing this particular
repo:

**a) Binary files need Git LFS.** The `corpus/*.pdf` files were committed
as regular blobs in the GitHub history; Hugging Face's pre-receive hook
rejects binary files pushed outside LFS ("Please use
https://huggingface.co/docs/hub/xet to store binary files"). Since
rewriting the existing GitHub history wasn't desirable, the fix was to
push a **fresh orphan commit** (current file state only, no history) with
LFS tracking set up first:

```powershell
git lfs install
git lfs track "*.pdf"       # writes .gitattributes

git checkout --orphan hf-space
git add -A
git reset .claude            # keep local agent tooling out of the Space
git commit -m "Deploy NexWealth AI to Hugging Face Spaces (Docker SDK)"
```

**b) Auth for `git push` over HTTPS.** Non-interactive shells can't do the
username/password prompt Hugging Face's git server expects. Use the token
directly in the remote URL for the push, then immediately strip it back
out of the remote config:

```powershell
git remote add space https://huggingface.co/spaces/<username>/<space-name>

$token = (Get-Content "$env:USERPROFILE\.cache\huggingface\token" -Raw).Trim()
git remote set-url space "https://<username>:$token@huggingface.co/spaces/<username>/<space-name>"
git push space hf-space:main --force   # --force: overwrites the Space's auto-generated placeholder commit
git remote set-url space "https://huggingface.co/spaces/<username>/<space-name>"   # scrub token back out
```

Then return to your normal branch and drop the temporary one:

```powershell
git checkout main
git branch -D hf-space
```

`origin`/`upstream` (GitHub) are untouched by any of this — `space` is a
separate remote only used for deploys.

## 10. Set secrets on the Space

The app reads `HF_TOKEN` and `ALPHA_VANTAGE_KEY` from the environment
(same as local `.env`, see [`.env.example`](.env.example)). On Hugging
Face:

**Space page → Settings → Variables and secrets → New secret**

- `HF_TOKEN` — Hugging Face token used for LLM inference calls.
- `ALPHA_VANTAGE_KEY` — Alpha Vantage API key for live market data.

Without these the app still runs — it falls back to the deterministic
rule-based LLM floor and the last cached CSV price row — but without live
LLM answers or fresh quotes.

## 11. Watch the build

The Space rebuilds the Docker image on every push (same build as local:
installs deps, then builds the RAG index + seeds customers, ~4 min).
Build logs are visible on the Space page under the **"Logs"** tab. Once
it's live, the app is reachable two ways:

- `https://huggingface.co/spaces/<username>/<space-name>` — the Space's
  repo/community page, with the running app embedded plus a Files tab
  showing the source.
- `https://<username>-<space-name>.hf.space` — the direct URL to just the
  running app, no repo chrome.

## 12. Sharing without exposing code

There's no per-Space toggle for "public app, hidden code" — visibility is
repo-wide. Practical options:

| Approach | Code visibility | Access |
|---|---|---|
| Public Space, share the direct `.hf.space` URL | Technically browsable via the repo page, but not surfaced to people following your link | Anyone with the link |
| Private Space | Fully hidden | Only you + invited collaborators (HF login required) |

For a demo to trusted people, public Space + direct URL is simplest. For
sharing with strangers without exposing source, use a private Space and
add them as collaborators, or check the Space's Settings page for any
current temporary-link sharing option.

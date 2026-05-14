# Conduit — Agent Development Guide

## Host Map

| Host | Role | Headwater services |
|---|---|---|
| caruana | primary host | `headwaterrouter` (8081), `bywater` (8080) |
| alphablue | GPU worker host | `deepwater` (8080) |

---

## The Inner Loop

**1. Make code changes locally.**

**2. Deploy conduit:**
```bash
bash scripts/deploy.sh                # push + pull on both hosts
bash scripts/deploy.sh caruana        # caruana only
bash scripts/deploy.sh alphablue      # alphablue only
```

**3. Restart headwater** (required — headwater imports conduit as a library and caches it at startup):
```bash
# from $BC/headwater
bash scripts/deploy.sh caruana        # restarts headwaterrouter + bywater
bash scripts/deploy.sh alphablue      # restarts deepwater
bash scripts/deploy.sh                # both hosts
```

---

## Ground Rules

- **Never SSH directly.** All remote operations go through `scripts/deploy.sh` in this repo and `$BC/headwater/scripts/deploy.sh`.
- **Always restart headwater after deploying conduit.** `uv sync` alone is not enough — the running process has the old code in memory.

---

## Project Layout

```
conduit-project/
  src/conduit/       main library
  evals/             eval scaffolding, abstractions, and tests
  jobs/              runnable eval entry points (Cronicle jobs)
  scripts/deploy.sh  deploy to remote hosts
  pyproject.toml
```

## Evals

See **`evals/ARCHITECTURE.md`** for the full guide: three-layer design (scaffolding / abstraction / job), how to add a new eval, run matrix shape, Cronicle shell command template, and key file index.

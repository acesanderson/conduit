# Conduit — Agent Development Guide

## Deployment

Conduit has no standalone deploy script. To deploy conduit changes to a remote host:

1. Push local changes: `git push`
2. Use headwater's deploy script to pull and restart the relevant service:

```bash
# from $BC/headwater
bash scripts/deploy.sh caruana      # restarts headwaterrouter + bywater
bash scripts/deploy.sh alphablue    # restarts deepwater
bash scripts/deploy.sh              # both hosts
```

Headwater imports conduit as a library — a service restart is required after any conduit code change.

**Never SSH directly.** All remote operations go through `$BC/headwater/scripts/deploy.sh`.

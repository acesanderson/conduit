#!/usr/bin/env bash
# Deploy conduit code changes to remote servers.
#
# Usage:
#   ./scripts/deploy.sh [--no-push] [caruana|alphablue|botvinnik|all]
#
# Targets:
#   caruana   — pull on caruana
#   alphablue — pull on alphablue
#   botvinnik — pull on botvinnik
#   all       — all three (default)
#
# Flags:
#   --no-push   skip the local "git push to origin"; just pull on remotes
#
# Auth: SSH-based git auth.
#   - Each host's repo must have origin set to the SSH URL
#     (git@github.com:acesanderson/conduit.git).
#   - Each host loads middlegame via keychain in ~/.bash_profile.
#   - Remote git/uv invocations are wrapped in `bash -lc` so the
#     keychain-managed ssh-agent and ~/.local/bin are visible.
#
# Precondition: the repo must already be cloned on each target host at the
# paths listed in REMOTE_REPO below. This script does NOT clone — the human
# clones manually as a one-time bootstrap per host.

set -euo pipefail

LOCAL_REPO="$HOME/Brian_Code/conduit-project"
SSH_CLONE_URL="git@github.com:acesanderson/conduit.git"

declare -A REMOTE_REPO=(
    [caruana]="/home/bianders/Brian_Code/conduit-project"
    [alphablue]="/home/fishhouses/Brian_Code/conduit-project"
    [botvinnik]="/home/fishhouses/Brian_Code/conduit-project"
)

# --- parse args ---
NO_PUSH=0
TARGET="all"

for arg in "$@"; do
    case "$arg" in
        --no-push) NO_PUSH=1 ;;
        caruana|alphablue|botvinnik|all) TARGET="$arg" ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

case "$TARGET" in
    caruana)   HOSTS=(caruana) ;;
    alphablue) HOSTS=(alphablue) ;;
    botvinnik) HOSTS=(botvinnik) ;;
    all)       HOSTS=(caruana alphablue botvinnik) ;;
esac

# --- preflight: repo cloned on each target ---
for host in "${HOSTS[@]}"; do
    repo="${REMOTE_REPO[$host]}"
    ssh "$host" "test -d $repo/.git" || {
        echo "ERR: cannot verify repo on $host at $repo" >&2
        echo "     Either the host is unreachable, or the repo needs to be cloned:" >&2
        echo "     ssh $host && git clone $SSH_CLONE_URL $repo" >&2
        exit 1
    }
done

# --- preflight: local in sync with origin (skipped under --no-push) ---
if [[ $NO_PUSH -eq 0 ]]; then
    git -C "$LOCAL_REPO" fetch
    local_sha=$(git -C "$LOCAL_REPO" rev-parse HEAD)
    upstream_sha=$(git -C "$LOCAL_REPO" rev-parse '@{u}')
    base_sha=$(git -C "$LOCAL_REPO" merge-base HEAD '@{u}')

    if [[ "$local_sha" == "$upstream_sha" ]]; then
        :  # in sync
    elif [[ "$base_sha" == "$upstream_sha" ]]; then
        :  # local ahead; push will fast-forward
    elif [[ "$base_sha" == "$local_sha" ]]; then
        echo "ERR: local is behind origin. Pull first." >&2
        exit 1
    else
        echo "ERR: local and origin have diverged. Reconcile before deploying." >&2
        echo "     local:    $local_sha" >&2
        echo "     upstream: $upstream_sha" >&2
        exit 1
    fi
fi

# --- push ---
if [[ $NO_PUSH -eq 0 ]]; then
    echo "==> pushing to origin..."
    git -C "$LOCAL_REPO" push
else
    echo "==> skipping push (--no-push)"
fi

# --- pull + sync on each target ---
remote_pull() {
    local host="$1"
    local repo="${REMOTE_REPO[$host]}"
    echo "==> [$host] pulling code..."
    ssh "$host" "bash -lc 'git -C $repo pull --ff-only'"
    echo "==> [$host] syncing dependencies..."
    ssh "$host" "bash -lc 'cd $repo && uv sync'"
}

for host in "${HOSTS[@]}"; do
    remote_pull "$host"
done

# --- verification ---
echo
for host in "${HOSTS[@]}"; do
    repo="${REMOTE_REPO[$host]}"
    sha=$(ssh "$host" "git -C $repo rev-parse --short HEAD")
    echo "==> [$host] now at $sha"
done

echo "==> deploy complete"

#!/usr/bin/env bash
# Deploy conduit code changes to remote servers.
#
# Usage:
#   ./scripts/deploy.sh [caruana|alphablue|botvinnik|all]
#
# Targets:
#   caruana   — pull on caruana
#   alphablue — pull on alphablue
#   botvinnik — pull on botvinnik
#   all       — all three (default)
#
# After deploying conduit, restart headwater on the affected host(s) using
# $BC/headwater/scripts/deploy.sh — headwater imports conduit as a library
# and must be restarted to pick up changes.

set -euo pipefail

LOCAL_REPO="$HOME/Brian_Code/conduit-project"

declare -A REMOTE_REPO=(
    [caruana]="/home/bianders/Brian_Code/conduit-project"
    [alphablue]="/home/fishhouses/Brian_Code/conduit-project"
    [botvinnik]="/home/fishhouses/Brian_Code/conduit-project"
)

# --- parse args ---
TARGET="all"

for arg in "$@"; do
    case "$arg" in
        caruana|alphablue|botvinnik|all) TARGET="$arg" ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# --- push local changes ---
echo "==> pushing to origin..."
git -C "$LOCAL_REPO" push

# --- pull on each target ---
remote_pull() {
    local host="$1"
    local repo="${REMOTE_REPO[$host]}"
    echo "==> [$host] pulling code..."
    ssh "$host" "git -C $repo pull --ff-only https://${GITHUB_PERSONAL_TOKEN}@github.com/acesanderson/conduit.git"
    echo "==> [$host] syncing dependencies..."
    ssh "$host" "cd $repo && uv sync"
}

case "$TARGET" in
    caruana)
        remote_pull caruana
        ;;
    alphablue)
        remote_pull alphablue
        ;;
    botvinnik)
        remote_pull botvinnik
        ;;
    all)
        remote_pull caruana
        remote_pull alphablue
        remote_pull botvinnik
        ;;
esac

echo "==> sync complete"
echo "==> Remember: restart headwater on affected host(s) to pick up conduit changes:"
echo "    bash \$BC/headwater/scripts/deploy.sh $TARGET"

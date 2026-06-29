#!/usr/bin/env bash
# Push mFrag to GitHub (rename Sun-lab/ctCraft → mFrag, then force-replace with this tree).
#
# Prerequisites (run once on this machine):
#   gh auth login
#   # or: echo "$GITHUB_TOKEN" | gh auth login --with-token
#
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="Sun-lab/ctCraft"
NEW_NAME="mFrag"

if ! gh auth status >/dev/null 2>&1; then
  echo "ERROR: gh is not authenticated. Run: gh auth login"
  exit 1
fi

echo "==> Renaming ${REPO} to ${NEW_NAME} (skip if already renamed)"
gh repo rename "${NEW_NAME}" --repo "${REPO}" -y 2>/dev/null || \
  gh repo rename "${NEW_NAME}" --repo "Sun-lab/${NEW_NAME}" -y 2>/dev/null || true

REMOTE_REPO="Sun-lab/${NEW_NAME}"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git init -b main
fi

if ! git remote get-url origin >/dev/null 2>&1; then
  git remote add origin "https://github.com/${REMOTE_REPO}.git"
else
  git remote set-url origin "https://github.com/${REMOTE_REPO}.git"
fi

git add -A
git status

if git diff --cached --quiet; then
  echo "Nothing to commit."
else
  git commit -m "$(cat <<'EOF'
Initial public release of mFrag manuscript code and pretrained models.

Packages mFrag-Weight and mFrag-Net inference, coupling weights (Supplementary File 1),
figure reproduction scripts, and lung-cancer pretrained checkpoints.
EOF
)"
fi

echo ""
echo "==> Pushing to https://github.com/${REMOTE_REPO}"
echo "    If the remote has unrelated history, use: git push -u origin main --force"
read -r -p "Force push and replace remote main? [y/N] " ans
if [[ "${ans}" =~ ^[Yy]$ ]]; then
  git push -u origin main --force
else
  git push -u origin main
fi

echo "Done: https://github.com/${REMOTE_REPO}"

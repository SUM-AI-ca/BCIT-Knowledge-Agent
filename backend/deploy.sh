#!/usr/bin/env bash
# Deploy backend modules to the production VM.
#
#   bash backend/deploy.sh                 # deploy what changed vs the VM's commit
#   bash backend/deploy.sh config.py ...   # or name the files explicitly
#   bash backend/deploy.sh --deps ...      # ALSO rebuild the VM venv first
#
# --deps is required whenever requirements.txt changed. This script otherwise
# copies .py files into the venv that is already there and restarts, so a
# dependency change deploys code against libraries it was never tested with —
# in the August 2026 graph round that would have been langchain 1.x code
# running on 0.3.x, plus an ImportError on langgraph, which is not installed.
# It builds .venv-new alongside the live one, import-smokes it, and only then
# swaps: the service keeps serving from the old venv until the last moment,
# and rollback is `mv .venv .venv-bad && mv .venv-old .venv`.
#
# Exists because the two ways this deploy goes wrong are both documented
# gotchas that a one-liner walks straight into: a long pasted command loses
# spaces at terminal wrap points, and `gcloud compute ssh` needs its own
# `gcloud auth login` — separate from the ADC login that `run_eval.py` uses.
#
# It also refuses to deploy nothing. The pipeline lives in query_rag.py,
# hybrid_retriever.py, reranker.py and config.py, so a change can touch none
# of server.py; a server.py-only deploy restarts the service on old code and
# still reports healthy.
set -euo pipefail
cd "$(dirname "$0")/.."

VM=bcit-rag-vm
ZONE=us-west1-b
# Passed explicitly on every call: `gcloud auth login` leaves the active
# project at whatever the account defaults to (info@sumai.ca lands on
# sumai-web-2026), and the VM lives elsewhere. Relying on ambient config means
# the deploy fails with "instance not found" after the upload has already run.
PROJECT=wine-agent-jh-2026
REMOTE=/opt/bcit-rag/backend
DEPLOYABLE=(config.py query_rag.py hybrid_retriever.py reranker.py server.py embeddings.py response_cache.py session_memory.py graph.py)

DEPS=0
if [ "${1:-}" = "--deps" ]; then DEPS=1; shift; fi

if [ $# -gt 0 ]; then
  FILES=("$@")
else
  # Default to every deployable module that differs from origin/main~1, which
  # is the closest cheap proxy for "what changed in this deploy". Override by
  # passing filenames when that is not what you mean.
  mapfile -t FILES < <(
    git diff --name-only HEAD~1..HEAD -- 'backend/*.py' \
      | grep -v '^backend/eval/' | xargs -r -n1 basename
  )
fi

if [ ${#FILES[@]} -eq 0 ]; then
  echo "Nothing to deploy: no backend module changed in the last commit."
  echo "Pass filenames explicitly if you meant something else, e.g."
  echo "  bash backend/deploy.sh query_rag.py"
  exit 1
fi

echo "==> checking the VM is reachable in $PROJECT before uploading anything"
gcloud compute instances describe "$VM" --zone="$ZONE" --project="$PROJECT" \
  --format='value(name,status)' || {
  echo
  echo "Cannot see $VM in $PROJECT as $(gcloud config get-value account 2>/dev/null)."
  echo "Check the logged-in account has access to $PROJECT."
  exit 1
}
echo
echo "About to deploy to $VM ($ZONE, $PROJECT):"
printf '  %s\n' "${FILES[@]}"
echo
echo "Local HEAD: $(git log --oneline -1)"

# The prompt is there to catch an interactive mistake. When stdin is not a
# terminal there is nobody to answer it, and `read` returning EOF under
# `set -e` would kill the script silently — no output, no deploy, no clue why.
# Running this command with an explicit file list IS the confirmation.
if [ -t 0 ]; then
  ok=""
  read -r -p "Continue? [y/N] " ok || true
  [ "$ok" = "y" ] || { echo "aborted"; exit 1; }
else
  echo "(stdin is not a terminal — proceeding without the prompt)"
fi

SRC=()
for f in "${FILES[@]}"; do
  [ -f "backend/$f" ] || { echo "no such file: backend/$f"; exit 1; }
  SRC+=("backend/$f")
done

echo "==> uploading"
[ "$DEPS" = 1 ] && SRC+=("backend/requirements.txt" "backend/.python-version")
gcloud compute scp "${SRC[@]}" "$VM:/tmp/" --zone="$ZONE" --project="$PROJECT"

if [ "$DEPS" = 1 ]; then
  echo
  echo "==> rebuilding the venv alongside the live one (service keeps serving)"
  # Built fresh rather than upgraded in place: a pip failure half-way through
  # an in-place upgrade leaves a venv that neither starts nor rolls back
  # cleanly. Here the running service is untouched until the mv, and the
  # import smoke runs against the NEW venv before anything is switched.
  gcloud compute ssh "$VM" --zone="$ZONE" --project="$PROJECT" --command="
    set -e
    UV=/home/park/.local/bin/uv
    test -x \$UV
    # Version comes from pyproject's requires-python, not from a path pinned
    # here: this line was hard-coded to 3.10 once and had to be edited during
    # the 3.13 upgrade, which is exactly the kind of edit that gets forgotten.
    # uv downloads the interpreter if the VM does not have it.
    PYVER=\$(tr -d '[:space:]' < $REMOTE/.python-version 2>/dev/null || true)
    PYVER=\${PYVER:-3.13}
    cd $REMOTE
    df -h /opt | tail -1
    sudo cp /tmp/requirements.txt $REMOTE/requirements.txt
    sudo cp /tmp/.python-version $REMOTE/.python-version
    # The modules have to be in place before the smoke, or it imports the
    # code that is being replaced — and a brand new module like graph.py is
    # not there at all, so the smoke would fail for the wrong reason.
    for f in ${FILES[*]}; do sudo cp /tmp/\$f $REMOTE/\$f; done
    sudo rm -rf .venv-new
    # uv, not python3 -m venv: the VM's system python3 has no ensurepip, and
    # the interpreter this project runs on is whatever uv manages, not
    # whatever the distro ships.
    #
    # --relocatable is load-bearing. Without it uv writes each console script
    # with a shebang naming its build path (#!/opt/bcit-rag/backend/.venv-new/
    # bin/python), so the moment .venv-new is renamed to .venv every entry
    # point points at a directory that no longer exists. systemd reports that
    # as 'Failed to execute .../uvicorn: No such file or directory' even
    # though uvicorn is installed, and the service crash-loops. Learned the
    # hard way on 2026-08-09; --relocatable emits a /bin/sh wrapper that
    # resolves the interpreter relative to the script instead.
    sudo \$UV venv --relocatable --python \$PYVER .venv-new
    sudo \$UV pip install -q --python .venv-new/bin/python -r requirements.txt
    test -x .venv-new/bin/uvicorn
    sudo .venv-new/bin/python -c 'import query_rag, graph, session_memory; print(\"import smoke OK\")'
    sudo rm -rf .venv-old
    sudo mv .venv .venv-old && sudo mv .venv-new .venv
    echo 'venv swapped; previous one kept at .venv-old'
  "
fi

echo "==> installing and restarting"
# /opt/bcit-rag is root-owned, hence the staged copy through /tmp.
gcloud compute ssh "$VM" --zone="$ZONE" --project="$PROJECT" --command="
  set -e
  for f in ${FILES[*]}; do sudo cp /tmp/\$f $REMOTE/\$f; done
  sudo systemctl restart bcit-chatbot
  # startup loads the 100k-chunk pickle, fits BM25 and builds the entity index
  for i in \$(seq 1 30); do
    sleep 3
    if curl -sf localhost:8000/health | grep -q '\"chatbot_loaded\":true'; then break; fi
  done
  curl -s localhost:8000/health; echo
  journalctl -u bcit-chatbot -n 60 --no-pager \
    | grep -E 'Controller graph|Entity index|Reranker loaded|Loaded .* documents|Response cache' || true
"

echo
echo "==> production check (this question is answered wrongly by the pre-August config)"
curl -s -m 90 -X POST https://bcitai.ca/chat -H 'Content-Type: application/json' \
  -d '{"message":"In COMP 1510, what percentage of the final grade is the final exam worth?"}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["reply"][:200])'
echo
echo "Expect 40%. If not, the old code is still serving."

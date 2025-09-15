#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Config via env (with defaults)
# -----------------------------
IMAGE_BASE="${IMAGE_BASE:-mloc_worker}"          # your local image repo/name (you have mloc:cpu-latest, mloc:cuda-latest)
TAG="${TAG:-latest}"                      # suffix after variant, e.g. cpu-latest / cuda-latest
DEVICE="${DEVICE:-auto}"                  # auto|cpu|cuda (or pass --device)
NAME="${NAME:-mloc}"                      # container name
RUNTIME_ARGS="${RUNTIME_ARGS:-}"          # extra docker run args (e.g. "-p 8000:8000 -e LOG_LEVEL=DEBUG")
DETECT_CMD="${DETECT_CMD:-nvidia-smi}"    # GPU detection command on host
PULL_POLICY="${PULL_POLICY:-if-not-present}" # if-not-present|always|never

# -----------------------------
# Helpers
# -----------------------------
usage() {
  cat <<EOF
Usage: $(basename "$0") [--device auto|cpu|cuda] [--tag TAG] [--name NAME] [--pull-policy if-not-present|always|never] [--] [extra docker run args]

Env overrides:
  IMAGE_BASE    (default: ${IMAGE_BASE})
  TAG           (default: ${TAG})
  DEVICE        (default: ${DEVICE})
  NAME          (default: ${NAME})
  RUNTIME_ARGS  (default: "${RUNTIME_ARGS}")
  PULL_POLICY   (default: ${PULL_POLICY})

Examples:
  IMAGE_BASE=mloc $(basename "$0")                 # auto-detect GPU, prefer local image
  DEVICE=cpu $(basename "$0")                      # force CPU variant
  PULL_POLICY=never $(basename "$0")               # never pull; use local only
  RUNTIME_ARGS="-p 8000:8000" $(basename "$0")     # pass extra docker args
EOF
}

log() { printf '[mloc] %s\n' "$*"; }
has_gpu() { command -v "${DETECT_CMD}" >/dev/null 2>&1 && "${DETECT_CMD}" -L >/dev/null 2>&1; }
image_exists() { docker image inspect "$1" >/dev/null 2>&1; }

# -----------------------------
# Parse minimal CLI flags
# -----------------------------
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) DEVICE="${2:-}"; shift 2 ;;
    --tag) TAG="${2:-}"; shift 2 ;;
    --name) NAME="${2:-}"; shift 2 ;;
    --pull-policy) PULL_POLICY="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done
[[ ${#EXTRA_ARGS[@]} -gt 0 ]] && RUNTIME_ARGS="${RUNTIME_ARGS} ${EXTRA_ARGS[*]}"

# -----------------------------
# Decide variant
# -----------------------------
case "${DEVICE}" in
  cpu)  variant="cpu" ;;
  cuda|gpu) variant="cuda" ;;
  auto)
    if has_gpu; then log "GPU detected; selecting CUDA image."; variant="cuda";
    else log "No GPU detected; selecting CPU image."; variant="cpu"; fi
    ;;
  *) log "Unknown DEVICE=${DEVICE}; use auto|cpu|cuda"; exit 1 ;;
esac

IMAGE="${IMAGE_BASE}:${variant}-${TAG}"

# -----------------------------
# Pull policy
# -----------------------------
case "${PULL_POLICY}" in
  never)
    log "Pull policy: never. Will not pull images."
    ;;
  if-not-present)
    if image_exists "${IMAGE}"; then
      log "Image found locally -> skipping pull."
    else
      log "Pulling image: ${IMAGE}"
      docker pull "${IMAGE}"
    fi
    ;;
  always)
    log "Pull policy: always. Pulling image: ${IMAGE}"
    docker pull "${IMAGE}"
    ;;
  *)
    log "Unknown PULL_POLICY=${PULL_POLICY}; use if-not-present|always|never"; exit 1 ;;
esac

# -----------------------------
# Run container
# -----------------------------
GPU_FLAG=""
if [[ "${variant}" = "cuda" ]]; then
  GPU_FLAG="--gpus all"   # requires NVIDIA Container Toolkit on host
fi

log "Running container: ${NAME}"
set -x
docker run --rm --name "${NAME}" ${GPU_FLAG} ${RUNTIME_ARGS} "${IMAGE}"

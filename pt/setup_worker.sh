#!/usr/bin/env bash
set -euxo pipefail

# ==== 基础设置 ====
USER="${USER:-$(whoami 2>/dev/null || echo root)}"  # 容器环境中可能没有 USER 变量
WORKER_COST_PER_HOUR="${WORKER_COST_PER_HOUR:-0.7}"     # 每小时成本（调度评分用）

# ==== SSH 连接配置（用于远程 worker）====
HOST_IP="${HOST_IP:-193.69.10.112}"                     # 远端 Host IP
SSH_PORT="${SSH_PORT:-50435}"                          # SSH 端口
SSH_USER="${SSH_USER:-root}"                        # SSH 用户名
SSH_KEY_PATH="${SSH_KEY_PATH:-$HOME/.ssh/kv_host}"  # SSH 私钥路径

# ==== 端口映射配置 ====
# 远端服务端口（Host 机器上监听的端口）
REMOTE_REDIS_PORT="${REMOTE_REDIS_PORT:-6379}"
REMOTE_ORCHESTRATOR_PORT="${REMOTE_ORCHESTRATOR_PORT:-8000}"

# 本地映射端口（建议与远端保持一致，除非有端口冲突）
LOCAL_REDIS_PORT="${LOCAL_REDIS_PORT:-6379}"
LOCAL_ORCHESTRATOR_PORT="${LOCAL_ORCHESTRATOR_PORT:-8000}"

# ==== 代码与环境配置 ====
REPO_URL="${REPO_URL:-https://github.com/mlsys-io/kv.run.git}"
REPO_PARENT="${REPO_PARENT:-/opt}"
REPO_DIR="${REPO_DIR:-$REPO_PARENT/kv.run}"
WORKER_ENV_FILE="${WORKER_ENV_FILE:-/opt/kv.worker.env}"

# ==== 目录配置 ====
RESULTS_DIR="${RESULTS_DIR:-/var/lib/kv-worker/results}"     # 结果目录
LOG_DIR="${LOG_DIR:-/var/log/kv-worker}"                      # 日志目录
LOG_FILE="${LOG_FILE:-${LOG_DIR}/worker.log}"                 # 主日志文件
AUTOSSH_LOGFILE="${AUTOSSH_LOGFILE:-${LOG_DIR}/autossh.log}"  # SSH 隧道日志
SSH_TUNNEL_SESSION_NAME="${SSH_TUNNEL_SESSION_NAME:-tunnel}"  # tmux 会话名称
AUTOSSH_GATETIME="${AUTOSSH_GATETIME:-0}"                     # autossh 启动前等待时间
SSH_SERVER_ALIVE_INTERVAL_SEC="${SSH_SERVER_ALIVE_INTERVAL_SEC:-30}"  # SSH 心跳间隔
SSH_SERVER_ALIVE_COUNT_MAX="${SSH_SERVER_ALIVE_COUNT_MAX:-3}"         # SSH 心跳失败次数

# ==== Worker 基本配置 ====
TASK_TOPIC="${TASK_TOPIC:-tasks}"                        # 订阅的任务主题
HEARTBEAT_INTERVAL_SEC="${HEARTBEAT_INTERVAL_SEC:-30}"   # 心跳间隔（秒）
WORKER_ID="${WORKER_ID:-$(hostname)-$RANDOM}"     # Worker ID（确保唯一）
WORKER_TAGS="${WORKER_TAGS:-}"                           # Worker 标签（调度器提示）
LOG_LEVEL="${LOG_LEVEL:-INFO}"                           # 日志级别
WORKER_LOG_TO_STDOUT="${WORKER_LOG_TO_STDOUT:-true}"     # 是否输出到控制台

# ==== Worker 性能配置 ====
WORKER_NETWORK_BANDWIDTH_BYTES_PER_SEC="${WORKER_NETWORK_BANDWIDTH_BYTES_PER_SEC:-}"  # 带宽限制
WORKER_EXECUTOR_MASK="${WORKER_EXECUTOR_MASK:-}"         # 执行器限制（逗号分隔）

# ==== 模型归档压缩配置 ====
MODEL_ARCHIVE_USE_PIGZ="${MODEL_ARCHIVE_USE_PIGZ:-1}"                  # 是否启用 pigz 多线程压缩
MODEL_ARCHIVE_COMPRESSION_LEVEL="${MODEL_ARCHIVE_COMPRESSION_LEVEL:-6}" # GZIP 压缩等级（0-9，pigz/内置均可用）
MODEL_ARCHIVE_PIGZ_THREADS="${MODEL_ARCHIVE_PIGZ_THREADS:-}"           # pigz 线程数（为空则默认使用全部 CPU）
MODEL_ARCHIVE_PIGZ_BIN="${MODEL_ARCHIVE_PIGZ_BIN:-pigz}"               # pigz 可执行文件路径
MODEL_ARCHIVE_TAR_BIN="${MODEL_ARCHIVE_TAR_BIN:-tar}"                  # tar 可执行文件路径
MODEL_CLEANUP_AFTER_UPLOAD="${MODEL_CLEANUP_AFTER_UPLOAD:-0}"          # 上传后清理模型产物（1/0，默认关闭）

# ==== API 访问配置 ====
HF_TOKEN="${HF_TOKEN:-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}"                                 # HuggingFace API 令牌
REDIS_URL="${REDIS_URL:-redis://127.0.0.1:${LOCAL_REDIS_PORT}/0}"     # Redis 连接 URL
ORCHESTRATOR_BASE_URL="${ORCHESTRATOR_BASE_URL:-http://127.0.0.1:${LOCAL_ORCHESTRATOR_PORT}}"  # Orchestrator API URL
RESULTS_UPLOAD_URL="${RESULTS_UPLOAD_URL:-}"            # 结果上传 URL（可选）

# ==== 创建必要的目录 ====
mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

# ================= 基础依赖 =================
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends \
  git curl ca-certificates tmux autossh netcat-openbsd procps lsof pigz
rm -rf /var/lib/apt/lists/*

# 安装 uv（若已存在则跳过）
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
else
  export PATH="$HOME/.local/bin:$PATH"
fi

# ================= 写入 SSH 私钥（如已存在则不覆盖） =================
mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh"

if [ ! -s "$SSH_KEY_PATH" ]; then
  cat >"$SSH_KEY_PATH" <<'EOF_KEY'
-----BEGIN OPENSSH PRIVATE KEY-----
b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAtzc2gtZWQyNTUxOQAAACDm7iK8dGO2SSWCR600UPE+uRn3yaUsZRMDiqU8CpHJUAAAAJAxG/w6MRv8OgAAAAtzc2gtZWQyNTUxOQAAACDm7iK8dGO2SSWCR600UPE+uRn3yaUsZRMDiqU8CpHJUAAAAECCkQ0g3QksRBhazTOkqzERcyycrgRQZOxmotICH3A6NebuIrx0Y7ZJJYJHrTRQ8T65GffJpSxlEwOKpTwKkclQAAAADGoxc2hlbkBsb2NhbAE=
-----END OPENSSH PRIVATE KEY-----
EOF_KEY
fi
chmod 600 "$SSH_KEY_PATH"

# ================= 预置 known_hosts，避免交互确认 =================
ssh-keyscan -p "$SSH_PORT" -t ed25519 "$HOST_IP" >> "$HOME/.ssh/known_hosts" 2>/dev/null || true
chmod 600 "$HOME/.ssh/known_hosts"

# 预热一次（失败即退出，避免隧道静默挂）
ssh -o StrictHostKeyChecking=yes -o UserKnownHostsFile="$HOME/.ssh/known_hosts" \
    -i "$SSH_KEY_PATH" -p "$SSH_PORT" "$SSH_USER@$HOST_IP" true

# ================= 建立 SSH 隧道（tmux 常驻） =================
tunnel_ready=0
if tmux has-session -t "$SSH_TUNNEL_SESSION_NAME" 2>/dev/null; then
  if ss -ltn | grep -q ":${LOCAL_REDIS_PORT} " && ss -ltn | grep -q ":${LOCAL_ORCHESTRATOR_PORT} "; then
    echo "SSH tunnel session ${SSH_TUNNEL_SESSION_NAME} already active; skipping recreation"
    tunnel_ready=1
  else
    echo "SSH tunnel session ${SSH_TUNNEL_SESSION_NAME} exists but ports are not listening; restarting"
    tmux kill-session -t "$SSH_TUNNEL_SESSION_NAME" 2>/dev/null || true
  fi
fi

if [ "$tunnel_ready" -eq 0 ]; then
  tmux kill-session -t "$SSH_TUNNEL_SESSION_NAME" 2>/dev/null || true
  mkdir -p "$(dirname "$AUTOSSH_LOGFILE")"
  touch "$AUTOSSH_LOGFILE"

  tmux new -d -s "$SSH_TUNNEL_SESSION_NAME" "
    AUTOSSH_LOGFILE='$AUTOSSH_LOGFILE' AUTOSSH_GATETIME=${AUTOSSH_GATETIME} \
    autossh -M 0 -NT -vv \
      -o ExitOnForwardFailure=yes \
      -o ServerAliveInterval=${SSH_SERVER_ALIVE_INTERVAL_SEC} -o ServerAliveCountMax=${SSH_SERVER_ALIVE_COUNT_MAX} \
      -o StrictHostKeyChecking=yes \
      -o UserKnownHostsFile='$HOME/.ssh/known_hosts' \
      -i '$SSH_KEY_PATH' -p '$SSH_PORT' '$SSH_USER@$HOST_IP' \
      -L ${LOCAL_REDIS_PORT}:localhost:${REMOTE_REDIS_PORT} \
      -L ${LOCAL_ORCHESTRATOR_PORT}:localhost:${REMOTE_ORCHESTRATOR_PORT}
  "
fi

# 等待本地端口监听
for i in {1..20}; do
  if ss -ltn | grep -q ":${LOCAL_REDIS_PORT} " && ss -ltn | grep -q ":${LOCAL_ORCHESTRATOR_PORT} "; then
    echo "SSH tunnels ready on 127.0.0.1:${LOCAL_REDIS_PORT}/${LOCAL_ORCHESTRATOR_PORT}"
    break
  fi
  sleep 0.5
done

# 健康检查
nc -zv 127.0.0.1 "${LOCAL_REDIS_PORT}"
curl -sI "http://127.0.0.1:${LOCAL_ORCHESTRATOR_PORT}" | head -n1 || true

# ================= 拉取代码 & 安装依赖（uv） =================
mkdir -p "$REPO_PARENT"
chown -R "$USER":"$USER" "$REPO_PARENT"

if [ -d "$REPO_DIR" ]; then
  echo "[Repo] Existing directory $REPO_DIR detected; removing before fresh clone"
  rm -rf "$REPO_DIR"
fi
git clone "$REPO_URL" "$REPO_DIR"
cd "$REPO_DIR"

if [ -f pyproject.toml ]; then
  uv sync
elif [ -f requirements.txt ]; then
  uv pip install -r requirements.txt
fi

# ================= 预下载模型权重到默认 HuggingFace 缓存 =================
# 说明：
# - 使用 HF_TOKEN 进行无交互登录
# - 使用 huggingface-cli download 预热模型到本地缓存 (~/.cache/huggingface)
# - 模型列表：Llama-3.2-1B-Instruct / Llama-3.2-3B-Instruct
LLAMA_1B_REPO="meta-llama/Llama-3.2-1B-Instruct"
LLAMA_3B_REPO="meta-llama/Llama-3.2-3B-Instruct"

# 确保 huggingface-cli 存在（huggingface_hub）
if ! command -v huggingface-cli >/dev/null 2>&1; then
  uv pip install --quiet "huggingface_hub>=0.24.0"
fi

# 登录 Hugging Face（写 token，避免交互式提示）
huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential --stdout >/dev/null 2>&1 || true

echo "[Model Sync] 预下载 ${LLAMA_1B_REPO} 到本地缓存 (~/.cache/huggingface)"
huggingface-cli download \
  "${LLAMA_1B_REPO}" \
  --local-dir "$(mktemp -d)" \
  --local-dir-use-symlinks False \
  --exclude ".git/*" \
  --revision main \
  >/dev/null 2>&1 || true
# 说明：我们传一个临时目录只是强制把所有快照对象拉到缓存，实际推理时框架会直接从缓存复用。

echo "[Model Sync] 预下载 ${LLAMA_3B_REPO} 到本地缓存 (~/.cache/huggingface)"
huggingface-cli download \
  "${LLAMA_3B_REPO}" \
  --local-dir "$(mktemp -d)" \
  --local-dir-use-symlinks False \
  --exclude ".git/*" \
  --revision main \
  >/dev/null 2>&1 || true

# ================= 结果目录 & 环境变量 =================
mkdir -p "$RESULTS_DIR" "$LOG_DIR"
chown -R "$USER":"$USER" "$RESULTS_DIR" "$LOG_DIR"

cat >"$WORKER_ENV_FILE" <<ENV
export REDIS_URL="${REDIS_URL}"
export TASK_TOPIC="${TASK_TOPIC}"
export RESULTS_DIR="${RESULTS_DIR}"
export ORCHESTRATOR_BASE_URL="${ORCHESTRATOR_BASE_URL}"
export HEARTBEAT_INTERVAL_SEC="${HEARTBEAT_INTERVAL_SEC}"
export WORKER_ID="${WORKER_ID}"
export WORKER_TAGS="${WORKER_TAGS}"
export LOG_LEVEL="${LOG_LEVEL}"
export WORKER_LOG_TO_STDOUT="${WORKER_LOG_TO_STDOUT}"
export WORKER_COST_PER_HOUR="${WORKER_COST_PER_HOUR}"
export WORKER_NETWORK_BANDWIDTH_BYTES_PER_SEC="${WORKER_NETWORK_BANDWIDTH_BYTES_PER_SEC}"
export WORKER_EXECUTOR_MASK="${WORKER_EXECUTOR_MASK}"
export RESULTS_UPLOAD_URL="${RESULTS_UPLOAD_URL}"
export HF_TOKEN="${HF_TOKEN}"
export MODEL_ARCHIVE_USE_PIGZ="${MODEL_ARCHIVE_USE_PIGZ}"
export MODEL_ARCHIVE_COMPRESSION_LEVEL="${MODEL_ARCHIVE_COMPRESSION_LEVEL}"
export MODEL_ARCHIVE_PIGZ_THREADS="${MODEL_ARCHIVE_PIGZ_THREADS}"
export MODEL_ARCHIVE_PIGZ_BIN="${MODEL_ARCHIVE_PIGZ_BIN}"
export MODEL_ARCHIVE_TAR_BIN="${MODEL_ARCHIVE_TAR_BIN}"
export MODEL_CLEANUP_AFTER_UPLOAD="${MODEL_CLEANUP_AFTER_UPLOAD}"
export LOG_DIR="${LOG_DIR}"
export LOG_FILE="${LOG_FILE}"
export AUTOSSH_LOGFILE="${AUTOSSH_LOGFILE}"
export SSH_TUNNEL_SESSION_NAME="${SSH_TUNNEL_SESSION_NAME}"
export AUTOSSH_GATETIME="${AUTOSSH_GATETIME}"
export SSH_SERVER_ALIVE_INTERVAL_SEC="${SSH_SERVER_ALIVE_INTERVAL_SEC}"
export SSH_SERVER_ALIVE_COUNT_MAX="${SSH_SERVER_ALIVE_COUNT_MAX}"
ENV

# 清理可能存在的后台 worker
set +e
pkill -f "uv run -m worker.main" 2>/dev/null || true
pkill -f "python -m worker.main" 2>/dev/null || true
set -e

echo "---------------------------------------------"
echo "[Worker] 前台运行（worker.main）"
echo "WORKER_ID            = ${WORKER_ID}"
echo "REDIS_URL            = ${REDIS_URL}"
echo "ORCHESTRATOR_BASE_URL= ${ORCHESTRATOR_BASE_URL}"
echo "隧道 tmux 会话       = ${SSH_TUNNEL_SESSION_NAME}   (查看: tmux a -t ${SSH_TUNNEL_SESSION_NAME})"
echo "autossh 日志         = ${AUTOSSH_LOGFILE}"
echo "---------------------------------------------"
echo

# ================= 前台启动 worker/main.py =================
source "$WORKER_ENV_FILE"
cd "$REPO_DIR"
git pull

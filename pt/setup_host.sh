#!/usr/bin/env bash
set -euxo pipefail

# 将所有可调节的环境变量集中在顶部，便于修改或通过外部覆盖
USER="${USER:-$(whoami 2>/dev/null || echo root)}"

# ==== Redis 配置 ====
REDIS_HOST="${REDIS_HOST:-195.139.22.91}"
REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_URL="${REDIS_URL:-redis://${REDIS_HOST}:${REDIS_PORT}/0}"

# ==== HTTP 服务配置 ====
ORCHESTRATOR_BIND_HOST="${ORCHESTRATOR_BIND_HOST:-0.0.0.0}"
ORCHESTRATOR_HTTP_PORT="${ORCHESTRATOR_HTTP_PORT:-8000}"
ORCHESTRATOR_TOKEN="${ORCHESTRATOR_TOKEN:-}"       # API 认证令牌，留空表示不启用认证
HOST_APP_RELOAD="${HOST_APP_RELOAD:-0}"            # 是否启用自动重载
HOST_APP_LOG_LEVEL="${HOST_APP_LOG_LEVEL:-info}"

# ==== 调度器配置 ====
ORCHESTRATOR_DISPATCH_MODE="${ORCHESTRATOR_DISPATCH_MODE:-adaptive}"     # 调度模式：adaptive, fixed_pipeline, static_round_robin, static_worker
ORCHESTRATOR_WORKER_SELECTION="${ORCHESTRATOR_WORKER_SELECTION:-best_fit}"  # worker 选择策略：best_fit, first_fit, min_satisfying
ENABLE_CONTEXT_REUSE="${ENABLE_CONTEXT_REUSE:-true}"                     # 启用上下文复用（模型/数据集缓存）
ENABLE_TASK_MERGE="${ENABLE_TASK_MERGE:-true}"                           # 启用任务合并（相同 DAG 节点）
TASK_MERGE_MAX_BATCH_SIZE="${TASK_MERGE_MAX_BATCH_SIZE:-4}"              # 最大批处理大小
WORKER_CACHE_TTL_SEC="${WORKER_CACHE_TTL_SEC:-3600}"                     # 缓存元数据过期时间
ENABLE_STAGE_WEIGHT_STICKINESS="${ENABLE_STAGE_WEIGHT_STICKINESS:-false}"  # 阶段权重粘性（优先使用已有权重的 worker）

# ==== Worker 监控配置 ====
ENABLE_WORKER_WATCHDOG="${ENABLE_WORKER_WATCHDOG:-true}"                 # 是否启用 worker 守护
WORKER_DEATH_CHECK_INTERVAL="${WORKER_DEATH_CHECK_INTERVAL:-30}"         # worker 存活检测间隔
WORKER_DEATH_GRACE_SEC="${WORKER_DEATH_GRACE_SEC:-60}"                   # worker 判定死亡的宽限秒数

# ==== 指标配置 ====
HOST_METRICS_ENABLE_DENSITY_PLOT="${HOST_METRICS_ENABLE_DENSITY_PLOT:-true}"  # 启用密度图
HOST_METRICS_DENSITY_BUCKET_SEC="${HOST_METRICS_DENSITY_BUCKET_SEC:-60}"       # 密度图桶宽（秒）

# ==== 弹性调度参数 ====
ENABLE_ELASTIC_SCALING="${ENABLE_ELASTIC_SCALING:-True}"                 # 启用弹性伸缩
ELASTIC_AUTO_DISABLE_IDLE_SEC="${ELASTIC_AUTO_DISABLE_IDLE_SEC:-30}"      # 空闲多久后禁用 worker
ELASTIC_AUTO_ENABLE_QUEUE_THRESHOLD="${ELASTIC_AUTO_ENABLE_QUEUE_THRESHOLD:-1}"  # 队列长度触发启用的阈值
ELASTIC_AUTO_DISABLE_QUEUE_MAX="${ELASTIC_AUTO_DISABLE_QUEUE_MAX:-10}"     # 禁用时允许的最大队列长度
ELASTIC_AUTO_POLL_INTERVAL_SEC="${ELASTIC_AUTO_POLL_INTERVAL_SEC:-5}"    # 弹性调度检查间隔
ELASTIC_AUTO_TOGGLE_COOLDOWN_SEC="${ELASTIC_AUTO_TOGGLE_COOLDOWN_SEC:-30}"  # 启用/禁用操作的冷却时间
ELASTIC_AUTO_MIN_ACTIVE_WORKERS="${ELASTIC_AUTO_MIN_ACTIVE_WORKERS:-1}"   # 最小活动 worker 数量
ELASTIC_AUTO_ENABLE_MULTI_WORKERS="${ELASTIC_AUTO_ENABLE_MULTI_WORKERS:-false}"  # 自动启用是否允许批量

# ==== 固定流水线调度 ====
FIXED_PIPELINE_BINDING_IDLE_TIMEOUT_SEC="${FIXED_PIPELINE_BINDING_IDLE_TIMEOUT_SEC:-300}"  # 绑定空闲释放阈值（秒）

# ==== 调度器得分权重 ====
SCHEDULER_LAMBDA_INFERENCE="${SCHEDULER_LAMBDA_INFERENCE:-0.4}"    # 推理任务的吞吐量权重
SCHEDULER_LAMBDA_TRAINING="${SCHEDULER_LAMBDA_TRAINING:-0.8}"      # 训练任务的吞吐量权重
SCHEDULER_LAMBDA_OTHER="${SCHEDULER_LAMBDA_OTHER:-0.5}"            # 其他任务的吞吐量权重
SCHEDULER_SELECTION_JITTER="${SCHEDULER_SELECTION_JITTER:-0.001}"  # 选择打分的抖动系数

# ==== 文件路径配置 ====
REPO_URL="${REPO_URL:-https://github.com/mlsys-io/kv.run.git}"
REPO_PARENT="${REPO_PARENT:-/opt}"
REPO_DIR="${REPO_DIR:-${REPO_PARENT}/kv.run}"
HOST_ENV_FILE="${HOST_ENV_FILE:-/opt/kv.host.env}"

# ==== 目录配置 ====
RESULTS_DIR="${RESULTS_DIR:-/var/lib/kv-host/results}"              # 结果目录
METRICS_DIR="${METRICS_DIR:-/var/lib/kv-host/metrics}"              # 指标输出目录
STATE_DIR="${STATE_DIR:-/var/lib/kv-host/state}"                    # 状态目录
ORCHESTRATOR_STATE_DIR="${ORCHESTRATOR_STATE_DIR:-$STATE_DIR}"      # 状态目录（可单独覆盖）
STATE_DIR="$ORCHESTRATOR_STATE_DIR"                                 # 统一使用 ORCHESTRATOR_STATE_DIR
LOG_DIR="${LOG_DIR:-/var/log/kv-host}"                              # 日志目录
LOG_FILE="${LOG_FILE:-${LOG_DIR}/host.log}"                         # 主日志文件
UV_RUN_LOG_FILE="${UV_RUN_LOG_FILE:-${LOG_DIR}/uv-host.log}"        # UV 运行日志

# ==== 日志配置 ====
LOG_LEVEL="${LOG_LEVEL:-INFO}"                 # 日志级别：DEBUG, INFO, WARNING, ERROR
LOG_MAX_BYTES="${LOG_MAX_BYTES:-5242880}"      # 单个日志文件最大字节数 (5MB)
LOG_BACKUP_COUNT="${LOG_BACKUP_COUNT:-5}"      # 保留的日志文件数量

# ==== 状态管理配置 ====
ORCHESTRATOR_STATE_ENABLED="${ORCHESTRATOR_STATE_ENABLED:-0}"   # 是否启用状态管理
STATE_FLUSH_INTERVAL_SEC="${STATE_FLUSH_INTERVAL_SEC:-5}"       # 状态刷新间隔（秒）

# ==== 基础依赖（无 systemd 环境）====
apt-get update -y
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  git curl ca-certificates tmux procps net-tools lsof redis-server
rm -rf /var/lib/apt/lists/*

# ==== 安装 uv ====
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
fi

# ==== 拉代码 ====
mkdir -p "$REPO_PARENT"
chown -R "$USER":"$USER" "$REPO_PARENT"
if [ ! -d "$REPO_DIR" ]; then
  git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"

# ==== 安装 orchestrator 依赖 ====
echo "Installing orchestrator dependencies via uv"
if [ -f pyproject.toml ]; then
  uv sync --extra inference
elif [ -f requirements.txt ]; then
  uv pip install -r requirements.txt
fi

# ==== 目录 ====
mkdir -p "$RESULTS_DIR" "$METRICS_DIR" "$STATE_DIR" "$LOG_DIR"
chown -R "$USER":"$USER" "$RESULTS_DIR" "$METRICS_DIR" "$STATE_DIR" "$LOG_DIR"
touch "$LOG_FILE" "$UV_RUN_LOG_FILE"

# ==== 生成最小 redis.conf（本地环回，无密码，配合 SSH 隧道）====
REDIS_CONF="/etc/redis/redis-no-systemd.conf"
cat > "$REDIS_CONF" <<CONF
bind 127.0.0.1
port ${REDIS_PORT}
protected-mode yes
daemonize yes
supervised no
save 900 1
save 300 10
save 60 10000
dir /var/lib/redis
logfile /var/log/redis-server.log
# appendonly/no：默认关闭，容器环境避免额外 IO；如需持久化可改为 yes
appendonly no
CONF

mkdir -p /var/lib/redis
chown -R redis:redis /var/lib/redis
mkdir -p /var/log
touch /var/log/redis-server.log
chown redis:redis /var/log/redis-server.log

# ==== 启动 Redis（不使用 systemctl）====
# 若已有旧进程，先杀掉
if pgrep -x redis-server >/dev/null; then
  pkill -9 redis-server || true
fi

# 前台转后台由 redis 自己完成（daemonize yes）
redis-server "$REDIS_CONF"

# 健康检查
for i in {1..20}; do
  if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" PING | grep -q PONG; then
    echo "Redis is up."
    break
  fi
  sleep 0.5
done

# ==== 写环境变量 ====
cat >"$HOST_ENV_FILE" <<ENV
export REDIS_URL="${REDIS_URL}"
export RESULTS_DIR="${RESULTS_DIR}"
export HOST_METRICS_DIR="${METRICS_DIR}"
export ORCHESTRATOR_METRICS_DIR="${METRICS_DIR}"
export ORCHESTRATOR_STATE_ENABLED="${ORCHESTRATOR_STATE_ENABLED}"
export ORCHESTRATOR_STATE_DIR="${ORCHESTRATOR_STATE_DIR}"
export STATE_FLUSH_INTERVAL_SEC="${STATE_FLUSH_INTERVAL_SEC}"

export LOG_FILE="${LOG_FILE}"
export LOG_DIR="${LOG_DIR}"
export LOG_LEVEL="${LOG_LEVEL}"
export LOG_MAX_BYTES="${LOG_MAX_BYTES}"
export LOG_BACKUP_COUNT="${LOG_BACKUP_COUNT}"
export UV_RUN_LOG_FILE="${UV_RUN_LOG_FILE}"

export ORCHESTRATOR_TOKEN="${ORCHESTRATOR_TOKEN}"
export ORCHESTRATOR_DISPATCH_MODE="${ORCHESTRATOR_DISPATCH_MODE}"
export ORCHESTRATOR_WORKER_SELECTION="${ORCHESTRATOR_WORKER_SELECTION}"
export ENABLE_CONTEXT_REUSE="${ENABLE_CONTEXT_REUSE}"
export ENABLE_TASK_MERGE="${ENABLE_TASK_MERGE}"
export TASK_MERGE_MAX_BATCH_SIZE="${TASK_MERGE_MAX_BATCH_SIZE}"
export ENABLE_ELASTIC_SCALING="${ENABLE_ELASTIC_SCALING}"
export WORKER_CACHE_TTL_SEC="${WORKER_CACHE_TTL_SEC}"
export ENABLE_STAGE_WEIGHT_STICKINESS="${ENABLE_STAGE_WEIGHT_STICKINESS}"
export SCHEDULER_LAMBDA_INFERENCE="${SCHEDULER_LAMBDA_INFERENCE}"
export SCHEDULER_LAMBDA_TRAINING="${SCHEDULER_LAMBDA_TRAINING}"
export SCHEDULER_LAMBDA_OTHER="${SCHEDULER_LAMBDA_OTHER}"
export SCHEDULER_SELECTION_JITTER="${SCHEDULER_SELECTION_JITTER}"
export ELASTIC_AUTO_DISABLE_IDLE_SEC="${ELASTIC_AUTO_DISABLE_IDLE_SEC}"
export ELASTIC_AUTO_ENABLE_QUEUE_THRESHOLD="${ELASTIC_AUTO_ENABLE_QUEUE_THRESHOLD}"
export ELASTIC_AUTO_DISABLE_QUEUE_MAX="${ELASTIC_AUTO_DISABLE_QUEUE_MAX}"
export ELASTIC_AUTO_POLL_INTERVAL_SEC="${ELASTIC_AUTO_POLL_INTERVAL_SEC}"
export ELASTIC_AUTO_TOGGLE_COOLDOWN_SEC="${ELASTIC_AUTO_TOGGLE_COOLDOWN_SEC}"
export ELASTIC_AUTO_MIN_ACTIVE_WORKERS="${ELASTIC_AUTO_MIN_ACTIVE_WORKERS}"
export ELASTIC_AUTO_ENABLE_MULTI_WORKERS="${ELASTIC_AUTO_ENABLE_MULTI_WORKERS}"
export FIXED_PIPELINE_BINDING_IDLE_TIMEOUT_SEC="${FIXED_PIPELINE_BINDING_IDLE_TIMEOUT_SEC}"
export ENABLE_WORKER_WATCHDOG="${ENABLE_WORKER_WATCHDOG}"
export WORKER_DEATH_CHECK_INTERVAL="${WORKER_DEATH_CHECK_INTERVAL}"
export WORKER_DEATH_GRACE_SEC="${WORKER_DEATH_GRACE_SEC}"

export HOST_APP_HOST="${ORCHESTRATOR_BIND_HOST}"
export HOST_APP_PORT="${ORCHESTRATOR_HTTP_PORT}"
export PORT="${ORCHESTRATOR_HTTP_PORT}"
export HOST_APP_RELOAD="${HOST_APP_RELOAD}"
export HOST_APP_LOG_LEVEL="${HOST_APP_LOG_LEVEL}"
export HOST_METRICS_ENABLE_DENSITY_PLOT="${HOST_METRICS_ENABLE_DENSITY_PLOT}"
export HOST_METRICS_DENSITY_BUCKET_SEC="${HOST_METRICS_DENSITY_BUCKET_SEC}"
ENV

# ==== 启动 orchestrator（前台运行）====
set +e
pkill -f "uv run -m host.main" 2>/dev/null || true
set -e

echo "Launching FlowMesh Host orchestrator (foreground mode)..."
echo "-----------------------------------------------------------"
echo "Environment file : ${HOST_ENV_FILE}"
echo "Working directory : ${REPO_DIR}"
echo "-----------------------------------------------------------"
echo

# 前台运行（实时显示日志）
source "${HOST_ENV_FILE}"
cd "${REPO_DIR}"
git pull
uv run -m host.main

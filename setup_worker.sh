#!/usr/bin/env bash
set -euxo pipefail

# ==== Host SSH 信息 ====
HOST_IP="71.209.132.34"   # ← 改成 host 的公网 IP
SSH_PORT=41151            # ← 改成 host 的 SSH 端口
SSH_USER="root"           # ← 改成实际用户名（root/ubuntu 等）
SSH_KEY="$HOME/.ssh/kv_host"

# ==== 安装 uv/工具 ====
sudo apt update -y
sudo apt install -y git curl ca-certificates tmux autossh
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# ==== 写入私钥（权限必须 600） ====
mkdir -p "$HOME/.ssh"
cat > "$SSH_KEY" <<'EOF'
-----BEGIN OPENSSH PRIVATE KEY-----
b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAtzc2gtZW
QyNTUxOQAAACDm7iK8dGO2SSWCR600UPE+uRn3yaUsZRMDiqU8CpHJUAAAAJAxG/w6MRv8
OgAAAAtzc2gtZWQyNTUxOQAAACDm7iK8dGO2SSWCR600UPE+uRn3yaUsZRMDiqU8CpHJUA
AAAECCkQ0g3QksRBhazTOkqzERcyycrgRQZOxmotICH3A6NebuIrx0Y7ZJJYJHrTRQ8T65
GffJpSxlEwOKpTwKkclQAAAADGoxc2hlbkBsb2NhbAE=
-----END OPENSSH PRIVATE KEY-----
EOF
chmod 600 "$SSH_KEY"

# ==== 建立 SSH 双隧道（本地直通 host 的 6379/8000）====
tmux kill-session -t tunnel 2>/dev/null || true
tmux new -d -s tunnel "
  autossh -M 0 -N -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
    -i '$SSH_KEY' -p '$SSH_PORT' '$SSH_USER@$HOST_IP' \
    -L 6379:localhost:6379 \
    -L 8000:localhost:8000
"
sleep 2
tmux ls || true
# 简易连通性测试
nc -zv 127.0.0.1 6379
curl -I http://127.0.0.1:8000 || true

# ==== 拉代码并用 uv 安装依赖 ====
sudo mkdir -p /opt
sudo chown -R "$USER":"$USER" /opt
cd /opt
if [ ! -d kv.run ]; then
  git clone https://github.com/mlsys-io/kv.run.git
fi
cd kv.run

if [ -f pyproject.toml ]; then
  uv sync
elif [ -f requirements.txt ]; then
  uv pip install -r requirements.txt
fi

# ==== 启动 worker（走隧道上的本地端口）====
cat >/opt/kv.worker.env <<'ENV'
export REDIS_URL=redis://127.0.0.1:6379/0
export ORCHESTRATOR_BASE_URL=http://127.0.0.1:8000
export WORKER_NAME=worker-$(hostname)
export ORCHESTRATOR_WORKER_SELECTION=best_fit
ENV

set +e
pkill -f "python -m worker.redis_worker" 2>/dev/null || true
set -e
# 用 uv 运行模块
bash -c "source /opt/kv.worker.env && cd /opt/kv.run && nohup uv run -m worker.redis_worker > /var/log/kv.worker.log 2>&1 &"

echo "---------------------------------------------"
echo "[Worker] 隧道会话: tmux session 'tunnel'"
echo "通过 127.0.0.1:6379/8000 访问 host；日志: /var/log/kv.worker.log"

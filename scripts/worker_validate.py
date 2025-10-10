#!/usr/bin/env python3
"""Local worker validation helper.

用法示例：

    python scripts/worker_validate.py run --scenario echo-local \
        --orchestrator-url http://127.0.0.1:8090 --token smoketoken

    python scripts/worker_validate.py run --scenario echo-http \
        --orchestrator-url http://127.0.0.1:8090 --token smoketoken

脚本会提交预置模板、轮询任务状态，并输出结果摘要。"""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict

import requests
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = REPO_ROOT / "templates"


def _load_template(name: str) -> Dict[str, Any]:
    path = TEMPLATE_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"模板 {name} 不存在: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _stamp_template(spec: Dict[str, Any], suffix: str) -> Dict[str, Any]:
    meta = spec.setdefault("metadata", {})
    base_name = meta.get("name", "task")
    meta["name"] = f"{base_name}-{suffix}"
    return spec


def _scenario_echo_local() -> Dict[str, Any]:
    spec = _load_template("echo_local.yaml")
    return _stamp_template(spec, suffix=uuid.uuid4().hex[:8])


def _scenario_echo_http(orchestrator_url: str, token: str | None) -> Dict[str, Any]:
    spec = _scenario_echo_local()
    destination = spec.setdefault("spec", {}).setdefault("output", {}).setdefault("destination", {})
    destination["type"] = "http"
    destination["url"] = orchestrator_url.rstrip("/") + "/api/v1/results"
    headers: Dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    destination["headers"] = headers
    destination.pop("path", None)
    return spec


SCENARIOS = {
    "echo-local": _scenario_echo_local,
    "echo-http": _scenario_echo_http,
}


def submit_task(orchestrator_url: str, token: str | None, payload: str) -> Dict[str, Any]:
    headers = {"Content-Type": "text/yaml"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = requests.post(f"{orchestrator_url.rstrip('/')}/api/v1/tasks", headers=headers, data=payload.encode("utf-8"), timeout=30)
    resp.raise_for_status()
    return resp.json()


def poll_task(orchestrator_url: str, token: str | None, task_id: str, timeout: float = 120.0, interval: float = 2.0) -> Dict[str, Any]:
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    deadline = time.time() + timeout
    last_payload: Dict[str, Any] | None = None
    while time.time() < deadline:
        resp = requests.get(f"{orchestrator_url.rstrip('/')}/api/v1/tasks/{task_id}", headers=headers, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        status = payload.get("status")
        if status in {"DONE", "FAILED"}:
            return payload
        last_payload = payload
        time.sleep(interval)
    raise TimeoutError(f"等待任务 {task_id} 超时，最后状态: {last_payload}")


def fetch_result(orchestrator_url: str, token: str | None, task_id: str) -> Dict[str, Any]:
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = requests.get(f"{orchestrator_url.rstrip('/')}/api/v1/results/{task_id}", headers=headers, timeout=30)
    if resp.status_code == 404:
        raise FileNotFoundError("结果文件不存在，请确认任务是否使用了 HTTP 回传或写入本地。")
    resp.raise_for_status()
    return resp.json()


def run_scenario(args: argparse.Namespace) -> None:
    orchestrator_url = args.orchestrator_url.rstrip("/")
    token = args.token

    builder = SCENARIOS.get(args.scenario)
    if not builder:
        raise SystemExit(f"未知场景 {args.scenario}，可用值: {', '.join(SCENARIOS)}")

    template = builder(orchestrator_url, token) if args.scenario == "echo-http" else builder()
    payload_text = yaml.safe_dump(template, sort_keys=False, allow_unicode=True)

    submission = submit_task(orchestrator_url, token, payload_text)
    task_id = submission["tasks"][0]["task_id"]
    print(f"已提交任务: {task_id}")

    task_payload = poll_task(orchestrator_url, token, task_id, timeout=args.timeout, interval=args.interval)
    status = task_payload.get("status")
    print(f"任务状态: {status}")
    if task_payload.get("error"):
        print(f"错误信息: {task_payload['error']}")

    if args.fetch_result and status == "DONE":
        try:
            result = fetch_result(orchestrator_url, token, task_id)
            print("结果预览:")
            print(yaml.safe_dump(result, sort_keys=False, allow_unicode=True))
        except FileNotFoundError as exc:
            print(f"提示: {exc}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="本地 worker 验证脚本")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="执行预置场景")
    run.add_argument("--scenario", choices=sorted(SCENARIOS.keys()), default="echo-local", help="验证场景")
    run.add_argument("--orchestrator-url", default=str(Path("http://127.0.0.1:8090")), help="Orchestrator 基础 URL")
    run.add_argument("--token", default=None, help="可选的 Bearer Token")
    run.add_argument("--timeout", type=float, default=180.0, help="轮询等待超时时间（秒）")
    run.add_argument("--interval", type=float, default=2.0, help="轮询间隔（秒）")
    run.add_argument("--fetch-result", action="store_true", help="任务完成后获取结果 JSON")
    run.set_defaults(func=run_scenario)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except Exception as exc:  # noqa: broad-except
        print(f"[验证失败] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

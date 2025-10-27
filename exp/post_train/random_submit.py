#!/usr/bin/env python3
import argparse
import os
import sys
import pathlib
import random
import requests
import json

def main():
    parser = argparse.ArgumentParser(
        description="Submit random YAML task(s) to the task API (with replacement)."
    )
    parser.add_argument("count", type=int, help="number of tasks to submit (>0)")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed for reproducible file selection",
    )
    args = parser.parse_args()

    if args.count <= 0:
        print(f"Count must be a positive integer; got '{args.count}'", file=sys.stderr)
        sys.exit(1)

    script_dir = pathlib.Path(__file__).resolve().parent

    host_url = os.environ.get("HOST_URL", "http://localhost:8000")
    task_endpoint = os.environ.get("TASK_ENDPOINT", "/api/v1/tasks")
    token = os.environ.get("TOKEN", "dev-token")

    task_files = sorted(
        p for p in script_dir.iterdir()
        if p.is_file() and p.suffix == ".yaml"
    )
    if not task_files:
        print(f"No YAML task files found in {script_dir}", file=sys.stderr)
        sys.exit(1)

    if args.seed is not None:
        random.seed(args.seed)

    print(f"Submitting {args.count} random task(s) (with replacement) to {host_url}{task_endpoint}")

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "text/yaml",
    }

    for i in range(1, args.count + 1):
        file_path = random.choice(task_files)
        print(f"[{i}/{args.count}] Submitting {file_path.name}")

        with open(file_path, "rb") as f:
            resp = requests.post(
                f"{host_url}{task_endpoint}",
                headers=headers,
                data=f,
            )

        # Try pretty JSON, else raw text
        try:
            parsed = resp.json()
            print(json.dumps(parsed, indent=2, ensure_ascii=False))
        except ValueError:
            print(resp.text)

if __name__ == "__main__":
    main()

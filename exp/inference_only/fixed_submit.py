#!/usr/bin/env python3
import argparse
import os
import sys
import time
import random
import shutil
import subprocess
from pathlib import Path
import glob

def parse_args():
    parser = argparse.ArgumentParser(
        description="Submit <count> random YAML task(s) at a fixed rate (requests/sec)."
    )
    parser.add_argument("count", type=int,
                        help="How many tasks to submit (>0)")
    parser.add_argument("rate_per_sec", type=float,
                        help="Requests per second (>0); e.g. 0.5 = one every 2s")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible YAML selection (default: None)",
    )
    parser.add_argument(
        "--host_url",
        default=os.getenv("HOST_URL", "http://localhost:8000"),
        help="Orchestrator base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--task_endpoint",
        default=os.getenv("TASK_ENDPOINT", "/api/v1/tasks"),
        help="Task submission endpoint (default: /api/v1/tasks)",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("TOKEN", "dev-token"),
        help="Auth token (default: dev-token)",
    )
    parser.add_argument(
        "--task_dir",
        default=str(Path(__file__).resolve().parent),
        help="Directory containing *.yaml to sample from (default: script dir)",
    )
    return parser.parse_args()

def validate_args(args):
    if args.count <= 0:
        print(f"Count must be a positive integer; got '{args.count}'", file=sys.stderr)
        sys.exit(1)
    try:
        rate = float(args.rate_per_sec)
    except ValueError:
        print(f"Rate must be a positive number; got '{args.rate_per_sec}'", file=sys.stderr)
        sys.exit(1)
    if rate <= 0:
        print(f"Rate must be a positive number; got '{args.rate_per_sec}'", file=sys.stderr)
        sys.exit(1)
    return rate

def gather_yaml_files(task_dir):
    files = sorted(glob.glob(os.path.join(task_dir, "*.yaml")))
    if not files:
        print(f"No YAML task files found in {task_dir}", file=sys.stderr)
        sys.exit(1)
    return files

def post_yaml(host_url, endpoint, token, yaml_path, use_jq):
    url = f"{host_url}{endpoint}"
    cmd = [
        "curl", "-sS", "-X", "POST", url,
        "-H", f"Authorization: Bearer {token}",
        "-H", "Content-Type: text/yaml",
        "--data-binary", f"@{yaml_path}",
    ]

    if use_jq:
        # pipe curl -> jq .
        p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p2 = subprocess.Popen(["jq", "."], stdin=p1.stdout,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p1.stdout.close()
        out, err = p2.communicate()
        rc = p2.returncode
        # if jq fails, fall back to raw curl output
        if rc != 0:
            out, err = p1.communicate()
            rc = p1.returncode
        return rc, out.decode(errors="ignore"), err.decode(errors="ignore")
    else:
        p = subprocess.run(cmd, capture_output=True, text=True)
        return p.returncode, p.stdout, p.stderr

def main():
    args = parse_args()
    rate_per_sec = validate_args(args)
    interval = 1.0 / rate_per_sec

    # Seed RNG if provided
    if args.seed is not None:
        random.seed(args.seed)

    yaml_files = gather_yaml_files(args.task_dir)
    total_files = len(yaml_files)

    use_jq = shutil.which("jq") is not None

    print(
        f"Randomly submitting {args.count} task(s) to "
        f"{args.host_url}{args.task_endpoint} at ~{rate_per_sec}/sec"
    )
    if args.seed is not None:
        print(f"Seed={args.seed}")

    for i in range(args.count):
        # choose random yaml (with replacement)
        chosen = random.choice(yaml_files)
        base = os.path.basename(chosen)
        print(f"[{i+1}/{args.count}] Submitting {base}")

        rc, out, err = post_yaml(
            args.host_url,
            args.task_endpoint,
            args.token,
            chosen,
            use_jq,
        )
        # mirror bash behavior: print body
        if out:
            sys.stdout.write(out)
            if not out.endswith("\n"):
                sys.stdout.write("\n")
        if rc != 0 and err:
            # if curl/jq failed, surface stderr
            sys.stderr.write(err)

        # sleep between requests, but not after the last one
        if (i + 1) < args.count:
            time.sleep(interval)

if __name__ == "__main__":
    main()

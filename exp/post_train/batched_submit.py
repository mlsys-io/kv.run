#!/usr/bin/env python3
import argparse
import os
import sys
import random
import subprocess
import shutil
from pathlib import Path
import glob

def parse_args():
    parser = argparse.ArgumentParser(
        description="Submit a unique random subset of YAML tasks (no replacement)."
    )
    parser.add_argument(
        "count",
        type=int,
        help="How many tasks to submit (>0)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible task selection (default: None)"
    )
    parser.add_argument(
        "--host_url",
        default=os.getenv("HOST_URL", "http://localhost:8000"),
        help="Orchestrator base URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--task_endpoint",
        default=os.getenv("TASK_ENDPOINT", "/api/v1/tasks"),
        help="Task submission endpoint (default: /api/v1/tasks)"
    )
    parser.add_argument(
        "--token",
        default=os.getenv("TOKEN", "dev-token"),
        help="Auth token (default: dev-token)"
    )
    parser.add_argument(
        "--task_dir",
        default=str(Path(__file__).resolve().parent),
        help="Directory containing *.yaml (default: script dir)"
    )
    return parser.parse_args()

def get_yaml_files(task_dir):
    # equivalent to: find "$SCRIPT_DIR" -maxdepth 1 -type f -name '*.yaml' | sort
    # => only direct children, no recursion
    p = Path(task_dir)
    files = [str(f) for f in p.iterdir() if f.is_file() and f.suffix == ".yaml"]
    files.sort()
    return files

def select_tasks(files, requested_count, rng):
    total = len(files)
    # mirror bash behavior:
    # if COUNT > TOTAL:
    #   warn, cap to TOTAL, and submit ALL files in sorted order
    if requested_count > total:
        print(
            f"Requested {requested_count} tasks but only {total} available; "
            f"submitting {total} instead.",
            file=sys.stderr
        )
        requested_count = total

    if requested_count >= total:
        # same as SELECTED_TASKS=("${TASK_FILES[@]}")
        return files

    # same semantics as: shuf -e -- "${TASK_FILES[@]}" | head -n "$COUNT"
    # i.e. random permutation, take first N, no repetition
    # random.sample gives us N unique elements in random order
    return rng.sample(files, requested_count)

def post_yaml(host_url, endpoint, token, yaml_path, use_jq):
    url = f"{host_url}{endpoint}"
    cmd = [
        "curl", "-sS", "-X", "POST", url,
        "-H", f"Authorization: Bearer {token}",
        "-H", "Content-Type: text/yaml",
        "--data-binary", f"@{yaml_path}",
    ]

    if use_jq:
        # curl | jq '.'
        p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p2 = subprocess.Popen(["jq", "."],
                              stdin=p1.stdout,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
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

    # validate count
    if args.count <= 0:
        print(f"Count must be a positive integer; got '{args.count}'", file=sys.stderr)
        sys.exit(1)

    # RNG with optional seed
    rng = random.Random()
    if args.seed is not None:
        rng.seed(args.seed)

    # gather yaml files
    task_files = get_yaml_files(args.task_dir)
    total = len(task_files)

    if total == 0:
        print(f"No YAML task files found in {args.task_dir}", file=sys.stderr)
        sys.exit(1)

    # choose which tasks to send (no replacement semantics)
    selected_tasks = select_tasks(task_files, args.count, rng)
    select_count = len(selected_tasks)

    print(f"Submitting {select_count} task(s) to {args.host_url}{args.task_endpoint}")
    if args.seed is not None:
        print(f"Seed={args.seed}")

    use_jq = shutil.which("jq") is not None

    # loop and send
    for i, file_path in enumerate(selected_tasks, start=1):
        basename = os.path.basename(file_path)
        print(f"[{i}/{select_count}] Submitting {basename}")

        rc, out, err = post_yaml(
            args.host_url,
            args.task_endpoint,
            args.token,
            file_path,
            use_jq,
        )

        # mimic bash: print response body
        if out:
            sys.stdout.write(out)
            if not out.endswith("\n"):
                sys.stdout.write("\n")

        if rc != 0 and err:
            # surface stderr on failure
            sys.stderr.write(err)

if __name__ == "__main__":
    main()

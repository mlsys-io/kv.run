#!/usr/bin/env python3
import argparse, os, random, subprocess, sys, time, glob, shutil, math
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(
        description="Ramped request submitter (rates in requests per MINUTE)."
    )
    p.add_argument("--duration_min", type=float, default=10.0,
                   help="Total run time in minutes (default: 10)")
    p.add_argument("--start_rate", type=float, default=10.0,
                   help="Start rate in req/min (default: 10 = 0.1667 req/s)")
    p.add_argument("--end_rate", type=float, default=1.0,
                   help="End rate in req/min (default: 1 = 0.0167 req/s)")
    p.add_argument("--mode", choices=["linear","exp"], default="linear",
                   help="Rate ramp mode: linear or exp (geometric). Default: linear")
    p.add_argument("--host_url", default=os.getenv("HOST_URL","http://localhost:8000"),
                   help="Orchestrator base URL (default: http://localhost:8000)")
    p.add_argument("--task_endpoint", default=os.getenv("TASK_ENDPOINT","/api/v1/tasks"),
                   help="Task submission endpoint (default: /api/v1/tasks)")
    p.add_argument("--token", default=os.getenv("TOKEN","dev-token"),
                   help="Auth token (default: dev-token)")
    p.add_argument("--task_dir", default=str(Path(__file__).parent.resolve()),
                   help="Directory with *.yaml to submit randomly (default: script dir)")
    p.add_argument("--dry_run", action="store_true",
                   help="Do not POST, only print the schedule")
    p.add_argument("--print_every", type=int, default=1,
                   help="Print every N requests (default: 1)")
    args = p.parse_args()

    args.duration_sec = args.duration_min * 60.0
    if args.duration_sec <= 0 or args.start_rate <= 0 or args.end_rate <= 0:
        sys.exit("duration_min, start_rate, and end_rate must be positive.")
    return args

def rate_at_minute(t_sec, T_sec, r0_min, r1_min, mode):
    """Return rate (req/min) at time t in [0, T]."""
    x = 0.0 if T_sec <= 0 else max(0.0, min(t_sec / T_sec, 1.0))
    if mode == "linear":
        return r0_min + (r1_min - r0_min) * x
    else:
        # geometric interpolation: r(t)=r0*(r1/r0)^(t/T); exact at endpoints
        ratio = r1_min / r0_min
        return r0_min * (ratio ** x)

def pick_yaml(task_dir):
    files = sorted(glob.glob(os.path.join(task_dir, "*.yaml")))
    if not files:
        sys.exit(f"No YAML task files found in {task_dir}")
    return random.choice(files)

def post_yaml(host_url, endpoint, token, yaml_path, use_jq):
    url = f"{host_url}{endpoint}"
    cmd = [
        "curl", "-sS", "-X", "POST", url,
        "-H", f"Authorization: Bearer {token}",
        "-H", "Content-Type: text/yaml",
        "--data-binary", f"@{yaml_path}",
    ]
    if use_jq:
        p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p2 = subprocess.Popen(["jq", "."], stdin=p1.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p1.stdout.close()
        out, err = p2.communicate()
        rc = p2.returncode
        if rc != 0:
            out, err = p1.communicate()
            rc = p1.returncode
        return rc, out.decode(errors="ignore"), err.decode(errors="ignore")
    else:
        p = subprocess.run(cmd, capture_output=True, text=True)
        return p.returncode, p.stdout, p.stderr

def main():
    args = parse_args()
    start_time = time.perf_counter()
    end_time = start_time + args.duration_sec
    use_jq = shutil.which("jq") is not None
    sent = 0
    next_send = start_time

    print(f"Mode={args.mode} | Ramp {args.start_rate:.1f} -> {args.end_rate:.1f} req/min "
          f"over {args.duration_min:.1f} min ({args.duration_sec:.1f}s)")
    print(f"POST {args.host_url}{args.task_endpoint}  (task_dir={args.task_dir})")

    while True:
        now = time.perf_counter()
        if now >= end_time:
            break

        # current rate in req/min -> convert to req/sec for scheduling
        r_min = rate_at_minute(now - start_time, args.duration_sec,
                               args.start_rate, args.end_rate, args.mode)
        r_sec = max(r_min / 60.0, 1e-6)
        interval = 1.0 / r_sec

        # timing
        if now < next_send:
            time.sleep(next_send - now)
            now = time.perf_counter()

        # send one request
        yaml_path = pick_yaml(args.task_dir)
        if not args.dry_run:
            rc, out, err = post_yaml(args.host_url, args.task_endpoint, args.token, yaml_path, use_jq)
            sent += 1
            if (sent % args.print_every) == 0:
                base = os.path.basename(yaml_path)
                print(f"[{sent}] {base} -> rc={rc}")
                if rc != 0:
                    sys.stderr.write(err or out)
        else:
            sent += 1
            print(f"[dry-run {sent}] would send {yaml_path}")

        # next send time based on *current* rate
        next_send = now + interval

    elapsed = time.perf_counter() - start_time
    avg_rate_min = (sent / max(elapsed, 1e-9)) * 60.0
    print(f"Done. Sent={sent}, elapsed={elapsed/60:.2f} min, avg_rate={avg_rate_min:.2f} req/min")

if __name__ == "__main__":
    main()

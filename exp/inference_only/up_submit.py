#!/usr/bin/env python3
import argparse, os, random, subprocess, sys, time, glob, shutil
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(
        description="Ramped request submitter (rates in requests per MINUTE)."
    )
    # Total duration is split evenly across cycles
    p.add_argument("--duration_min", type=float, default=10.0,
                   help="Total run time in minutes (default: 10). Split evenly across cycles.")
    p.add_argument("--start_rate", type=float, default=0.6,
                   help="LOW rate in req/min to start each cycle (default: 0.6 = 0.01 req/s)")
    p.add_argument("--end_rate", type=float, default=6.0,
                   help="HIGH rate in req/min to end each cycle (default: 6 = 0.1 req/s)")
    p.add_argument("--mode", choices=["linear","exp"], default="linear",
                   help="Rate ramp mode: linear or exp (geometric). Default: linear")
    p.add_argument("--cycles", type=int, default=2,
                   help="Number of low→high cycles to run (default: 2)")
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
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for reproducible YAML selection (default: None)")
    args = p.parse_args()

    if args.duration_min <= 0:
        sys.exit("duration_min must be positive.")
    if args.start_rate <= 0 or args.end_rate <= 0:
        sys.exit("start_rate and end_rate must be positive.")
    if args.cycles <= 0:
        sys.exit("cycles must be a positive integer.")

    args.duration_sec_total = args.duration_min * 60.0
    args.duration_sec_cycle = args.duration_sec_total / args.cycles
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
        p2 = subprocess.Popen(["jq", "."], stdin=p1.stdout,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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

def run_one_cycle(args, cycle_idx, sent_so_far, use_jq):
    """Run a single low→high ramp cycle. Returns (sent_count_increment, elapsed_sec_cycle)."""
    cycle_start = time.perf_counter()
    cycle_end = cycle_start + args.duration_sec_cycle
    next_send = cycle_start
    sent_in_cycle = 0

    print(f"\n[Cycle {cycle_idx+1}/{args.cycles}] "
          f"{args.mode} ramp {args.start_rate:.2f} → {args.end_rate:.2f} req/min "
          f"over {args.duration_sec_cycle/60.0:.2f} min")

    while True:
        now = time.perf_counter()
        if now >= cycle_end:
            break

        # current rate in req/min -> convert to req/sec for scheduling
        r_min = rate_at_minute(
            now - cycle_start,
            args.duration_sec_cycle,
            args.start_rate,
            args.end_rate,
            args.mode,
        )
        r_sec = max(r_min / 60.0, 1e-6)
        interval = 1.0 / r_sec

        # timing throttle
        if now < next_send:
            time.sleep(next_send - now)
            now = time.perf_counter()

        # send one request
        yaml_path = pick_yaml(args.task_dir)
        if not args.dry_run:
            rc, out, err = post_yaml(
                args.host_url,
                args.task_endpoint,
                args.token,
                yaml_path,
                use_jq,
            )
            sent_so_far += 1
            sent_in_cycle += 1
            if (sent_so_far % args.print_every) == 0:
                base = os.path.basename(yaml_path)
                print(f"[{sent_so_far}] {base} -> rc={rc}")
                if rc != 0:
                    sys.stderr.write(err or out)
        else:
            sent_so_far += 1
            sent_in_cycle += 1
            print(f"[dry-run {sent_so_far}] would send {yaml_path}")

        # next send time based on *current* rate
        next_send = now + interval

    elapsed_cycle = time.perf_counter() - cycle_start
    avg_rate_cycle = (sent_in_cycle / max(elapsed_cycle, 1e-9)) * 60.0
    print(f"[Cycle {cycle_idx+1}] Sent={sent_in_cycle}, "
          f"elapsed={elapsed_cycle/60.0:.2f} min, avg_rate={avg_rate_cycle:.2f} req/min")

    return sent_in_cycle, elapsed_cycle

def main():
    args = parse_args()

    # apply RNG seed (affects pick_yaml randomness)
    if args.seed is not None:
        random.seed(args.seed)

    use_jq = shutil.which("jq") is not None
    print(f"Mode={args.mode} | Cycles={args.cycles} | "
          f"Ramp {args.start_rate:.2f} → {args.end_rate:.2f} req/min per cycle")
    print(f"Total duration={args.duration_min:.2f} min "
          f"({args.duration_sec_total:.1f}s), per-cycle={args.duration_sec_cycle/60.0:.2f} min")
    print(f"POST {args.host_url}{args.task_endpoint}  (task_dir={args.task_dir})")
    if args.seed is not None:
        print(f"Seed={args.seed}")

    global_start = time.perf_counter()
    total_sent = 0
    total_elapsed = 0.0

    for c in range(args.cycles):
        inc, elapsed = run_one_cycle(args, c, total_sent, use_jq)
        total_sent += inc
        total_elapsed += elapsed

    global_elapsed = time.perf_counter() - global_start
    avg_rate_min = (total_sent / max(global_elapsed, 1e-9)) * 60.0
    print(f"\nDone. TOTAL Sent={total_sent}, "
          f"elapsed={global_elapsed/60.0:.2f} min, avg_rate={avg_rate_min:.2f} req/min")

if __name__ == "__main__":
    main()

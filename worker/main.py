# worker/main.py
import redis
from utils import get_logger
from config import WorkerConfig
from hw import collect_hw
from lifecycle import Lifecycle
from redis_worker import RedisWorker
from runner import Runner
from power import PowerMonitor

from executors import EXECUTOR_REGISTRY, IMPORT_ERRORS, EXECUTOR_CLASS_NAMES


def initialize_executors(
    logger,
    *,
    registry: dict[str, object] | None = None,
    import_errors: dict[str, str] | None = None,
    cuda_available: bool | None = None,
):
    """Initialize executor registry and handle graceful degradation.

    Allows dependency/GPU overrides in tests; returns (executors, default_executor).
    """

    registry = registry or EXECUTOR_REGISTRY
    import_errors = import_errors or IMPORT_ERRORS

    def check_cuda() -> bool:
        if cuda_available is not None:
            return cuda_available
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def init_executor(key: str, *, gpu_required: bool = False):
        cls = registry.get(key)
        if cls is None:
            reason = import_errors.get(EXECUTOR_CLASS_NAMES.get(key, key), "dependency missing")
            logger.info("Skipping executor %s: %s", key, reason)
            return None

        if gpu_required and not check_cuda():
            logger.info("Executor %s requires a GPU; unavailable, skipping", key)
            return None

        try:
            return cls()
        except Exception as exc:  # noqa: broad-except
            logger.warning("Failed to initialize executor %s: %s", key, exc)
            return None

    executors: dict[str, object] = {}
    default_executor = init_executor("default")
    if not default_executor:
        raise SystemExit("HFTransformers executor unavailable; install inference extras (mloc[inference])")
    executors["default"] = default_executor

    for key in ["echo", "rag", "agent", "sft", "lora_sft"]:
        inst = init_executor(key)
        if inst:
            executors[key] = inst

    for key in ["vllm", "ppo", "dpo"]:
        inst = init_executor(key, gpu_required=True)
        if inst:
            executors[key] = inst

    return executors, default_executor

def main():
    cfg = WorkerConfig.from_env()
    logger = get_logger(name="worker", level=cfg.log_level)

    redis_url = cfg.redis_url
    if not redis_url:
        logger.warning("No REDIS_URL configured, falling back to default localhost")
        redis_url = "redis://localhost:6379/0"
    rds = redis.from_url(redis_url, decode_responses=True)
    rds.ping()

    rworker = RedisWorker(rds, cfg.worker_id)
    lifecycle = Lifecycle(
        rworker,
        cfg.hb_interval_sec,
        cfg.hb_ttl_sec,
        cost_per_hour=cfg.cost_per_hour,
        power_monitor=PowerMonitor(),
    )
    hardware = collect_hw(bandwidth_bytes_per_sec=cfg.network_bandwidth_bytes_per_sec)
    lifecycle.start(env={}, hardware=hardware, tags=cfg.tags)

    executors, default_executor = initialize_executors(logger)

    runner = Runner(
        lifecycle,
        rds,
        cfg.topic,
        cfg.results_dir,
        executors,
        default_executor,
        logger,
        network_bandwidth_bytes_per_sec=cfg.network_bandwidth_bytes_per_sec,
    )
    try:
        runner.start()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received; initiating shutdown")
    finally:
        lifecycle.shutdown()


if __name__ == "__main__":
    main()

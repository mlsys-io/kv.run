# worker/main.py
import redis
from utils import get_logger
from config import WorkerConfig
from hw import collect_hw
from lifecycle import Lifecycle
from redis_worker import RedisWorker
from runner import Runner

# Always available CPU path
from executors import HFTransformersExecutor, RAGExecutor, AgentExecutor

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
    lifecycle = Lifecycle(rworker, cfg.hb_interval_sec, cfg.hb_ttl_sec)
    lifecycle.start(env={}, hardware=collect_hw(), tags=cfg.tags)

    executors = {
        "default": HFTransformersExecutor(),   # CPU ok
        "rag": RAGExecutor(),
        "agent": AgentExecutor(),              # Agent tasks
    }

    # Try to add vLLM only if usable
    try:
        import torch
        if torch.cuda.is_available():
            from executors import VLLMExecutor
            executors["vllm"] = VLLMExecutor()
        else:
            logger.info("CUDA not available; skipping vLLM executor")
    except Exception as e:
        logger.info("vLLM not installed/usable; skipping (%s)", e)

    # (Optional) other GPU-bound executors with similar guards
    try:
        from executors import PPOExecutor, DPOExecutor
        # If GPU not available, these will raise
        executors["ppo"] = PPOExecutor()
        executors["dpo"] = DPOExecutor()
    except Exception as e:
        logger.info("PPO/DPO not available; skipping (%s)", e)

    # Pick a safe default
    default_executor = executors.get("default")

    Runner(lifecycle, rds, cfg.topic, cfg.results_dir, executors, default_executor, logger).start()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

# worker/main.py
import redis
from utils import get_logger
from config import WorkerConfig
from hw import collect_hw
from lifecycle import Lifecycle
from redis_worker import RedisWorker
from runner import Runner

# Always available CPU path
from executors import EXECUTOR_REGISTRY, IMPORT_ERRORS, EXECUTOR_CLASS_NAMES

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

    def _init_executor(key: str, *, gpu_required: bool = False):
        """按需初始化执行器，并在缺失依赖时输出中文提示。"""
        cls = EXECUTOR_REGISTRY.get(key)
        if cls is None:
            reason = IMPORT_ERRORS.get(EXECUTOR_CLASS_NAMES.get(key, key), "可选依赖未安装")
            logger.info("跳过执行器 %s：%s", key, reason)
            return None
        if gpu_required:
            try:
                import torch
                if not torch.cuda.is_available():
                    logger.info("执行器 %s 需要 GPU 环境，当前不可用，已跳过", key)
                    return None
            except Exception as exc:
                logger.info("检查 GPU 能力失败（执行器 %s）：%s", key, exc)
                return None
        try:
            return cls()
        except Exception as exc:
            logger.warning("初始化执行器 %s 失败：%s", key, exc)
            return None

    executors = {}
    default_executor = _init_executor("default")
    if not default_executor:
        logger.error("HFTransformers 执行器不可用，请安装 inference 扩展依赖 (pip install mloc[inference])")
        raise SystemExit("缺少 CPU 推理执行器依赖，无法启动 worker")
    executors["default"] = default_executor  # CPU ok

    for key in ["echo", "rag", "agent", "sft", "lora_sft"]:
        inst = _init_executor(key)
        if inst:
            executors[key] = inst

    vllm_executor = _init_executor("vllm", gpu_required=True)
    if vllm_executor:
        executors["vllm"] = vllm_executor

    for key in ["ppo", "dpo"]:
        inst = _init_executor(key, gpu_required=True)
        if inst:
            executors[key] = inst

    runner = Runner(lifecycle, rds, cfg.topic, cfg.results_dir, executors, default_executor, logger)
    try:
        runner.start()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received; initiating shutdown")
    finally:
        lifecycle.shutdown()


if __name__ == "__main__":
    main()

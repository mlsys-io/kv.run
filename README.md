# MLOC - Modular LLM Operations Container

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116+-green.svg)](https://fastapi.tiangolo.com)
[![Redis](https://img.shields.io/badge/Redis-6.4+-red.svg)](https://redis.io)
[![vLLM](https://img.shields.io/badge/vLLM-0.10+-purple.svg)](https://vllm.ai)

MLOC is a distributed system for Large Language Model (LLM) inference and fine-tuning operations. It provides a scalable, fault-tolerant architecture using an orchestrator-worker pattern with Redis as the message broker.

## Architecture

```
+-------------+    HTTP API    +--------------+    Redis Pub/Sub    +-------------+
|   Client    | -------------> | Orchestrator | -----------------> |   Worker    |
+-------------+                +--------------+                    +-------------+
                                      |                                    |
                                      |                                    |
                                      v                                    v
                              +-------------+                      +-------------+
                              |    Redis    |                      |  Executors  |
                              |  (Message   |                      |  (vLLM,     |
                              |   Broker)   |                      |   etc.)     |
                              +-------------+                      +-------------+
```

### Core Components

- **Orchestrator**: Central service for task scheduling and worker management
- **Worker**: Execution nodes that process LLM tasks
- **Redis**: Message broker and state store
- **Executors**: Pluggable task execution modules
  - **vLLM Executor**: High-performance LLM inference
  - **PPO Executor**: Reinforcement learning training with Proximal Policy Optimization

## Features

- **Distributed Task Execution**: Scale horizontally by adding more workers
- **Resource-Aware Scheduling**: Intelligent task assignment based on hardware requirements
- **Fault Tolerance**: Heartbeat monitoring and automatic cleanup of stale workers
- **Multiple Executors**: Support for both inference and training workflows
  - **vLLM Inference**: High-throughput text generation
  - **PPO Training**: Reinforcement learning fine-tuning
- **YAML Task Definitions**: Declarative task specification
- **RESTful API**: HTTP endpoints for task submission and monitoring

## Installation

### Prerequisites

- Python 3.12+
- Redis server
- NVIDIA GPU (for GPU inference)
- CUDA toolkit (for GPU support)

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd kv.run
   ```

2. **Install dependencies**:
   
   For basic inference functionality:
   ```bash
   uv sync
   ```
   
   For PPO training functionality:
   ```bash
   uv sync --extra ppo
   ```
   
   For development:
   ```bash
   uv sync --extra ppo --extra dev
   ```

3. **Start Redis** (if not already running):
   ```bash
   redis-server
   ```

## Quick Start

### 1. Start the Orchestrator

```bash
export REDIS_URL="redis://localhost:6379/0"
export ORCHESTRATOR_TOKEN="dev-token"  # Optional: for API authentication

python -m orchestrator.server
```

The orchestrator will be available at `http://localhost:8000`

### 2. Start a Worker

For inference tasks only:
```bash
export REDIS_URL="redis://localhost:6379/0"
export TASK_TOPICS="tasks.inference"
export RESULTS_DIR="./results"

python -m worker.listener
```

For both inference and PPO training:
```bash
export REDIS_URL="redis://localhost:6379/0"
# Note: Don't set TASK_TOPICS to use default (tasks.inference,tasks.ppo)
export RESULTS_DIR="./results"

python -m worker.listener
```

### 3. Submit Tasks

#### For vLLM Inference:
```bash
cd client
./vllm_inference.sh
```

#### For PPO Training:
```bash
cd client
./ppo_training.sh
```

#### Or submit directly via curl:

Inference task:
```bash
curl -X POST "http://localhost:8000/api/v1/tasks" \
  -H "Authorization: Bearer dev-token" \
  -H "Content-Type: text/yaml" \
  --data-binary @templates/inference_vllm_mistral.yaml
```

PPO training task:
```bash
curl -X POST "http://localhost:8000/api/v1/tasks" \
  -H "Authorization: Bearer dev-token" \
  -H "Content-Type: text/yaml" \
  --data-binary @templates/ppo_training_mistral.yaml
```

## Task Definition

Tasks are defined using YAML files.

### vLLM Inference Task Example

```yaml
apiVersion: mloc/v1
kind: InferenceTask
metadata:
  name: mistral-7b-infer
  owner: alice

spec:
  taskType: "inference"
  
  resources:
    replicas: 1
    hardware:
      cpu: "8"
      memory: "32Gi"
      gpu:
        type: "any"
        count: 1

  model:
    source:
      type: "huggingface"
      identifier: "mistralai/Mistral-7B-Instruct-v0.1"
    vllm:
      tensor_parallel_size: 1
      gpu_memory_utilization: 0.9

  inference:
    max_tokens: 128
    temperature: 0.7
    prompts:
      - "Explain quantum computing in simple terms."
      - "Write a Python function to sort a list."
```

### PPO Training Task Example

```yaml
apiVersion: mloc/v1
kind: TrainingTask
metadata:
  name: mistral-7b-ppo-training
  owner: alice

spec:
  taskType: "ppo"
  
  resources:
    replicas: 1
    hardware:
      cpu: "16"
      memory: "64Gi"
      gpu:
        type: "any"
        count: 1

  model:
    source:
      type: "huggingface"
      identifier: "mistralai/Mistral-7B-Instruct-v0.1"
    config:
      fp16: true
      device_map_auto: true

  reward_model:
    identifier: "cardiffnlp/twitter-roberta-base-sentiment-latest"
    type: "sentiment"

  data:
    prompts:
      - "Write a helpful response: How can I improve my productivity?"
      - "Create a motivational message for someone learning to code:"
      - "Explain quantum computing in an encouraging way:"

  training:
    learning_rate: 1.41e-5
    batch_size: 4
    steps: 50
    ppo_epochs: 4
    target_kl: 0.1
    save_model: true

  generation:
    max_new_tokens: 256
    temperature: 0.7
```

## Configuration

### Orchestrator Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | Required | Redis connection URL |
| `ORCHESTRATOR_TOKEN` | None | Bearer token for API authentication |
| `HEARTBEAT_TTL_SEC` | 120 | Worker heartbeat timeout |
| `PORT` | 8000 | HTTP server port |
| `LOG_LEVEL` | INFO | Logging level |

### Worker Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | Required | Redis connection URL |
| `TASK_TOPICS` | "tasks.inference,tasks.ppo" | Comma-separated list of topics to subscribe |
| `RESULTS_DIR` | "./results" | Directory for task results |
| `HEARTBEAT_INTERVAL_SEC` | 30 | Heartbeat interval |
| `WORKER_ID` | auto-generated | Fixed worker ID |
| `WORKER_TAGS` | None | Comma-separated worker tags |

### Executor Environment Variables

| Variable | Description |
|----------|-------------|
| `VLLM_MODEL` | Default vLLM model identifier |
| `PPO_MODEL` | Default PPO training model identifier |

## API Reference

### Orchestrator Endpoints

#### Worker Management

- `GET /workers` - List all workers
- `GET /workers/{worker_id}` - Get worker details
- `POST /admin/cleanup` - Clean up stale workers

#### Task Management

- `POST /api/v1/tasks` - Submit a new task
- `GET /api/v1/tasks` - List all tasks
- `GET /api/v1/tasks/{task_id}` - Get task details

#### Health Check

- `GET /healthz` - Health check endpoint

### Task Submission Example

```bash
curl -X POST "http://localhost:8000/api/v1/tasks" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: text/yaml" \
  --data-binary @task.yaml
```

Response:
```json
{
  "ok": true,
  "task_id": "123e4567-e89b-12d3-a456-426614174000",
  "worker_id": "worker-001",
  "topic": "tasks.inference"
}
```

## Development

### Adding Custom Executors

1. Create a new executor class inheriting from `Executor`:

```python
from worker.executors.base_executor import Executor, ExecutionError

class MyCustomExecutor(Executor):
    name = "my-executor"
    
    def run(self, task: dict, out_dir: Path) -> dict:
        # Your custom logic here
        result = {"ok": True, "message": "Task completed"}
        self.save_json(out_dir / "responses.json", result)
        return result
```

2. Register the executor in `worker/listener.py`:

```python
from worker.executors import MyCustomExecutor

# In Runner.__init__()
self.executor = MyCustomExecutor()
```

### Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/
```

### Code Formatting

```bash
# Format code
black .
isort .

# Lint
ruff check .
```

## Monitoring

### Worker Status

Check worker status via the API:

```bash
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/workers"
```

### Task Results

Task results are stored in `RESULTS_DIR/<task_id>/responses.json`:

```json
{
  "ok": true,
  "model": "mistralai/Mistral-7B-Instruct-v0.1",
  "items": [
    {
      "index": 0,
      "output": "Generated text response...",
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 155,
    "total_tokens": 180,
    "latency_sec": 0.85,
    "num_requests": 1
  }
}
```

### Logs

- Orchestrator logs: `orchestrator.log` (configurable)
- Worker logs: stdout/stderr

## Troubleshooting

### Common Issues

#### 1. Worker Not Subscribing to PPO Tasks

**Problem**: Worker only shows `subscribed to topics: ['tasks.inference']`

**Solution**: 
```bash
# Clear any existing TASK_TOPICS environment variable
unset TASK_TOPICS

# Restart worker (it will use default topics: tasks.inference,tasks.ppo)
export REDIS_URL="redis://localhost:6379/0"
export RESULTS_DIR="./results"
python -m worker.listener
```

**Root Cause**: Previously set `TASK_TOPICS` environment variable overrides the default configuration.

#### 2. PPO Training Dependencies Missing

**Problem**: `PPO dependencies not installed` error

**Solution**:
```bash
# Install PPO dependencies
uv sync --extra ppo
# OR
pip install trl transformers torch datasets accelerate
```

#### 3. Task Submission Successful But No Worker Response

**Debug Steps**:
1. **Check worker logs** - Should show task acceptance and executor selection
2. **Verify topic subscription** - Worker should subscribe to correct topics
3. **Check Redis connection** - Both orchestrator and worker need Redis access
4. **Verify task assignment** - Check if task was assigned to the correct worker

**Debug Commands**:
```bash
# Check Redis connectivity
redis-cli ping

# Check worker subscription
# Look for: "subscribed to topics: ['tasks.inference', 'tasks.ppo']"

# Check task assignment in orchestrator logs
# Look for: "Publish to topic=tasks.ppo receivers=1"
```

#### 4. GPU Memory Issues During PPO Training

**Problem**: CUDA out of memory errors

**Solutions**:
- Reduce `batch_size` in training config
- Enable `fp16: true` in model config
- Reduce `max_new_tokens` in generation config
- Use `gradient_accumulation_steps` to simulate larger batches

#### 5. Slow PPO Training

**Optimization Tips**:
- Use smaller models for experimentation
- Reduce number of training steps
- Use gradient accumulation instead of large batch sizes
- Enable `optimize_cuda_cache: true`

### Debug Mode

To enable detailed debugging:

```bash
# Set debug logging level
export LOG_LEVEL="DEBUG"

# Start worker with debug output
python -m worker.listener
```

Look for these key log messages:
- `TASK_TOPICS environment variable: None`
- `Using topics: ['tasks.inference', 'tasks.ppo']`
- `Selected executor: PPOExecutor for task_type: ppo`
- `Starting PPO training task`
- `PPO Step 0/50: mean_reward=0.1234`

## Deployment

### Docker Deployment

1. **Build images**:
```bash
# Orchestrator
docker build -t mloc-orchestrator -f docker/Dockerfile.orchestrator .

# Worker
docker build -t mloc-worker -f docker/Dockerfile.worker .
```

2. **Run with Docker Compose**:
```bash
docker-compose up -d
```

### Production Considerations

- Use a Redis cluster for high availability
- Set up proper logging aggregation
- Monitor GPU utilization and memory usage
- Configure resource limits for containers
- Use a reverse proxy (nginx) for the orchestrator
- Set up SSL/TLS for production deployments

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [vLLM](https://vllm.ai) for high-performance LLM inference
- [FastAPI](https://fastapi.tiangolo.com) for the web framework
- [Redis](https://redis.io) for reliable message brokering

## Support

- Create an issue for bug reports or feature requests
- Check the [documentation](docs/) for detailed guides
- Join our community discussions
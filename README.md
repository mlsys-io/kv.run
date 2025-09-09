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
- **Executors**: Pluggable task execution modules (vLLM, custom executors)

## Features

- **Distributed Task Execution**: Scale horizontally by adding more workers
- **Resource-Aware Scheduling**: Intelligent task assignment based on hardware requirements
- **Fault Tolerance**: Heartbeat monitoring and automatic cleanup of stale workers
- **Pluggable Executors**: Support for different LLM frameworks (vLLM included)
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
   ```bash
   uv sync
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

```bash
export REDIS_URL="redis://localhost:6379/0"
export TASK_TOPICS="tasks.inference"
export RESULTS_DIR="./results"

python -m worker.listener
```

### 3. Submit a Task

Use the provided client script:

```bash
cd client
./vllm_inference.sh
```

Or submit directly via curl:

```bash
curl -X POST "http://localhost:8000/api/v1/tasks" \
  -H "Authorization: Bearer dev-token" \
  -H "Content-Type: text/yaml" \
  --data-binary @templates/inference_vllm_mistral.yaml
```

## Task Definition

Tasks are defined using YAML files. Here's an example for vLLM inference:

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
| `TASK_TOPICS` | "tasks.inference" | Comma-separated list of topics to subscribe |
| `RESULTS_DIR` | "./results" | Directory for task results |
| `HEARTBEAT_INTERVAL_SEC` | 30 | Heartbeat interval |
| `WORKER_ID` | auto-generated | Fixed worker ID |
| `WORKER_TAGS` | None | Comma-separated worker tags |

### vLLM Executor Environment Variables

| Variable | Description |
|----------|-------------|
| `VLLM_MODEL` | Default model identifier |

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
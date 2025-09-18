# Agent Executor - Youtu-Agent Integration

This directory integrates the [Youtu-Agent (UTU)](https://github.com/TencentCloudADP/youtu-agent) framework into MLOC as a pluggable executor to provide autonomous agent capabilities.

## Acknowledgments

We would like to thank the **Youtu-Agent (UTU)** team for their excellent open-source contribution. This integration leverages their framework to bring powerful autonomous agent capabilities to the MLOC distributed system.

- **Original Project**: [Youtu-Agent](https://github.com/TencentCloudADP/youtu-agent)
- **License**: Apache 2.0
- **Integration Purpose**: Autonomous agent task execution within MLOC workers

## About the Integration

Youtu-Agent is a comprehensive framework for building, running, and evaluating autonomous agents based on open-source models. By integrating it into MLOC, we enable:

- **Streaming Agent Execution**: Real-time agent task processing with progress tracking
- **Multi-Agent Support**: Various pre-configured agent types for different use cases
- **Tool Integration**: Search, document processing, code execution, and web browsing capabilities
- **Distributed Processing**: Agent tasks distributed across MLOC worker nodes

## Quick Setup

### 1. Configure API Keys

Copy the example configuration file and add your API keys:

```bash
cd worker/executors/agent/
cp secrets.yaml.example secrets.yaml
```

Edit `secrets.yaml` with your configuration:

```yaml
# secrets.yaml - DO NOT commit this file
# LLM Configuration (Required)
UTU_LLM_TYPE: "chat.completions"
UTU_LLM_MODEL: "deepseek-chat"  # or gpt-4, gpt-3.5-turbo, etc.
UTU_LLM_BASE_URL: "https://api.deepseek.com"  # or your API endpoint
UTU_LLM_API_KEY: "your-llm-api-key-here"

# Tool API Keys (Optional but recommended)
SERPER_API_KEY: "your-serper-api-key"  # For web search functionality
JINA_API_KEY: "your-jina-api-key"      # For web content extraction
```

### 2. Install Dependencies

The Youtu-Agent framework is already included. Install any additional dependencies:

```bash
# From the agent directory
cd worker/executors/agent/utu
uv sync --all-extras
```

### 3. Test the Integration

Run a simple agent task:

```bash
# From the root directory
./client/agent_search.sh
```

## Available Agent Configurations

The integration supports multiple pre-configured agent types:

| Config Name | Description | Use Case |
|-------------|-------------|----------|
| `default` | General-purpose agent with search tools | General queries and research |
| `base` | Basic agent without external tools | Simple text processing |
| `simple_search` | Search-focused agent | Information retrieval and web search |
| `examples/paper_collector` | Academic research agent | Literature review and paper collection |

## Task Templates

Use the provided YAML templates in the `templates/` directory:

### Basic Search Agent
```yaml
# templates/agent_query_search.yaml
apiVersion: mloc/v1
kind: AgentTask
spec:
  taskType: "agent"
  configName: "simple_search"
  task: "Search for the latest AI research trends and provide a summary"
```

### Academic Paper Collector
```yaml
# templates/agent_paper_collector.yaml
apiVersion: mloc/v1
kind: AgentTask
spec:
  taskType: "agent"
  configName: "examples/paper_collector"
  task: "Find 3 recent papers about machine learning and summarize them"
```

## Supported LLM Providers

The integration works with various LLM providers through OpenAI-compatible APIs:

- **DeepSeek**: `deepseek-chat`, `deepseek-coder`
- **OpenAI**: `gpt-4`, `gpt-3.5-turbo`, `gpt-4-turbo`
- **Local Models**: Via Ollama, vLLM, or other OpenAI-compatible servers
- **Cloud Providers**: Azure OpenAI, AWS Bedrock (with compatible endpoints)

## Output Structure

Agent tasks produce standardized MLOC output format:

```json
{
  "ok": true,
  "config_name": "simple_search",
  "task": "User's original task description",
  "result": "Agent's final response",
  "execution_log": [
    "Loaded config: simple_search",
    "Created agent: SearchAgent",
    "Tool call: web_search",
    "Generated output: 1500 characters"
  ],
  "usage": {
    "tokens": 1250
  },
  "items": [
    {
      "response": "Agent's final response"
    }
  ]
}
```

## Configuration Files

### secrets.yaml (Required)
Contains API keys and LLM configuration. **Never commit this file to version control.**

### secrets.yaml.example (Template)
Template file showing required configuration structure. Copy and modify this file.

## Contributing

When contributing to this integration:

1. Respect the Youtu-Agent project's Apache 2.0 license
2. Keep integration code minimal and focused
3. Update documentation for configuration changes
4. Test with multiple agent configurations and models

## License

This integration maintains compatibility with:
- **MLOC Project License**
- **Youtu-Agent Apache 2.0 License**

Special thanks to the Youtu-Agent team for their valuable open-source contribution that makes this integration possible.
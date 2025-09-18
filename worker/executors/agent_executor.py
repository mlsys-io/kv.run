#!/usr/bin/env python3
"""
Agent Executor

Integrates youtu-agent framework into MLOC system.
Supports task-based agent execution with streaming output and progress tracking.
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add agent directory to sys.path for utu imports
agent_dir = Path(__file__).parent / "agent"
sys.path.insert(0, str(agent_dir))

from .base_executor import Executor, ExecutionError

logger = logging.getLogger("worker.agent")
logger.propagate = False  # Prevent duplicate logs

# Configure agent-specific concise log format
if not logger.handlers:
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)

class AgentExecutor(Executor):
    """Agent executor using youtu-agent (utu) framework"""

    name = "agent"

    def __init__(self):
        super().__init__()
        self._initialized = False

    def prepare(self) -> None:
        """Initialize utu (youtu-agent) dependencies"""
        if self._initialized:
            return

        try:
            # Load secrets from secrets.yaml if environment variables are not set
            self._setup_environment_from_secrets()

            # Import utu modules from the correct path
            from utu.agents import get_agent
            from utu.config import AgentConfig, ConfigLoader
            from utu.utils import setup_logging, PrintUtils

            logger.debug("utu (youtu-agent) modules imported successfully")
            self._initialized = True
        except ImportError as e:
            logger.error(f"utu modules import failed: {e}")
            logger.error(f"Working directory: {os.getcwd()}")
            logger.error(f"utu path exists: {(agent_dir / 'utu').exists()}")
            raise ExecutionError(f"Failed to import utu modules: {e}")

    def _setup_environment_from_secrets(self):
        """Setup environment variables from secrets.yaml if not already set"""
        import yaml

        secrets_path = agent_dir / "secrets.yaml"

        if not secrets_path.exists():
            logger.warning(f"secrets.yaml not found at {secrets_path}")
            return

        try:
            with open(secrets_path, 'r', encoding='utf-8') as f:
                secrets = yaml.safe_load(f) or {}

            # Set UTU environment variables if not already set
            # Read configuration values directly from secrets.yaml
            env_mappings = {
                'UTU_LLM_TYPE': secrets.get('UTU_LLM_TYPE'),
                'UTU_LLM_MODEL': secrets.get('UTU_LLM_MODEL'),
                'UTU_LLM_BASE_URL': secrets.get('UTU_LLM_BASE_URL'),
                'UTU_LLM_API_KEY': secrets.get('UTU_LLM_API_KEY'),
                'SERPER_API_KEY': secrets.get('SERPER_API_KEY'),
                'JINA_API_KEY': secrets.get('JINA_API_KEY'),
            }

            # Record successfully loaded configurations
            loaded_configs = []
            for env_var, value in env_mappings.items():
                if value and not os.getenv(env_var):
                    os.environ[env_var] = str(value)
                    if 'API_KEY' in env_var:
                        # Only show first few and last few characters of API keys
                        masked_value = f"{str(value)[:6]}...{str(value)[-4:]}"
                        logger.info(f"✅ Loaded from secrets.yaml: {env_var} = {masked_value}")
                    else:
                        logger.info(f"✅ Loaded from secrets.yaml: {env_var} = {value}")
                    loaded_configs.append(env_var)
                elif os.getenv(env_var):
                    logger.debug(f"Environment variable already exists: {env_var}")
                else:
                    logger.warning(f"⚠️ Missing configuration: {env_var}")

            if loaded_configs:
                logger.info(f"📋 Loaded {len(loaded_configs)} configuration items in total")
            else:
                logger.warning("⚠️ No configurations loaded from secrets.yaml")

        except Exception as e:
            logger.warning(f"Failed to load secrets.yaml: {e}")
            # Continue without secrets, will use environment variables if available

    def run(self, task: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
        """Execute agent task using youtu-agent (utu) framework"""
        self.ensure_dir(out_dir)

        # Initialize utu modules
        self.prepare()

        spec = task.get("spec", {})
        agent_config_name = spec.get("configName", "default")
        task_input = spec.get("task", "")


        # Execute agent task using the same pattern as run_once.py
        try:
            result = asyncio.run(self._run_agent_task(agent_config_name, task_input, out_dir))

            output = {
                "ok": True,
                "config_name": agent_config_name,
                "task": task_input,
                "result": result.get("output", ""),
                "execution_log": result.get("log", []),
                "usage": result.get("usage", {}),
                "items": [{"response": result.get("output", "")}]
            }

            self.save_json(out_dir / "responses.json", output)
            logger.debug("Agent task completed successfully")
            return output

        except Exception as e:
            logger.exception(f"Agent task failed: {e}")
            error_output = {
                "ok": False,
                "config_name": agent_config_name,
                "task": task_input,
                "error": str(e),
                "items": []
            }
            self.save_json(out_dir / "responses.json", error_output)
            return error_output

    async def _run_agent_task(self, config_name: str, task_input: str, out_dir: Path) -> Dict[str, Any]:
        """Execute agent task using utu framework with detailed logging and timeout control"""
        import asyncio
        from utu.agents import get_agent
        from utu.config import AgentConfig, ConfigLoader
        from utu.tracing.setup import setup_tracing

        logger.info("🚀 Starting agent task execution")
        logger.info(f"📋 Configuration: {config_name}")
        logger.info(f"📝 Task description: {task_input[:200]}...")
        logger.debug(f"Output directory: {out_dir}")

        # Setup tracing (optional)
        try:
            setup_tracing()
            logger.debug("Tracing setup completed")
        except Exception as e:
            logger.warning(f"Failed to setup tracing: {e}")

        # Setup utu logging directory
        utu_log_dir = out_dir / "utu_logs"
        utu_log_dir.mkdir(exist_ok=True)
        logger.debug(f"Created log directory: {utu_log_dir}")

        execution_log = []
        usage_stats = {}
        
        try:
            # Load agent configuration
            logger.info("Loading agent configuration...")
            config: AgentConfig = ConfigLoader.load_agent_config(config_name)
            logger.info(f"✅ Configuration loaded: {config.name if hasattr(config, 'name') else config_name}")
            execution_log.append(f"Loaded config: {config_name}")

            # Create agent instance
            logger.info("Creating agent instance...")
            agent = get_agent(config)
            logger.info(f"✅ Agent created: {type(agent).__name__}")
            execution_log.append(f"Created agent: {type(agent).__name__}")

            # Build agent (loading tools and models)
            logger.info("Building agent (loading tools and models)...")
            await asyncio.wait_for(agent.build(), timeout=120)  # 2 minute timeout
            logger.info("✅ Agent built successfully")
            execution_log.append("Agent built successfully")

            # Execute the task
            logger.info("🎯 Starting task execution...")
            logger.debug(f"Task input length: {len(task_input)} characters")
            execution_log.append(f"Starting task execution: {task_input[:100]}...")

            # Set timeout duration
            task_timeout = 600  # 10 minute timeout
            logger.debug(f"Task timeout set to {task_timeout} seconds")

            # Use streaming execution (agent supports it based on usage patterns)
            logger.info("Using streaming execution with progress tracking...")
            result_streaming = agent.run_streamed(input=task_input)

            # Monitor execution stream
            async for event in result_streaming.stream_events():
                # Check agent switching events
                if hasattr(event, 'new_agent') and event.new_agent:
                    agent_name = getattr(event.new_agent, 'name', 'Unknown')
                    logger.info(f"🤖 Agent switched: {agent_name}")
                    execution_log.append(f"Agent switched: {agent_name}")
                
                # Check Orchestra task planning events
                elif hasattr(event, 'name') and hasattr(event, 'item'):
                    if event.name == 'plan':
                        logger.info("📋 Task planning completed")
                        execution_log.append("Task planning completed") 
                        if event.item:
                            plan_summary = str(event.item)[:200]
                            logger.info(f"📝 Planning content: {plan_summary}...")
                            
                # Check RunItem events (tool calls, outputs, etc.)
                elif hasattr(event, 'item') and hasattr(event.item, 'type'):
                    item = event.item
                    item_type = item.type
                    agent_name = getattr(getattr(item, 'agent', None), 'name', 'Unknown')
                    
                    if item_type == "tool_call_item":
                        if hasattr(item, 'raw_item'):
                            tool_name = getattr(item.raw_item, 'name', 'Unknown Tool')
                            tool_args = getattr(item.raw_item, 'arguments', '')[:100]
                            logger.info(f"🔧 [{agent_name}] Tool call: {tool_name}({tool_args}...)")
                            execution_log.append(f"Tool call: {tool_name}")
                            
                    elif item_type == "tool_call_output_item":
                        if hasattr(item, 'output'):
                            output_preview = str(item.output)[:150]
                            logger.info(f"📤 [{agent_name}] Tool output: {output_preview}...")
                            execution_log.append(f"Tool output received")
                            
                    elif item_type == "message_output_item":
                        if hasattr(item, 'content') and item.content:
                            for content_item in item.content:
                                if getattr(content_item, 'type', None) == 'text':
                                    text_preview = getattr(content_item, 'text', '')[:100]
                                    if text_preview.strip():
                                        logger.info(f"💬 [{agent_name}] Output: {text_preview}...")
                                        
                    elif item_type == "reasoning_item":
                        if hasattr(item, 'summary'):
                            for summary_item in item.summary:
                                if getattr(summary_item, 'type', None) == 'summary_text':
                                    reasoning_text = getattr(summary_item, 'text', '')[:100]
                                    if reasoning_text.strip():
                                        logger.info(f"🧠 [{agent_name}] Reasoning: {reasoning_text}...")
                
                # Check raw response events (including token generation)
                elif hasattr(event, 'data') and hasattr(event.data, 'type'):
                    # Character counting is removed.
                    pass
                            
            # Wait for streaming result to complete
            await asyncio.wait_for(result_streaming._run_impl_task, timeout=task_timeout)
            result = result_streaming
            logger.info(f"✅ Execution completed")
            execution_log.append(f"Streaming execution completed")

        except asyncio.TimeoutError:
            logger.error(f"⏰ Task timeout ({task_timeout} seconds)")
            raise ExecutionError(f"Agent task timed out")
        except Exception as e:
            logger.error(f"❌ Error during execution: {str(e)}")
            raise ExecutionError(f"Agent task failed: {str(e)}")

        # Extract output based on result type
        if hasattr(result, "final_output") and result.final_output:
            output = str(result.final_output)
            logger.info(f"📄 Final output extracted: {len(output)} characters")
        else:
            # Fallback: try common attributes
            output = getattr(result, "output", None) or getattr(result, "final", None) or str(result)
            logger.info(f"📄 Output extracted (fallback): {len(output)} characters")

        execution_log.append(f"Generated output: {len(output)} characters")

        # Save output to file
        output_file = out_dir / "agent_output.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output)
        logger.info(f"💾 Output saved to: {output_file}")
        execution_log.append(f"Saved output to file")

        # Log first 100 characters of output for debugging
        logger.info(f"Final output preview ({len(output)} characters): {output[:100]}...")

        # Cleanup agent resources
        try:
            logger.debug("Starting agent cleanup...")
            await asyncio.wait_for(agent.cleanup(), timeout=30)
            logger.debug("Agent cleanup completed")
            execution_log.append("Agent cleanup completed")
        except Exception as e:
            logger.warning(f"Agent cleanup failed: {e}")
            execution_log.append(f"Agent cleanup failed: {e}")

        return {
            "output": output,
            "log": execution_log,
            "usage": usage_stats
        }


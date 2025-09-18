from agents import Agent, Runner, RunResultStreaming, TResponseInputItem, trace

from ..config import ModelConfigs
from ..utils import AgentsUtils, get_logger
from .base_agent import BaseAgent
from .common import TaskRecorder

logger = get_logger(__name__)


class LLMAgent(BaseAgent):
    """Minimal agent that wraps a model."""

    def __init__(self, config: ModelConfigs):
        self.config = config
        self.agent = Agent(
            name="LLMAgent",
            model=AgentsUtils.get_agents_model(**config.model_provider.model_dump()),
            model_settings=config.model_settings,
            # tools=config.tools,
            # output_type=config.output_type,
        )

    def set_instructions(self, instructions: str):
        self.agent.instructions = instructions

    async def run(self, input: str | list[TResponseInputItem], trace_id: str = None) -> TaskRecorder:
        # TODO: customized the agent name
        trace_id = trace_id or AgentsUtils.gen_trace_id()
        task_recorder = TaskRecorder(input, trace_id)

        if AgentsUtils.get_current_trace():
            run_result = await Runner.run(self.agent, input)
        else:
            trace_id = trace_id or AgentsUtils.gen_trace_id()
            with trace(workflow_name="llm_agent", trace_id=trace_id):
                run_result = await Runner.run(self.agent, input)
        task_recorder.add_run_result(run_result)
        task_recorder.set_final_output(run_result.final_output)
        return task_recorder

    def run_streamed(self, input: str | list[TResponseInputItem], trace_id: str = None) -> RunResultStreaming:
        if AgentsUtils.get_current_trace():
            return Runner.run_streamed(self.agent, input)
        else:
            trace_id = trace_id or AgentsUtils.gen_trace_id()
            with trace(workflow_name="llm_agent", trace_id=trace_id):
                return Runner.run_streamed(self.agent, input)

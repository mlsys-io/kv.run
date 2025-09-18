import asyncio
import json

from agents import AgentUpdatedStreamEvent, trace
from agents._run_impl import QueueCompleteSentinel
from agents.tracing import function_span

from ..config import AgentConfig, ConfigLoader
from ..utils import AgentsUtils, get_logger
from .base_agent import BaseAgent
from .orchestra import (
    AnalysisResult,
    BaseWorkerAgent,
    CreatePlanResult,
    OrchestraStreamEvent,
    OrchestraTaskRecorder,
    PlannerAgent,
    ReporterAgent,
    SimpleWorkerAgent,
    Subtask,
    WorkerResult,
)

logger = get_logger(__name__)


class OrchestraAgent(BaseAgent):
    def __init__(self, config: AgentConfig | str):
        """Initialize the orchestra agent"""
        if isinstance(config, str):
            config = ConfigLoader.load_agent_config(config)
        self.config = config
        # init subagents
        self.planner_agent = PlannerAgent(config)
        self.worker_agents = self._setup_workers()
        self.reporter_agent = ReporterAgent(config)

    def set_planner(self, planner: PlannerAgent):
        self.planner_agent = planner

    def _setup_workers(self) -> dict[str, BaseWorkerAgent]:
        workers = {}
        for name, config in self.config.workers.items():
            assert config.type == "simple", f"Only support SimpleAgent as worker in orchestra agent, get {config}"
            workers[name] = SimpleWorkerAgent(config=config)
        return workers

    async def build(self):
        await self.planner_agent.build()
        for worker_agent in self.worker_agents.values():
            await worker_agent.build()
        await self.reporter_agent.build()

    async def run(self, input: str, trace_id: str = None) -> OrchestraTaskRecorder:
        """Run the orchestra agent

        1. plan
        2. sequentially execute subtasks
        3. report
        """
        # setup
        trace_id = trace_id or AgentsUtils.gen_trace_id()
        logger.info(f"> trace_id: {trace_id}")

        # TODO: error_tracing
        task_recorder = OrchestraTaskRecorder(task=input, trace_id=trace_id)
        with trace(workflow_name="orchestra_agent", trace_id=trace_id):
            await self.plan(task_recorder)
            for task in task_recorder.plan.todo:
                await self.work(task_recorder, task)
            result = await self.report(task_recorder)
            task_recorder.set_final_output(result.output)
        return task_recorder

    def run_streamed(self, input: str, trace_id: str = None) -> OrchestraTaskRecorder:
        trace_id = trace_id or AgentsUtils.gen_trace_id()
        logger.info(f"> trace_id: {trace_id}")

        with trace(workflow_name="orchestra_agent", trace_id=trace_id):
            task_recorder = OrchestraTaskRecorder(task=input, trace_id=trace_id)
            # Kick off the actual agent loop in the background and return the streamed result object.
            task_recorder._run_impl_task = asyncio.create_task(self._start_streaming(task_recorder))
        return task_recorder

    async def _start_streaming(self, task_recorder: OrchestraTaskRecorder):
        task_recorder._event_queue.put_nowait(AgentUpdatedStreamEvent(new_agent=self.planner_agent))
        plan = await self.plan(task_recorder)
        task_recorder._event_queue.put_nowait(OrchestraStreamEvent(name="plan", item=plan))
        for task in task_recorder.plan.todo:
            # print(f"> processing {task}")
            # DONOT send this event because Runner will send it
            # task_recorder._event_queue.put_nowait(
            #     AgentUpdatedStreamEvent(new_agent=self.worker_agents[task.agent_name])
            # )
            worker_agent = self.worker_agents[task.agent_name]
            result_streaming = worker_agent.work_streamed(task_recorder, task)
            async for event in result_streaming.stream.stream_events():
                task_recorder._event_queue.put_nowait(event)
            result_streaming.output = result_streaming.stream.final_output
            result_streaming.trajectory = AgentsUtils.get_trajectory_from_agent_result(result_streaming.stream)
            task_recorder.add_worker_result(result_streaming)
            # print(f"< processed {task}")
        task_recorder._event_queue.put_nowait(AgentUpdatedStreamEvent(new_agent=self.reporter_agent))
        result = await self.report(task_recorder)
        task_recorder.set_final_output(result.output)
        task_recorder._event_queue.put_nowait(OrchestraStreamEvent(name="report", item=result))
        task_recorder._event_queue.put_nowait(QueueCompleteSentinel())
        task_recorder._is_complete = True

    async def plan(self, task_recorder: OrchestraTaskRecorder) -> CreatePlanResult:
        """Step1: Plan"""
        with function_span("planner") as span_planner:
            plan = await self.planner_agent.create_plan(task_recorder)
            assert all(t.agent_name in self.worker_agents for t in plan.todo), (
                f"agent_name in plan.todo must be in worker_agents, get {plan.todo}"
            )
            task_recorder.set_plan(plan)
            span_planner.span_data.input = json.dumps({"input": task_recorder.task}, ensure_ascii=False)
            span_planner.span_data.output = plan.to_dict()
        return plan

    async def work(self, task_recorder: OrchestraTaskRecorder, task: Subtask) -> WorkerResult:
        """Step2: Work"""
        worker_agent = self.worker_agents[task.agent_name]
        result = await worker_agent.work(task_recorder, task)
        task_recorder.add_worker_result(result)
        return result

    async def report(self, task_recorder: OrchestraTaskRecorder) -> AnalysisResult:
        """Step3: Report"""
        with function_span("reporter") as span_fn:
            analysis_result = await self.reporter_agent.report(task_recorder)
            task_recorder.add_reporter_result(analysis_result)
            span_fn.span_data.input = json.dumps(
                {
                    "input": task_recorder.task,
                    "task_records": [{"task": r.task, "output": r.output} for r in task_recorder.task_records],
                },
                ensure_ascii=False,
            )
            span_fn.span_data.output = analysis_result.to_dict()
        return analysis_result

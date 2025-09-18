from ...config import AgentConfig
from ...utils import AgentsUtils
from ..simple_agent import SimpleAgent
from .common import OrchestraTaskRecorder, Subtask, WorkerResult


class BaseWorkerAgent:
    async def build(self):
        pass

    async def work(self, task_recorder: OrchestraTaskRecorder, subtask: Subtask) -> WorkerResult:
        raise NotImplementedError

    def work_streamed(self, task_recorder: OrchestraTaskRecorder, subtask: Subtask) -> WorkerResult:
        raise NotImplementedError


TEMPLATE = r"""Original Problem:
{problem}

Plan:
{plan}

Previous Trajectory:
{trajectory}

Current Task:
{task}
""".strip()


class SimpleWorkerAgent(BaseWorkerAgent):
    def __init__(self, config: AgentConfig):
        self.agent = SimpleAgent(config=config)

    async def build(self):
        await self.agent.build()

    def _format_task(self, task_recorder: OrchestraTaskRecorder, subtask: Subtask) -> str:
        str_plan = task_recorder.get_plan_str()
        str_traj = task_recorder.get_trajectory_str()
        return TEMPLATE.format(
            problem=task_recorder.task,
            plan=str_plan,
            trajectory=str_traj,
            task=subtask.task,
        )

    async def work(self, task_recorder: OrchestraTaskRecorder, subtask: Subtask) -> WorkerResult:
        """search webpages for a specific subtask, return a report"""
        aug_task = self._format_task(task_recorder, subtask)
        recorder = await self.agent.run(aug_task, trace_id=task_recorder.trace_id)
        return WorkerResult(
            task=subtask.task,
            output=recorder.final_output,
            trajectory=AgentsUtils.get_trajectory_from_agent_result(recorder.get_run_result()),
        )

    def work_streamed(self, task_recorder: OrchestraTaskRecorder, subtask: Subtask) -> WorkerResult:
        aug_task = self._format_task(task_recorder, subtask)
        run_result_streaming = self.agent.run_streamed(aug_task, trace_id=task_recorder.trace_id)
        result = WorkerResult(
            task=subtask.task,
            output="",
            trajectory=[],
            stream=run_result_streaming,
        )
        return result

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from uuid import uuid4

from pydantic_ai import RunContext

from agent.research_steps import gather_sources, read_sources, synthesize_report, read_pdf
from comms.nats.client import build_nats_client
from constants import AGENT_RESPONSE_SUBJECT
from models import AgentResponseEvent, TelegramMessageEvent
from storage.service import StorageService

# The participant ID used by the agent system, typically 0
AGENT_PARTICIPANT_ID = 0
DEEP_RESEARCH_OUTPUT_DIR = Path("deep_research_output")
FEEDBACK_PLACEHOLDER = "<!-- Research task feedback and refinements go here -->\n"

logger = logging.getLogger(__name__)


def _workspace_dir(task_id: int) -> Path:
    return DEEP_RESEARCH_OUTPUT_DIR / f"task_{task_id}"


def _sync_research_workspace(task, event: TelegramMessageEvent | None = None) -> Path:
    task_dir = _workspace_dir(task.id)
    task_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = task_dir / "metadata.json"
    metadata: dict[str, object] = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}

    metadata.update(
        {
            "task_id": task.id,
            "topic": task.topic,
            "status": task.status,
            "step": task.step,
            "created_at": task.created_at.isoformat(),
            "updated_at": task.updated_at.isoformat(),
        }
    )
    if event is not None:
        metadata["request_event"] = event.model_dump(mode="json")

    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    questions = task.specific_questions.split("\n") if task.specific_questions else []
    questions_body = "\n".join(f"- {question}" for question in questions if question.strip()) or "- None provided\n"
    (task_dir / "specific_questions.md").write_text(questions_body, encoding="utf-8")

    feedback_body = task.feedback.strip() if task.feedback else FEEDBACK_PLACEHOLDER.strip()
    (task_dir / "feedback.md").write_text(feedback_body + "\n", encoding="utf-8")

    if task.report:
        (task_dir / "report.md").write_text(task.report, encoding="utf-8")
        if task.report.startswith("Sources:\n"):
            sources = [line for line in task.report.splitlines()[1:] if line.strip()]
            (task_dir / "sources.txt").write_text("\n".join(sources) + ("\n" if sources else ""), encoding="utf-8")

    if task.sources_content:
        (task_dir / "sources_content.md").write_text(task.sources_content, encoding="utf-8")

    return task_dir

async def send_proactive_update(event: TelegramMessageEvent, message: str):
    """Send a proactive update to the user."""
    try:
        nats_client = build_nats_client()
        await nats_client.connect()

        response_event = AgentResponseEvent(
            event_id=f"agent|{uuid4()}",
            source="agent",
            request_event_id=event.event_id,
            channel_id=event.channel_id,
            sender_id=AGENT_PARTICIPANT_ID,
            reply_to_message_id=event.message_id,
            response=message,
        )

        await nats_client.publish_model(AGENT_RESPONSE_SUBJECT, response_event)
    finally:
        await nats_client.close()

async def research_task_runner(task_id: int, event: TelegramMessageEvent) -> None:
    """The main runner for a research task. It will execute the research steps sequentially."""
    storage_service = StorageService()
    await storage_service.start()
    
    try:
        task = storage_service.research_tasks.get_by_id(task_id)
        if not task:
            await storage_service.close()
            return

        _sync_research_workspace(task, event)

        specific_questions = task.specific_questions.split("\n") if task.specific_questions else []

        if task.step == "gather_sources":
            storage_service.research_tasks.update_status(task_id=task_id, status="in_progress")
            await send_proactive_update(event, f"Gathering sources for your research on '{task.topic}'...")
            sources = await gather_sources(task.topic, specific_questions)
            report = "Sources:\n" + "\n".join(sources)
            storage_service.research_tasks.update_report(task_id=task_id, report=report)
            storage_service.research_tasks.update_step(task_id=task_id, step="read_sources")
            storage_service.research_tasks.update_status(task_id=task_id, status="paused")
            updated_task = storage_service.research_tasks.get_by_id(task_id)
            if updated_task:
                _sync_research_workspace(updated_task, event)
                task = updated_task
            
            # Notify user
            report = f"I have gathered the following sources for your research on '{task.topic}':\n" + "\n".join(sources)
            report += "\n\nProvide feedback or tell me to continue."
            await send_proactive_update(event, report)

        elif task.step == "read_sources":
            storage_service.research_tasks.update_status(task_id=task_id, status="in_progress")
            await send_proactive_update(event, f"Reading sources for your research on '{task.topic}'...")
            sources = [url.strip() for url in task.report.split("\n")[1:] if url.strip()] # Get sources from the report safely
            sources_content = await read_sources(sources)
            storage_service.research_tasks.update_sources_content(task_id=task_id, sources_content=sources_content)
            storage_service.research_tasks.update_step(task_id=task_id, step="synthesize_report")
            storage_service.research_tasks.update_status(task_id=task_id, status="paused")
            updated_task = storage_service.research_tasks.get_by_id(task_id)
            if updated_task:
                _sync_research_workspace(updated_task, event)
                task = updated_task

            # Notify user
            report = f"I have read the sources for your research on '{task.topic}'. Provide feedback or tell me to continue to the final report."
            await send_proactive_update(event, report)

        elif task.step == "synthesize_report":
            storage_service.research_tasks.update_status(task_id=task_id, status="in_progress")
            await send_proactive_update(event, f"Synthesizing the report for your research on '{task.topic}'...")
            report = await synthesize_report(task.topic, specific_questions, task.sources_content, task.feedback)
            storage_service.research_tasks.update_report(task_id=task_id, report=report)
            storage_service.research_tasks.update_status(task_id=task_id, status="completed")
            storage_service.research_tasks.update_step(task_id=task_id, step="completed")
            updated_task = storage_service.research_tasks.get_by_id(task_id)
            if updated_task:
                _sync_research_workspace(updated_task, event)
                task = updated_task
            await send_proactive_update(event, report)

    except Exception as e:
        error_message = f"An error occurred while running the research task: {e}"
        storage_service.research_tasks.update_report(task_id=task_id, report=error_message)
        storage_service.research_tasks.update_status(task_id=task_id, status="failed")
        failed_task = storage_service.research_tasks.get_by_id(task_id)
        if failed_task:
            _sync_research_workspace(failed_task, event)
        await send_proactive_update(event, error_message)
    finally:
        await storage_service.close()


async def start_deep_research_task(ctx: RunContext[TelegramMessageEvent], topic: str, specific_questions: list[str]) -> str:
    """Start an asynchronous deep research task on a topic. Use this when the user requests in-depth research or a detailed report.
    This tool runs in the background and returns immediately.
    
    Args:
        topic: The main topic to research.
        specific_questions: Specific questions or areas of focus.
    """
    event = ctx.deps
    
    storage_service = StorageService()
    await storage_service.start()
    task = storage_service.research_tasks.create(
        topic=topic,
        specific_questions="\n".join(specific_questions)
    )
    storage_service.research_tasks.update_step(task_id=task.id, step="gather_sources")
    task = storage_service.research_tasks.get_by_id(task.id) or task
    _sync_research_workspace(task, event)
    await storage_service.close()
    
    asyncio.create_task(research_task_runner(task.id, event))
    return f"I have started deep research on '{topic}'. The task ID is {task.id}. I will keep working in the background and send you a detailed report once I've compiled my findings."

async def continue_research_task(ctx: RunContext[TelegramMessageEvent], task_id: int) -> str:
    """Continue a deep research task from the last step.
    
    Args:
        task_id: The ID of the research task.
    """
    event = ctx.deps
    
    storage_service = StorageService()
    await storage_service.start()
    task = storage_service.research_tasks.get_by_id(task_id)
    await storage_service.close()

    if task is None:
        return f"Research task with ID {task_id} not found."
        
    if task.status == "completed":
        return f"Research task {task_id} on '{task.topic}' is already completed."

    asyncio.create_task(research_task_runner(task.id, event))
    
    response = f"Continuing research on '{task.topic}'.\n\n"
    response += f"Next step: {task.step}.\n"
    if task.feedback:
        response += f"Applying feedback: {task.feedback}\n"
    
    response += "\nI will now proceed with the research. I will let you know when the next step is complete."
    
    return response

async def get_research_task_status(task_id: int) -> str:
    """Get the status of a deep research task.
    
    Args:
        task_id: The ID of the research task.
    """
    storage_service = StorageService()
    await storage_service.start()
    task = storage_service.research_tasks.get_by_id(task_id)
    await storage_service.close()
    
    if task is None:
        return f"Research task with ID {task_id} not found."
    
    if task.status == "completed":
        return f"Research task {task_id} on '{task.topic}' is completed.\n\nReport:\n{task.report}"
    elif task.status == "failed":
        return f"Research task {task_id} on '{task.topic}' failed.\n\nError:\n{task.report}"
    else:
        return f"Research task {task_id} on '{task.topic}' is currently {task.status} at step {task.step}."

async def provide_feedback_to_research_task(task_id: int, feedback: str) -> str:
    """Provide feedback to a deep research task.
    
    Args:
        task_id: The ID of the research task.
        feedback: The feedback to provide to the research task.
    """
    storage_service = StorageService()
    await storage_service.start()
    task = storage_service.research_tasks.get_by_id(task_id)
    
    if task is None:
        await storage_service.close()
        return f"Research task with ID {task_id} not found."
    
    storage_service.research_tasks.update_feedback(task_id=task_id, feedback=feedback)
    updated_task = storage_service.research_tasks.get_by_id(task_id)
    await storage_service.close()

    if updated_task is not None:
        _sync_research_workspace(updated_task)
    
    return f"Feedback has been provided to research task {task_id}. The research will be refined based on your feedback."

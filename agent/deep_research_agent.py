from __future__ import annotations

import asyncio
from uuid import uuid4

from pydantic_ai import RunContext

from agent.research_steps import gather_sources, read_sources, synthesize_report, read_pdf
from comms.nats.client import build_nats_client
from constants import AGENT_RESPONSE_SUBJECT
from models import AgentResponseEvent, TelegramMessageEvent
from storage.service import StorageService

# The participant ID used by the agent system, typically 0
AGENT_PARTICIPANT_ID = 0

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

        specific_questions = task.specific_questions.split("\n") if task.specific_questions else []

        if task.step == "gather_sources":
            storage_service.research_tasks.update_status(task_id=task_id, status="in_progress")
            await send_proactive_update(event, f"Gathering sources for your research on '{task.topic}'...")
            sources = await gather_sources(task.topic, specific_questions)
            report = "Sources:\n" + "\n".join(sources)
            storage_service.research_tasks.update_report(task_id=task_id, report=report)
            storage_service.research_tasks.update_step(task_id=task_id, step="read_sources")
            storage_service.research_tasks.update_status(task_id=task_id, status="paused")
            
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
            await send_proactive_update(event, report)

    except Exception as e:
        storage_service.research_tasks.update_status(task_id=task_id, status="failed")
        await send_proactive_update(event, f"An error occurred while running the research task: {e}")
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
    await storage_service.close()
    
    return f"Feedback has been provided to research task {task_id}. The research will be refined based on your feedback."

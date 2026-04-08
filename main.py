import asyncio
import logging
import warnings
import logfire

warnings.filterwarnings("ignore", category=DeprecationWarning, module="nats.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="google.genai.*")

from agent.runner import AgentReceiverRunner
from comms.base_runner import BaseRunner
from comms.telegram.runner import TelegramRunner
from config.env_config import init_config, get_env
from agent.continuous_research import continuous_research_loop
from agent.compressor import compression_loop
from storage import StorageService

def setup_logging() -> None:
    init_config()
    
    # Read LOG_LEVEL from environment, default to INFO
    log_level_str = get_env("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.WARNING)

    logfire_level = log_level_str.lower()
    if logfire_level == "warning":
        logfire_level = "warn"
    elif logfire_level not in ["trace", "debug", "info", "warn", "error", "fatal"]:
        logfire_level = "info"
        
    logfire.configure(
        send_to_logfire='if-token-present',
        console=logfire.ConsoleOptions(min_log_level=logfire_level)
    )


async def run() -> None:
    setup_logging()

    storage_service = StorageService()
    await storage_service.start()

    runners: list[BaseRunner] = [
        AgentReceiverRunner(storage_service=storage_service),
        TelegramRunner(storage_service=storage_service),
    ]
    
    loop_task = asyncio.create_task(continuous_research_loop())
    compress_task = asyncio.create_task(compression_loop(storage_service))

    try:
        for runner in runners:
            await runner.start()

        await asyncio.Event().wait()
    finally:
        loop_task.cancel()
        compress_task.cancel()
        for runner in reversed(runners):
            await runner.stop()
        await storage_service.close()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()

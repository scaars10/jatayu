import asyncio

from agent.runner import AgentReceiverRunner
from comms.base_runner import BaseRunner
from comms.telegram.runner import TelegramRunner
from config.env_config import init_config


async def run() -> None:
    init_config()

    runners: list[BaseRunner] = [
        AgentReceiverRunner(),
        TelegramRunner(),
    ]

    try:
        for runner in runners:
            await runner.start()

        await asyncio.Event().wait()
    finally:
        for runner in reversed(runners):
            await runner.stop()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()

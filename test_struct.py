import asyncio
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

class Out(BaseModel):
    ans: str

async def main():
    agent = Agent(TestModel(), output_type=Out)
    res = await agent.run('hello')
    if hasattr(res, 'data'): print("data:", res.data)
    if hasattr(res, 'output'): print("output:", res.output)
    if hasattr(res, 'response'): print("response:", res.response)

asyncio.run(main())
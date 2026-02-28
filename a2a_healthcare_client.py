"""
Healthcare Client (The "User")
------------------------------
This file represents YOU (the end-user). It's a simple script that 
pretends to be a chat interface, submitting a question directly to the 
Head Doctor (Healthcare Agent - Orchestrator) and printing the response.
"""

import asyncio
import os
from typing import Any

from beeai_framework.adapters.a2a.agents import A2AAgent
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.middleware.trajectory import EventMeta, GlobalTrajectoryMiddleware

from helpers import setup_env


# This piece of code just tells the system to hide the messy "behind the scenes"
# logs and only clearly print the final information that happens during the run.
class ConciseGlobalTrajectoryMiddleware(GlobalTrajectoryMiddleware):
    def _format_prefix(self, meta: EventMeta) -> str:
        prefix = super()._format_prefix(meta)
        return prefix.rstrip(": ")

    def _format_payload(self, value: Any) -> str:
        return ""


async def main() -> None:
    setup_env()

    host = os.getenv("AGENT_HOST")
    healthcare_agent_port = int(os.getenv("HEALTHCARE_AGENT_PORT")) # This points to the Orchestrator

    # 1. Connect to the Orchestrator
    # We define the A2AAgent we want to talk to (our Head Doctor on port 9996).
    agent = A2AAgent(
        url=f"http://{host}:{healthcare_agent_port}", memory=UnconstrainedMemory()
    )
    
    # 2. Ask the Question
    # We use agent.run() to send a text prompt over the network.
    # Notice this question requires BOTH the Provider Agent (for location) and Policy Agent (for insurance).
    response = await agent.run(
        "I'm based in Austin, TX. How do I get mental health therapy near me and what does my insurance cover?"
    ).middleware(ConciseGlobalTrajectoryMiddleware())
    
    # 3. Print the Final Answer
    print(response.last_message.text)


if __name__ == "__main__":
    asyncio.run(main())

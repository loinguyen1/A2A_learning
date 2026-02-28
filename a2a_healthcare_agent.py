"""
Healthcare Agent (The Orchestrator / "Head Doctor")
---------------------------------------------------
This acts as the "boss" agent. It interacts with the user, but it doesn't 
actually know the answers itself. Instead, it figures out which sub-agents 
have the right skills to answer the question, delegates the work to them, 
and then summarizes everything for the user.

It uses the BeeAI Framework to manage this routing logic.
"""

import asyncio
import os
from typing import Any

from beeai_framework.adapters.a2a.agents import A2AAgent
from beeai_framework.adapters.a2a.serve.server import A2AServer, A2AServerConfig
from beeai_framework.adapters.gemini import GeminiChatModel
from beeai_framework.adapters.vertexai import VertexAIChatModel  # noqa: F401
# The RequirementAgent is a type of agent that enforces rules (requirements) before answering.
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.agents.requirement.requirements.conditional import (
    ConditionalRequirement,
)
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.middleware.trajectory import EventMeta, GlobalTrajectoryMiddleware
from beeai_framework.serve.utils import LRUMemoryManager
from beeai_framework.tools import Tool
# HandoffTool is what allows an agent to pass a question to ANOTHER agent
from beeai_framework.tools.handoff import HandoffTool
# ThinkTool allows the agent to privately reason about its plan before acting
from beeai_framework.tools.think import ThinkTool

from helpers import setup_env


# Log only tool calls (hides messy internal thinking from the terminal)
class ConciseGlobalTrajectoryMiddleware(GlobalTrajectoryMiddleware):
    def _format_prefix(self, meta: EventMeta) -> str:
        prefix = super()._format_prefix(meta)
        return prefix.rstrip(": ")

    def _format_payload(self, value: Any) -> str:
        return ""


def main() -> None:
    print("Running A2A Orchestrator Agent")
    setup_env()

    host = os.getenv("AGENT_HOST")
    policy_agent_port = os.getenv("POLICY_AGENT_PORT")
    research_agent_port = os.getenv("RESEARCH_AGENT_PORT")
    provider_agent_port = os.getenv("PROVIDER_AGENT_PORT")
    healthcare_agent_port = int(os.getenv("HEALTHCARE_AGENT_PORT"))

    # Log only tool calls
    GlobalTrajectoryMiddleware(target=[Tool])

    # 1. CONNECT TO THE EMPLOYEES (Sub-Agents)
    # The orchestrator reads the "Agent Cards" of the 3 sub-agents running on different ports.
    # If any of these aren't running, this startup will crash!
    policy_agent = A2AAgent(
        url=f"http://{host}:{policy_agent_port}", memory=UnconstrainedMemory()
    )
    # Run `check_agent_exists()` to fetch and populate AgentCard
    asyncio.run(policy_agent.check_agent_exists())
    print("\tℹ️", f"{policy_agent.name} initialized")

    research_agent = A2AAgent(
        url=f"http://{host}:{research_agent_port}", memory=UnconstrainedMemory()
    )
    asyncio.run(research_agent.check_agent_exists())
    print("\tℹ️", f"{research_agent.name} initialized")

    provider_agent = A2AAgent(
        url=f"http://{host}:{provider_agent_port}", memory=UnconstrainedMemory()
    )
    asyncio.run(provider_agent.check_agent_exists())
    print("\tℹ️", f"{provider_agent.name} initialized")


    # 2. CREATE THE ORCHESTRATOR
    healthcare_agent = RequirementAgent(
        name="Healthcare Agent",
        description="A personal concierge for Healthcare Information, customized to your policy.",
        llm=GeminiChatModel(
            "gemini-3-flash-preview",
            allow_parallel_tool_calls=True,
        ),
        # If using Vertex AI
        # llm = VertexAIChatModel(
        #    model_id="gemini-3-flash-preview",
        #    project= os.environ.get("GOOGLE_CLOUD_PROJECT"),
        #    location="global",
        #    allow_parallel_tool_calls=True,
        # ),
        
        # 3. DEFINE THE TOOLS
        # Tools here are actually just "Handoffs" to the sub-agents.
        tools=[
            thinktool := ThinkTool(),
            policy_tool := HandoffTool(
                target=policy_agent,
                name=policy_agent.name,
                description=policy_agent.agent_card.description,
            ),
            research_tool := HandoffTool(
                target=research_agent,
                name=research_agent.name,
                description=research_agent.agent_card.description,
            ),
            provider_tool := HandoffTool(
                target=provider_agent,
                name=provider_agent.name,
                description=provider_agent.agent_card.description,
            ),
        ],
        
        # 4. DEFINE THE RULES (Requirements)
        # This forces the agent to use its 'ThinkTool' first to plan its strategy.
        # It also prevents it from looping/calling the same sub-agent over and over.
        requirements=[
            ConditionalRequirement(
                thinktool,
                force_at_step=1,
                force_after=Tool,
                consecutive_allowed=False,
            ),
            ConditionalRequirement(
                policy_tool,
                consecutive_allowed=False,
                max_invocations=1,
            ),
            ConditionalRequirement(
                research_tool,
                consecutive_allowed=False,
                max_invocations=1,
            ),
            ConditionalRequirement(
                provider_tool,
                consecutive_allowed=False,
                max_invocations=1,
            ),
        ],
        role="Healthcare Concierge",
        
        # 5. BUSINESS RULES
        # Strict instructions on how the agent should compile the final answer.
        instructions=(
            f"""You are a concierge for healthcare services. Your task is to handoff to one or more agents to answer questions and provide a detailed summary of their answers. Be sure that all of their questions are answered before responding.

            IMPORTANT: When returning answers about providers, only output providers from `{provider_agent.name}` and only provide insurance information based on the results from `{policy_agent.name}`.

            In your output, put which agent gave you the information!"""
        ),
    )

    print("\tℹ️", f"{healthcare_agent.meta.name} initialized")

    # 6. START THE SERVER
    # Register the agent with the A2A server and run the HTTP server
    # we use LRU memory manager to keep limited amount of sessions in the memory
    A2AServer(
        config=A2AServerConfig(
            port=healthcare_agent_port, protocol="jsonrpc", host=host
        ),
        memory_manager=LRUMemoryManager(maxsize=100),
    ).register(healthcare_agent, send_trajectory=True).serve()


if __name__ == "__main__":
    main()

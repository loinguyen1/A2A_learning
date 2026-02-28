"""
Provider Agent
--------------
This agent focuses on finding doctors and clinics.
It demonstrates another powerful concept: Building an agent with LangChain 
and connecting it to an external "Tool Server" (MCP Server) for data.

Why do it this way?
This shows that A2A works with LangChain. It also shows how an AI 
Agent can securely query a database (using the MCP protocol) rather 
than having the database inside the agent's code directly.
"""

import asyncio
import os

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
# LangChain is a very popular framework for building AI chains/agents
from langchain.agents import create_agent
from langchain_litellm import ChatLiteLLM
# MCP adapters allow LangChain to talk to our mcpserver.py Tool Server
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import StdioConnection
# LangGraph is the underlying engine for LangChain agents
from langgraph.graph.state import CompiledStateGraph
# The wrapper to make LangGraph compatible with A2A networks
from langgraph_a2a_server import A2AServer

from helpers import setup_env


def main() -> None:
    print("Running Healthcare Provider Agent")
    setup_env()

    HOST = os.getenv("AGENT_HOST", "localhost")
    PORT = int(os.getenv("PROVIDER_AGENT_PORT"))

    # 1. Connect to the MCP Server
    # Here, our agent acts as an MCP "Client". It runs "uv run mcpserver.py" in the background
    # and says "Hey mcpserver, what tools do you have that I can use?"
    mcp_client = MultiServerMCPClient(
        {
            "find_healthcare_providers": StdioConnection(
                transport="stdio",
                command="uv",
                args=["run", "mcpserver.py"],
            )
        }
    )
    
    # 2. Create the LangChain Agent
    # We pass it the tools it requested from the MCP client.
    agent: CompiledStateGraph = create_agent(
        model=ChatLiteLLM(
            model="gemini/gemini-3-flash-preview",
            # For Vertex AI:
            # model="vertex_ai/gemini-3-flash-preview",
            max_tokens=1000,
        ),
        # Notice how we dynamically load the tools from the MCP server here!
        tools=asyncio.run(mcp_client.get_tools()),
        name="HealthcareProviderAgent",
        system_prompt="Find and list healthcare providers using the find_healthcare_providers MCP Tool.",
    )

    # 3. Create the A2A Agent Card
    # This advertises our agent to the A2A network so the Orchestrator knows what it does.
    agent_card = AgentCard(
        name="HealthcareProviderAgent",
        description="Find healthcare providers by location and specialty.",
        url=f"http://{HOST}:{PORT}/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[
            AgentSkill(
                id="find_healthcare_providers",
                name="Find Healthcare Providers",
                description="Finds providers based on location/specialty.",
                tags=["healthcare", "providers", "doctor", "psychiatrist"],
                examples=[
                    "Are there any Psychiatrists near me in Boston, MA?",
                    "Find a pediatrician in Springfield, IL.",
                ],
            )
        ],
    )

    # 4. Wrap with the A2A Server
    # Now our LangGraph agent is available on port 9997 for others to talk to.
    server = A2AServer(
        graph=agent,
        agent_card=agent_card,
        host=HOST,
        port=PORT,
    )

    server.serve(app_type="starlette")


if __name__ == "__main__":
    main()

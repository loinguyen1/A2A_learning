"""
Research Agent
--------------
This agent focuses on searching the web for medical information (like symptoms or treatments).
It is built using Google's ADK (Agent Development Kit) instead of the bare A2A types.

Why do it this way? To show that A2A works with different agent-building frameworks!
Here we just build an standard ADK agent, then use `to_a2a` to magically wrap it
so it can join the A2A network and talk to the Orchestrator.
"""

import os
import uvicorn

# The ADK utility that converts a normal ADK agent into an A2A-compatible web server
from google.adk.a2a.utils.agent_to_a2a import to_a2a
# ADK's standard LLM agent class
from google.adk.agents.llm_agent import LlmAgent
# A built-in tool provided by ADK to search the web
from google.adk.tools import google_search

from helpers import setup_env

setup_env()

PORT = int(os.getenv("RESEARCH_AGENT_PORT"))
HOST = os.getenv("AGENT_HOST")


def main() -> None:
    # 1. Create the standard ADK Agent.
    # We give it a specific role ("HealthResearchAgent") and the ability to search Google.
    root_agent = LlmAgent(
        model="gemini-3-pro-preview",
        name="HealthResearchAgent",
        tools=[google_search],
        description="Provides healthcare information about symptoms, health conditions, treatments, and procedures using up-to-date web resources.",
        instruction="You are a healthcare research agent tasked with providing information about health conditions. Use the google_search tool to find information on the web about options, symptoms, treatments, and procedures. Cite your sources in your responses. Output all of the information you find.",
    )

    # 2. Make the agent A2A-compatible.
    # The `to_a2a` function wraps the ADK agent giving it an A2A "Agent Card" and server logic.
    a2a_app = to_a2a(root_agent, host=HOST, port=PORT)
    print("Running Health Research Agent")
    
    # 3. Start the server so it can listen for questions from the Healthcare Agent (Orchestrator).
    uvicorn.run(a2a_app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()

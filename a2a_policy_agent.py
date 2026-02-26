"""
A2A (Agent-To-Agent) Health Insurance Policy Agent
--------------------------------------------------
This file defines and runs an Agent using the A2A protocol.
As a Data Analyst, you can think of this as setting up a "service" or "API"
that other agents (or users) can talk to.

Here, we are wrapping a standard Python class (`PolicyAgent`, which contains our core logic/LLM calls)
into a web server that speaks the standard A2A language, allowing it to easily interact with other agents.
"""

import os

# uvicorn is a lightning-fast web server for Python. We use it to run our agent's API.
import uvicorn

# These are core A2A components that handle the server-side logic of an agent.
# Let's break down the role of each:
# 
# 1. AgentExecutor & RequestContext:
#    - AgentExecutor is a class you must inherit from to define how your agent handles incoming messages.
#    - RequestContext holds the details of the incoming request (who sent it, what the message is, etc.).
from a2a.server.agent_execution import AgentExecutor, RequestContext

# 2. A2AStarletteApplication:
#    - Starlette is a fast web framework (similar to Flask or FastAPI).
#    - This specific class essentially builds the "House" for your agent; it packages your
#      agent's logic, its description (AgentCard), and its routing into a web server application.
from a2a.server.apps import A2AStarletteApplication

# 3. EventQueue:
#    - When your agent finishes thinking and generates an answer, it doesn't just "return" it.
#    - It places the answer into an EventQueue. Imagine this as an "Outbox" where messages wait
#      to be delivered back over the network to whoever asked the question.
from a2a.server.events import EventQueue

# 4. DefaultRequestHandler:
#    - This is the "Bouncer" or "Receptionist" at the front door. It receives the raw HTTP web
#      requests, translates them, and decides what to do with them (like passing them to your AgentExecutor).
from a2a.server.request_handlers import DefaultRequestHandler

# 5. InMemoryTaskStore:
#    - When someone asks a question, it might take the AI a few seconds to answer.
#    - This store acts like a "clipboard" keeping track of ongoing conversations/tasks in the server's memory.
from a2a.server.tasks import InMemoryTaskStore

# These types help us define our agent's "business card" so other agents know what it can do.
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

# A utility to create a text message in the format that A2A expects.
from a2a.utils import new_agent_text_message

# These are custom helpers for this specific project (loading environment variables, etc.)
from helpers import setup_env
# This is our core logic class. It likely contains the prompt engineering and LLM calls.
from policy_agent import PolicyAgent


class PolicyAgentExecutor(AgentExecutor):
    """
    The Executor is the bridge between the A2A server and your custom agent logic.
    Whenever this agent receives a request over the network, the A2A server will call
    the `execute` method of this class.
    """
    def __init__(self) -> None:
        # Initialize our custom logic (the brain of the agent)
        self.agent = PolicyAgent()

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """
        This method is triggered when a user or another agent sends a message to this agent.
        """
        # 1. Get the user's question from the request context
        prompt = context.get_user_input()
        
        # 2. Pass the question to our core policy agent to get an answer (e.g., using an LLM to read a policy DB)
        response = self.agent.answer_query(prompt)
        
        # 3. Format the answer into a standard A2A text message
        message = new_agent_text_message(response)
        
        # 4. Put the message into the event queue, which sends it back to the caller
        await event_queue.enqueue_event(message)

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """
        This method handles cancellation requests (e.g., if the user stops the generation).
        We do nothing here right now, but we could stop the LLM processing if needed.
        """
        pass


def main() -> None:
    print("Running A2A Health Insurance Policy Agent")
    # Load environment variables (like API keys)
    setup_env()
    
    # Determine which port and host this agent will listen on.
    PORT = int(os.getenv("POLICY_AGENT_PORT", 9999))
    HOST = os.getenv("AGENT_HOST", "localhost")

    # Step 1: Define what this agent is good at.
    # Other routing agents will look at this description to decide if they should send questions here.
    skill = AgentSkill(
        id="insurance_coverage",
        name="Insurance coverage",
        description="Provides information about insurance coverage options and details.",
        tags=["insurance", "coverage"],
        examples=["What does my policy cover?", "Are mental health services included?"],
    )

    # Step 2: Create an Agent Card. Think of this as the agent's profile in the network.
    # It includes the URL where the agent lives, what kinds of inputs/outputs it supports (text),
    # and its skills.
    agent_card = AgentCard(
        name="InsurancePolicyCoverageAgent",
        description="Provides information about insurance policy coverage options and details.",
        url=f"http://{HOST}:{PORT}/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill],
    )

    # Step 3: Set up the request handler. This manages incoming HTTP requests,
    # tracks their status in memory (task_store), and executes our custom logic (agent_executor).
    request_handler = DefaultRequestHandler(
        agent_executor=PolicyAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    # Step 4: Build the web server application using all our configured parts.
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    # Step 5: Run the web server using Uvicorn! It will now wait for incoming requests.
    uvicorn.run(server.build(), host=HOST, port=PORT)


if __name__ == "__main__":
    main()

"""
MCP (Model Context Protocol) Server
-----------------------------------
This file is NOT an A2A agent. It is a tool/database server.
Think of MCP an adapter that securely exposes a specific set of tools (like querying a doctor database)
to any AI agent that knows how to speak the MCP protocol.

Why use this? 
Instead of giving your AI agent raw database credentials, you build an MCP server.
The AI agent asks the MCP server "Can you run this query for me?", and the server does the work.
This is highly secure and modular.
"""

import json
from pathlib import Path

# FastMCP makes it easy to create a tool server
from mcp.server.fastmcp import FastMCP

# 1. Initialize the MCP server
mcp = FastMCP("doctorserver")

# 2. Load the actual data the tools will interact with (our fake doctor database)
# Adjusted path to match project structure
doctors: list = json.loads(Path("data/doctors.json").read_text())


# 3. Define the Tool
# The `@mcp.tool()` decorator tells the MCP server to expose this Python function to AI agents.
# The AI agent can read the docstring (""""This tool returns..."""") to understand what the tool does.
@mcp.tool()
def list_doctors(state: str | None = None, city: str | None = None) -> list[dict]:
    """This tool returns a list of doctors practicing in a specific location. The search is case-insensitive.

    Args:
        state: The two-letter state code (e.g., "CA" for California).
        city: The name of the city or town (e.g., "Boston").

    Returns:
        A JSON string representing a list of doctors matching the criteria.
        If no criteria are provided, an error message is returned.
        Example: '[{"name": "Dr John James", "specialty": "Cardiology", ...}]'
    """
    # Input validation: ensure at least one search term is given.
    if not state and not city:
        return [{"error": "Please provide a state or a city."}]

    target_state = state.strip().lower() if state else None
    target_city = city.strip().lower() if city else None

    return [
        doc
        for doc in doctors
        if (not target_state or doc["address"]["state"].lower() == target_state)
        and (not target_city or doc["address"]["city"].lower() == target_city)
    ]


# 4. Kick off server if file is run
# "stdio" (Standard Input/Output) means the AI agent will communicate with this server 
# using the terminal (running it as a background process), rather than over a web port.
if __name__ == "__main__":
    mcp.run(transport="stdio")

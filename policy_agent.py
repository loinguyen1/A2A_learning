"""
Core Agent Logic (The "Brains" of the Agent)
--------------------------------------------
As a Data Analyst, you are used to writing SQL or Python to query a structured database.
This file does something similar, but instead of querying a database, it queries an AI 
model (an LLM) using an unstructured document (a PDF file).

While `a2a_policy_agent.py` handles the networking (making this agent available 
to other agents on the internet), this file (`policy_agent.py`) contains the actual 
logic for reading a document and answering questions about it.
"""

import base64
from pathlib import Path

# litellm is a Python library that acts like a universal translator for AI models.
# It lets you write code once, and easily switch between different AI models 
# (like Google's Gemini, OpenAI's GPT-4, or Anthropic's Claude) just by changing one line of code.
import litellm

# A helper function to load API keys and other settings from a .env file
from helpers import setup_env


class PolicyAgent:
    def __init__(self) -> None:
        """
        Think of this __init__ method like your 'data setup' phase before you run an analysis.
        This runs only once when the agent is first started.
        """
        setup_env() # Load API keys
        
        # 1. READ THE FILE: We open the health insurance policy PDF file.
        # "rb" means "read binary". We are reading the raw bytes of the PDF.
        with Path("data/2026AnthemgHIPSBC.pdf").open("rb") as file:
            
            # 2. ENCODE THE FILE: We can't easily send raw files over an internet API request to an AI.
            # So, we convert the raw PDF file into a long string of text using 'base64' encoding.
            # This 'self.pdf_data' variable now holds the entire PDF document in a format the AI can read over the web.
            self.pdf_data = base64.standard_b64encode(file.read()).decode("utf-8")

    def answer_query(self, prompt: str) -> str:
        """
        This is the main function that gets called whenever someone asks the agent a question.
        The 'prompt' variable contains the text of the user's question (e.g., "What is the deductible?").
        """
        
        # 3. ASK THE AI: We make a request to the AI model using litellm.
        response = litellm.completion(
            # We are telling it to use Google's 'Gemini 3 Flash' model.
            # This specific model is "multimodal", meaning it's really good at reading files (like PDFs).
            model="gemini/gemini-3-flash-preview",
            
            # For Vertex AI
            # model="vertex_ai/gemini-3-flash-preview",
            
            # These settings tell the AI not to overthink the answer and to keep it relatively short (under 1000 tokens).
            reasoning_effort="minimal",
            max_tokens=1000,
            
            # 4. THE MESSAGE: The 'messages' list is how we talk to the AI.
            messages=[
                {
                    # A 'system' role is used to give the AI its permanent instructions and personality.
                    # Think of this as the business rules for your data analysis.
                    "role": "system",
                    "content": "You are an expert insurance agent designed to assist with coverage queries. Use the provided documents to answer questions about insurance policies. If the information is not available in the documents, respond with 'I don't know'",
                },
                {
                    # A 'user' role is the actual request we are making right now.
                    "role": "user",
                    "content": [
                        # First, we pass the user's actual question...
                        {"type": "text", "text": prompt},
                        
                        # Second, we pass the PDF document we loaded earlier!
                        # This allows the AI to "read" the PDF to find the answer.
                        {
                            "type": "image_url",
                            "image_url": {
                                # We send the base64-encoded PDF data here.
                                "url": f"data:application/pdf;base64,{self.pdf_data}"
                            },
                        },
                    ],
                },
            ],
        )

        # 5. RETURN THE ANSWER: The AI sends back a large JSON object with a lot of metadata.
        # We dig into that object (response.choices[0].message.content) to extract just the text answer.
        # The '.replace("$", r"\$")' part is a minor formatting fix so dollar signs show up correctly in chat interfaces.
        return response.choices[0].message.content.replace("$", r"\$")

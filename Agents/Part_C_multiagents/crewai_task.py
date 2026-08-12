from crewai import Agent, Task, Crew, Process, LLM
import os
from dotenv import load_dotenv

load_dotenv()

print("=== CrewAI Roles + Tasks ===")

llm = LLM(
    model="openai/llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    temperature=0.3
)

researcher = Agent(
    role='Senior research analyst',
    goal="""Research topics deeply and comprehensive,
    accurate information with sources."""
    backstory="""You are a senior research analyst
    with 10 years experience.
    You specialized in technology and AI topics.
    """
)
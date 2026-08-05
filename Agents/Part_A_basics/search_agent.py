from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

print("=== Search Agent ===")

search_tool = TavilySearchResults(
    max_results = 3,
    api_key=os.getenv("TAVILY_API_KEY")
)

print("Direct Search Test :")
results = search_tool.invoke("Langchain AI framework 2024")
for r in results:
    print(f"→ {r['url']}")
    print(f"  {r['content'][:1000]}")
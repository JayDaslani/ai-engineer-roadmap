from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
import os
from typing import TypedDict, List
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model='llama-3.3-70b-versatile',
    api_key=os.getenv('GROQ_API_KEY'),
    temperature=0.2
)

search_tool = DuckDuckGoSearchRun()

class ResearchState(TypedDict):
    topic: str
    category: str
    research: str
    summary: str
    key_points: List[str]
    report: str

def classify_topic(state: ResearchState):
    print("Classifying topic ...")
    prompt = f"""
    Classify this research topic into ONE category (e.g., Tech, Science, Finance, Health, History): {state['topic']}
    Return Only the category name.
    """
    response = llm.invoke(prompt)
    category = response.content.strip()
    print(f" Category : {'category'}")
    return {'category': category}

def research_topic(state: ResearchState):
    print("Searching web for research data ...")
    query = f"""{state['topic']} {state['category']} key fact overview"""

    try:
        data= search_tool.run(query)
    except:
        response = llm.invoke(f"Provide detailed research fact about : {state['topic']}")
        data = response.content

    return {"research": data}

def summarize(state: ResearchState):
    print(f" Summarizing research ...")
    prompt = f"""
    Summarizing the following research data in 2 concise paragraph : {state['research']}"""
    response = llm.invoke(prompt)
    return {"summary": response.content}

def extract_key_points(state: ResearchState):
    print("Extracting key points ...")
    prompt = f"""
    Extract 4-5 bullet key points from this summary :\n {state['summary']}
    \n\nReturn each key point on a new line starting with '- '
    """
    response = llm.invoke(prompt)

    points = [p.strip('- ').strip() for p in response.content.split('\n') if p.strip()]
    return {"key_points": points}

def generate_report(state: ResearchState):
    print("Compiling final report ...")
    points_formatted = "\n".join([f"* {p}" for p in state['key_points']])
    
    report = f"""
==================================================
              RESEARCH REPORT
==================================================
TOPIC    : {state['topic']}
CATEGORY : {state['category']}
==================================================

SUMMARY:
{state['summary']}

KEY POINTS:
{points_formatted}

==================================================
*Report generated via LangGraph Research Agent*
"""
    return {"report": report}

graph = StateGraph(ResearchState)

graph.add_node('classify', classify_topic)
graph.add_node('research', research_topic)
graph.add_node('summary', summarize)
graph.add_node('key_points', extract_key_points)
graph.add_node('report', generate_report)

graph.set_entry_point('classify')
graph.add_edge('classify', 'research')
graph.add_edge('research', 'summary')
graph.add_edge('summary', 'key_points')
graph.add_edge('key_points', 'report')
graph.add_edge('report', END)

app = graph.compile()

print("=== Smart Research Assistant ===")
user_topic = input('Enter research topic: ').strip() or "Artificial Intelligence Agents 2026"

final_state = app.invoke({
    "topic": user_topic,
    "category": "",
    "research": "",
    "summary": "",
    "key_points": [],
    "report": ""
})

print(final_state['report'])



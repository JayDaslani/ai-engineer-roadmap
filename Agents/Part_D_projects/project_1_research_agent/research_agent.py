import os
import re
from datetime import datetime
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()

llm = ChatGroq(
    model='llama-3.3-70b-versatile',
    api_key=os.getenv('GROQ_API_KEY'),
    temperature=0.3
)

search_tool = DuckDuckGoSearchRun()

class ResearchState(TypedDict):
    topic: str
    search_results: str
    summary: str
    report: str

def save_report_to_file(topic: str, report_content: str) -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    reports_dir = os.path.join(base_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    clean_topic = re.sub(r'[^\w\s-]', '', topic).strip().lower().replace(' ', '_')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{clean_topic}_{timestamp}.md"
    filepath = os.path.join(reports_dir, filename)

    with open(filepath, "w", encoding='utf-8') as f:
        f.write(report_content)

    return filepath

def search_node(state: ResearchState):
    print(f"Web search starting for: '{state['topic']}'...")
    query = f"{state['topic']} key facts overview trends"

    try:
        results = search_tool.run(query)
    except Exception as e:
        print(f"⚠️ Search failed, triggering LLM fallback: {e}")
        fallback_prompt = f"Provide detailed facts and technical overview regarding : {state['topic']}"
        response = llm.invoke(fallback_prompt)
        results = response.content

    return {'search_results': results}

def summary_node(state: ResearchState):
    print("📝 Summarizing search data...")
    prompt=f"""
    You are an expert research analyst.
    Summarize the following web research data for the topic : '{state['topic']}'
    
    SEARCH DATA : {state['search_results']}
    
    Provide a concise, 2-paragraph summary highlighting core concepts, facts and current state.
    """

    response = llm.invoke(prompt)
    return {"summary": response.content}

def report_node(state: ResearchState):
    print("Generating structural final report & saving to file ...")
    prompt = f"""
    You are a Senior Techincal Editor.
    Create a clean, well-structured, professional Markdown Research Report based on the details below: 
    
    TOPIC : {state['topic']}
    SUMMARY : {state['summary']}
    
    Structure the Markdown output exactly with these sections :
    # Research Report: {state['topic']}
    
    ## 1. Executive Summary
    (Write a concise overview)
    
    ## 2. Key Findings & Insights
    (Provide 4-5 key bullet points with bold headers)
    
    ## 3. Current Market / Tech Trends
    (Detail key trends)
    
    ## 4. Conclusion & Next Steps
    (Wrap up insights)
    """

    response = llm.invoke(prompt)
    report_text = response.content.strip()

    saved_path = save_report_to_file(state['topic'], report_text)
    print(f"💾 Report saved successfully at: {saved_path}")

    return {"report": report_text}


builder = StateGraph(ResearchState)

builder.add_node("search", search_node)
builder.add_node('summary', summary_node)
builder.add_node('report', report_node)

builder.set_entry_point('search')
builder.add_edge('search', 'summary')
builder.add_edge('summary', 'report')
builder.add_edge('report', END)

app = builder.compile()

def run_research(topic: str) -> dict:
    intial_state = {
        'topic': topic,
        'search_results': "",
        'summary': "",
        "report": ""
    }
    final_state = app.invoke(intial_state)
    return final_state

print("=== Testing Research Agent ===")
test_topic = "AI Agents in Algorithimic Trading"
output = run_research(test_topic)

print("="*50)
print(output['report'])


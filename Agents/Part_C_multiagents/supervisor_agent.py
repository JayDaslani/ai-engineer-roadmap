from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage
)
from typing import TypedDict, List, Literal
import os
from dotenv import load_dotenv

load_dotenv()

print("=== Supervisor Agent System ===")

llm = ChatGroq(
    model='llama-3.3-70b-versatile',
    api_key=os.getenv('GROQ_API_KEY'),
    temperature=0.3
)

class Supervisorstate(TypedDict):
    messages: List
    next_agent: str
    task: str
    results: List[str]
    final_response: str
    iteration: int

# ─────────────────────────────
# WORKER AGENTS
# ─────────────────────────────
    
def research_agent(state: Supervisorstate):
    print(" [Research Agent] Working...")

    prompt = f"""
    You are a Research specialist.
    Research this topic thoroughly : {state['task']}
   
    Provide facts, data, and insights.
    Keep it concise - 3-4 paragraphs.
    """

    response = llm.invoke(prompt)
    result = f"[Research] : {response.content}"

    return {
        "results": state['results'] + [result],
        "messages": state['messages'] + [AIMessage(content=result)]
    }

def writer_agent(state: Supervisorstate):
    print("  [Writer Agent] Working...")

    context = "\n".join(state['results'])
    prompt = f"""
    You are a content writer specialist.
    Write engaging content about: {state['task']}
    
    Use this research if avaliable : {context}
    Write 2-3 clear paragraphs.
    """

    response = llm.invoke(prompt)
    result = f"[Written Content]: {response.content}"

    return {
        "results": state['results'] + [result],
        "messages": state['messages'] + [AIMessage(content=result)]
    }

def analyst_agent(state: Supervisorstate):
    print("  [Analyst Agent] Working...")

    context = "\n".join(state['results'])
    prompt = f"""
    You are a Data Analyst Specialist.
    Analyze and provide insights about : {state['task']}
    
    Available data: {context}
    
    Provide:
    - Key findings
    - Trends 
    - Recommendations
    """

    response = llm.invoke(prompt)
    result = f"[Analysis]: {response.content}"

    return {
        "results": state['results'] + [result],
        "messages": state['messages'] + [AIMessage(content=result)]
    }


workers = ['research', 'writer', 'analyst', 'Finish']

def supervisor_node(state: Supervisorstate):
    print(" [Supervisor] Deciding..."f"(iteration {state['iteration']})")
    result_summary = ("\n".join(state['results'][-2:]) if state['results'] else "No results yet")

    prompt = f"""
    You are a Supervisor managing a team.
    
    Task: {state['task']}
    
    Work done so far : {result_summary}
    
    Available workers:
    - research : For gathering information
    - writer: For creating content
    - analyst: For analysis and insights
    - FINISH : When task is complete
    
    Iteration : {state['iteration']}/3

    Rules: 
    1. Start with research
    2. Then writer or analyst
    3. After 2-3 workers -> FINISH
    4. If iteration >= 3 -> FINISH

    Reply with ONLY one worker name.
    """

    response = llm.invoke(prompt)
    next_agent = response.content.strip().lower()

    if next_agent not in [w.lower() for w in workers]:
        next_agent = 'FINISH'

    if state['iteration'] >= 3:
        next_agent = 'FINISH'

    print(f" [Supervisor] → {next_agent}")

    return {
        'next_agent': next_agent,
        'iteration': state['iteration'] + 1
    }

def final_node(state: Supervisorstate):
    print(f" [Final] Compiling response...")

    all_results = "\n\n".join(state['results'])

    prompt = f"""
    Compile a comprehensive final response for : {state['task']}
    
    Based on team's work: {all_results}
    
    Create a well-structured final answer.
    """

    response = llm.invoke(prompt)
    return {"final_response": response.content}

def route_to_worker(state: Supervisorstate):
    next_agent = state['next_agent'].upper()

    if next_agent == 'FINISH':
        return "final"
    elif "RESEARCH" in next_agent:
        return "research"
    elif "WRITER" in next_agent:
        return 'writer'
    elif 'ANALYST' in next_agent:
        return 'analyst'
    else:
        return 'final'
    

graph = StateGraph(Supervisorstate)

graph.add_node("supervisor", supervisor_node)
graph.add_node('research', research_agent)
graph.add_node('writer', writer_agent)
graph.add_node('analyst', analyst_agent)
graph.add_node('final', final_node)

graph.set_entry_point('supervisor')

graph.add_conditional_edges(
    'supervisor',
    route_to_worker,
    {
        'research': 'research',
        'writer': 'writer',
        'analyst': 'analyst',
        'final': 'final'
    }
)

graph.add_edge('research', 'supervisor')
graph.add_edge('writer', 'supervisor')
graph.add_edge('analyst', 'supervisor')

graph.add_edge('final', END)

app = graph.compile()

def run_supervisor(task):
    print("="*50)
    print(f"Task : {task}")
    print("="*50)

    result = app.invoke({
        "messages": [HumanMessage(content=task)],
        "next_agent": "",
        "task": task,
        "results": [],
        "final_response": "",
        'iteration': 0
    })

    print("="*50)
    print('FINAL RESPONSE : ')
    print("="*50)
    print(result['final_response'][:500])
    print(f"Agent used : {len(result['results'])}")

run_supervisor("What are the largest trends in AI Engineering?")

run_supervisor('Explain benefits of python for Data science.')

print("=== Interactive Supervisor ===")
print("quit - for close")

while True:
    task = input("Task :")
    if task.lower() == 'quit':
        print('Bye!')
        break
    run_supervisor(task)

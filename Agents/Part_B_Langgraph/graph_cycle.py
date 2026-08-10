from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from typing import TypedDict, List
import os
from dotenv import load_dotenv

load_dotenv()

print("=== Graph Cycles ===")

class LoopState(TypedDict):
    count: int
    max_count: int
    messages: List[str]
    done: bool

def process_node(state: LoopState):
    count = state['count'] + 1
    messages = state['messages'] + [f"Processing Iteration : {count}"]
    print(f"[Process] Iteration : {count}")

    done = count  >= state['max_count']

    return {
        "count": count,
        "messages": messages,
        "done": done
    }

def finish_node(state: LoopState):
    print("[Finish] Done!")
    return {
        "messages": state['messages'] + ["Process complete!"]
    }

def should_continue(state: LoopState):
    if state['done']:
        print('[Router] -> finish')
        return "finish"
    print("[Router] -> loop again")
    return "process"

graph = StateGraph(LoopState)

graph.add_node("process", process_node)
graph.add_node("finish", finish_node)

graph.set_entry_point("process")

graph.add_conditional_edges(
    "process",
    should_continue,
    {
        "process": "process",
        "finish": "finish"
    }
)

graph.add_edge("finish", END)

app = graph.compile()

print("--- Loop Test (3 iterations) ---")
result = app.invoke({
    "count": 0,
    "max_count": 3,
    "messages": [],
    "done": False
})

print(f"Messages: ")
for msg in result['messages']:
    print(f" -> {msg}")

print("=== Research Agent with Retry ===")

llm = ChatGroq(
    model='llama-3.3-70b-versatile',
    api_key=os.getenv('GROQ_API_KEY'),
    temperature=0.3
)

class ResearchState(TypedDict):
    question: str
    attempts: int
    max_attempts: int
    answer: str
    satisfied: bool


def research_node(state: ResearchState):
    attempts = state['attempts'] + 1
    print(f" [Research] Attempt {attempts}")

    prompt = f"""Answer this question concisely: {state['question']}
    
    Attempt Number : {attempts}
    Give a clear, direct answer.
    """
    response = llm.invoke(prompt)
    answer = response.content

    return {
        "attempts": attempts,
        "answer": answer
    }

def evaluate_node(state: ResearchState):
    print(f"  [Evaluate] Checking answer...")

    prompt = f"""
    Question : {state['question']}
    Answer : {state['answer']}
    
    Is this answer complete and satisfactory?
    Reply with ONLY: yes or no
    """

    response = llm.invoke(prompt)
    satisfied = "yes" in response.content.lower()
    print(f"  [Evaluate] Satisfied: {satisfied}")
    
    return {"satisfied": satisfied}

def should_retry(state: ResearchState):
    if state['satisfied']:
        print(" [Router] → Answer good!")
        return "done"
    if state['attempts'] >= state['max_attempts']:
        print("  [Router] → Max attempts reached!")
        return "done"
    print("  [Router] → Retrying...")
    return "retry"

def done_node(state: ResearchState):
    return {"answer": state['answer']}


research_graph = StateGraph(ResearchState)

research_graph.add_node("research", research_node)
research_graph.add_node("evaluate", evaluate_node)
research_graph.add_node("done", done_node)

research_graph.set_entry_point("research")

research_graph.add_edge('research', 'evaluate')

research_graph.add_conditional_edges(
    'evaluate',
    should_retry,
    {
        "retry": "research",
        "done": "done"
    }
)

research_graph.add_edge("done", END)

research_app = research_graph.compile()

print("--- Research with Retry ---")
result = research_app.invoke({
    "question": "What is machine learning?",
    "attempts": 0,
    "max_attempts": 3,
    "answer": "",
    "satisfied": False
})

print(f"Final answer ({result['attempts']} attempts):")
print(result["answer"][:200])

print("=== Max Iterations Saftey ===")

class SafeState(TypedDict):
    counter: str
    result: str

def increment(state: SafeState):
    print(f"  Counter: {state['counter']}")
    return {"counter": state["counter"] + 1}

def check_done(state: SafeState):
    if state["counter"] >= 5:
        return "end"
    return "continue"


safe_graph = StateGraph(SafeState)
safe_graph.add_node("increment", increment)
safe_graph.set_entry_point("increment")

safe_graph.add_conditional_edges(
    "increment",
    check_done,
    {
        "continue": "increment",
        "end": END
    }
)

safe_app = safe_graph.compile()

result = safe_app.invoke(
    {"counter": 0, "result": ""},
    config={"recursion_limit": 10}
)

print(f"Final counter: {result['counter']}")

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from typing import TypedDict, List
import os
from dotenv import load_dotenv

load_dotenv()

print("=== LangGraph Basics ===")

class SimpleState(TypedDict):
    message: str
    result: str
    step: int

def node_greet(state: SimpleState):
    print(" [Node: Greet] Running...")
    return {
        "result": f"Hello! You said:  {state['message']}",
        "step": state["step"] + 1
    }

def node_process(state: SimpleState):
    print(" [Node: Process] Running...")
    return{
        "result": state['result'] + "| Processed!",
        "step": state["step"] + 1
    }

def node_finish(state: SimpleState):
    print(" [Node: Finish] Running...")
    return{
        "result": state['result'] + "  | Done!",
        "step": state["step"] + 1
    }

graph = StateGraph(SimpleState)

graph.add_node("greet", node_greet)
graph.add_node("process", node_process)
graph.add_node("finish", node_finish)

graph.add_edge("greet", "process")
graph.add_edge("process", "finish")
graph.add_edge("finish", END)

graph.set_entry_point("greet")

app = graph.compile()

print("--- Simple Graph Run ---")

result = app.invoke({
    "message": "Nameste!",
    "result": "",
    "step": 0
})

print(f"Final Result: {result['result']}")
print(f"Steps taken: {result['step']}")

print("=== LLM + Langgraph ===")

class ChatState(TypedDict):
    messages: List
    response: str

llm = ChatGroq(
    model='llama-3.3-70b-versatile',
    api_key=os.getenv('GROQ_API_KEY'),
    temperature=0.7
)

def user_input_node(state: ChatState):
    print("[Node: User Input] Processing ...")
    return {'messages': state['messages']}

def llm_node(state: ChatState):
    print("[Node: LLM] Calling AI...")
    response = llm.invoke(state['messages'])
    return {
        "response": response.content,
        "messages": state['messages'] + [AIMessage(content=response.content)]
    }

def output_node(state: ChatState):
    print("[Node: Output] Preparing response...")
    return {"response": state['response']}

chat_graph = StateGraph(ChatState)

chat_graph.add_node("input", user_input_node)
chat_graph.add_node("llm", llm_node)
chat_graph.add_node("output", output_node)

chat_graph.add_edge("input", "llm")
chat_graph.add_edge("llm", "output")
chat_graph.add_edge("output", END)

chat_graph.set_entry_point("input")

chat_app = chat_graph.compile()


print("--- LLM graph run ---")
result = chat_app.invoke({
    'messages': [HumanMessage(content="What is Python ?")],
    'response': ""
})

print(f"AI Response : {result['response'][:200]}")

print("=== State Managment ===")

class CounterState(TypedDict):
    count: int
    history: List[str]
    message: str

def increment_node(state: CounterState):
    new_count = state['count'] + 1
    new_histroy = state['history'] + [f"Count: {new_count}"]
    print(f"Count : {new_count}")
    return {
        "count": new_count,
        "history": new_histroy
    }

def double_node(state: CounterState):
    new_count = state['count'] * 2
    new_history = state['history'] + [f"Doubled : {new_count}"]
    print(f"Doubled : {new_count}")
    return{
        "count": new_count,
        "history": new_history
    }

def summary_node(state: CounterState):
    summary = (
        f"Final count : {state['count']}\n"
        f"Steps : {state['history']}"
    )
    return {'message': summary}


counter_graph = StateGraph(CounterState)

counter_graph.add_node('increment', increment_node)
counter_graph.add_node('double', double_node)
counter_graph.add_node('summary', summary_node)

counter_graph.add_edge('increment', 'double')
counter_graph.add_edge('double', 'summary')
counter_graph.add_edge('summary', END)

counter_graph.set_entry_point('increment')

counter_app = counter_graph.compile()

print("--- Counter Graph ---")
result = counter_app.invoke({
    "count": 5,
    "history": [],
    "message": ""
})

print(f"{result['message']}")
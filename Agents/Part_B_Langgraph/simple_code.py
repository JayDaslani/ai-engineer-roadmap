from langgraph.graph import StateGraph, END
from typing import TypedDict

class MyState(TypedDict):
    message : str
    result : str

def node1(state):
    return {"result": "Node 1 done"}

def node2(state):
    return {"result": "Node 2 done"}

graph = StateGraph(MyState)

graph.add_node("node1", node1)
graph.add_node("node2", node2)

graph.add_edge("node1", "node2")
graph.add_edge("node2", END)

graph.set_entry_point("node1")

app = graph.compile()


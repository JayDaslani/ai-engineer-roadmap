from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from typing import TypedDict, List, Annotated
import operator
import os
import math
from dotenv import load_dotenv

load_dotenv()

print("=== Stateful Agent ===")

llm = ChatGroq(
    model='llama-3.3-70b-versatile',
    api_key=os.getenv('GROQ_API_KEY'),
    temperature=0
)

@tool
def calculator(expression: str) -> str:
    """It calculates Math expression."""
    try:
        result = eval(
            expression,
            {"__builtins__": {}},
            {"sqrt": math.sqrt, "pi": math.pi}
        )
        return f"Result: {result}"
    except Exception as e:
        return f"Error : {e}"
    

@tool
def get_weather(city: str) -> str:
    """It provides weather information for a city"""
    data = {
        "Pune": "26°C, Pleasant",
        "Mumbai": "28°C, Humid",
        "Delhi": "35°C, Hot",
        "Ahmedabad": "32°C, Sunny"
    }
    return data.get(city, "City not found")

@tool
def save_note(note: str) -> str:
    """It saves important notes.
    Use it when you need to save something for user."""
    return f"Note saved : '{note}'"

tools = [calculator, get_weather, save_note]
tools_map = {t.name: t for t in tools}

llm_with_tools = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[List, operator.add]
    session_id: str


def llm_node(state: AgentState):
    print("[LLM] Thinking....")

    system = SystemMessage(
        content="""You are a helpful assistant.
        You have access to tools:
        -calculator: math calculations
        -get_weather: city weather
        -save_note: save important notes
        
        Use tools when needed."""
    )

    messages = [system] + state['messages']
    response = llm_with_tools.invoke(messages)

    print(f"[LLM] Response type : {'tool_call' if response.tool_calls else 'text'}")

    return {"messages": [response]}

def tool_node(state: AgentState):
    last_message = state['messages'][-1]
    results = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call['name']
        tool_args = tool_call['args']
    
        print(f"[Tool] Running : {tool_name}")
        print(f"[Tool] Args : {tool_args}")

        if tool_name in tools_map:
            result = tools_map[tool_name].invoke(tool_args)
            print(f"[Tool] Result : {result}")
        else:
            result = f"Tool {tool_name} not found"

        results.append(ToolMessage(content=str(result),tool_call_id=tool_call['id']))

    return {'messages': results}

def should_use_tool(state: AgentState):
    last_message = state['messages'][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print("  [Router] → Using tool")
        return "tool"
    
    print("  [Router] → Done")
    return "end"
    
memory = MemorySaver()

graph = StateGraph(AgentState)

graph.add_node('llm', llm_node)
graph.add_node('tool', tool_node)

graph.set_entry_point('llm')

graph.add_conditional_edges(
    'llm',
    should_use_tool,
    {
        "tool": "tool",
        "end": END
    }
)

graph.add_edge("tool", "llm")

app = graph.compile(checkpointer=memory)

def chat(message, session_id='default'):
    print(f"{'='*45}")
    print(f"You: {message}")

    config = {
        "configurable": {
            "thread_id": session_id
        }
    }

    result = app.invoke(
        {
            "messages": [HumanMessage(content=message)],
            "session_id": session_id
        },
        config=config
    )

    last_ai = None
    for msg in reversed(result['messages']):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            last_ai = msg
            break

    response = last_ai.content if last_ai else "..."
    print(f"Agent : {response}")
    return response

print("=== Testing Stateful Agent ===")

chat('My name is Jay')
chat('Calculate: 10*15 + sqrt(25)')
chat("Tell me the weather in pune.")
chat('What is my name ?')
chat('Save note: LangGraph seekh liya!')

while True:
    q = input("You: ")
    if q.lower() == 'quit':
        print("Bye!")
        break
    chat(q)






from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from typing import TypedDict, List
import os
from dotenv import load_dotenv

load_dotenv()

print("=== Conditional Graph ===")

class MessageState(TypedDict):
    message: str
    category: str
    response: str

def classify_node(state: MessageState):
    message = state['message'].lower()

    if any(word in message for word in ['weather', 'mausam', 'temperature']):
        category = 'weather'
    elif any(word in message for word in ["calculate", "math", "+", "-", "*", "/"]):
        category = 'math'
    elif any(word in message for word in ['hello', 'hi', 'namaste', 'hey']):
        category = 'greeting'
    else:
        category = 'general'

    print(f"[Classify] Category : {category}")
    return {"category": category}

def weather_node(state: MessageState):
    print("[Weather node] running ...")
    return {
        "response": "Today's weather : 28°C, Sunny!"
    }

def math_node(state: MessageState):
    print("[Math node] running ...")
    return {
        "response": "I am solving a math problem."
    }

def greeting_node(state: MessageState):
    print("[Greeting node ] running...")
    return {
        "response": "hello!How are you?I can help you."
    }

def general_node(state: MessageState):
    print("[general node] running...")
    return {
        "response": "I will help you!"
    }

def route_message(state: MessageState):
    category = state['category']
    print(f"[Router] Routing to: {category}")
    return category

graph = StateGraph(MessageState)

graph.add_node("classify", classify_node)
graph.add_node('weather', weather_node)
graph.add_node('math', math_node)
graph.add_node('greeting', greeting_node)
graph.add_node('general', general_node)

graph.set_entry_point('classify')

graph.add_conditional_edges(
    "classify",
    route_message,
    {
        "weather": "weather",
        "math": "math",
        "greeting": "greeting",
        "general": "general"
    }
)

graph.add_edge('weather', END)
graph.add_edge('math', END)
graph.add_edge('greeting', END)
graph.add_edge('general', END)

app = graph.compile()

def test(message):
    print(f"Input : '{message}'")
    result = app.invoke({
        "message": message,
        "category": "",
        "response": ""
    })
    print(f"Response: {result['response']}")

test("Hello!")
test("What is today's weather ?")
test("Calcuate this : 20 + 10")
test("What is python ?")

print("=== LLM Based Classification ===")

llm = ChatGroq(
    model='llama-3.3-70b-versatile',
    api_key=os.getenv('GROQ_API_KEY'),
    temperature=0
)

class SmartState(TypedDict):
    message: str
    intent: str
    response: str

def llm_classify(state: SmartState):
    prompt = f"""Classify this message into ONE category:
    -greeting 
    -question
    -task
    -unkown
    
    message = {state['message']}

    Reply with ONLY the category name.
    Nothing else.
    """

    result = llm.invoke(prompt)
    intent = result.content.strip().lower()
    print(f"[LLM Classify] Intent : {intent}")
    return {"intent": intent}

def handle_greeting(state: SmartState):
    response = llm.invoke(
        f"Respond warmly to: {state['message']}"
    )
    return {'response': response.content}

def handle_question(state: SmartState):
    response = llm.invoke(
        f"Answer this : {state['message']}"
    )
    return {'response': response.content}

def handle_task(state: SmartState):
    response = llm.invoke(
        f"Complete this task : {state['message']}"
    )
    return {'response': response.content}

def handle_unknown(state: SmartState):
    return {
        "response": "I didn't understand. Ask again."
    }

def route_intent(state: SmartState):
    intent = state['intent']
    if intent in ["greeting", "question", "task"]:
        return intent
    return "unknown"

smart_graph = StateGraph(SmartState)

smart_graph.add_node("classify", llm_classify)
smart_graph.add_node("greeting", handle_greeting)
smart_graph.add_node("question", handle_question)
smart_graph.add_node("task", handle_task)
smart_graph.add_node("unknown", handle_unknown)

smart_graph.set_entry_point('classify')

smart_graph.add_conditional_edges(
    "classify",
    route_intent,
    {
        "greeting": "greeting",
        "question": "question",
        "task": "task",
        "unknown": "unknown"
    }
)

smart_graph.add_edge('greeting', END)
smart_graph.add_edge('question', END)
smart_graph.add_edge('task', END)
smart_graph.add_edge('unknown', END)

smart_app = smart_graph.compile()

def smart_test(message):
    print(f"Input : '{message}'")
    result = smart_app.invoke({
        'message': message,
        'intent': "",
        'response': ""
    })
    print(f"Response : {result['response'][:250]}")

smart_test("Hello! How are you ?")
smart_test("What is python ?")
smart_test("Write a poem on flowers")


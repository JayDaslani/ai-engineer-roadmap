from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from typing import TypedDict
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model='llama-3.3-70b-versatile',
    api_key=os.getenv('GROQ_API_KEY'),
    temperature=0.3
)

class SupportState(TypedDict):
    user_message: str
    category: str
    response: str
    quality_score: int
    escalated: bool
    attempts: int


def classify_node(state: SupportState):
    prompt = f"""
    Classify this customer message: {state['user_message']}
    
    Categories:
    -faq (general questions)
    -tech (technical issues)
    -billing (payment/billing)
    
    Reply with ONLY category name.
    """

    result = llm.invoke(prompt)
    category = result.content.strip().lower()

    if category not in ['faq', 'tech', 'billing']:
        category='faq'
    print(f"  [Classify] → {category}")

    return {
        "category": category,
        "attempts": state['attempts'] + 1
    }

def faq_node(state: SupportState):
    print(" [FAQ Bot] Answering...")
    prompt = f"""
    You are a helpful FAQ bot.
    Answer this general question : {state['user_message']}
    
    Keep answer short and clear.
    """

    result = llm.invoke(prompt)
    return {'response': result.content}

def tech_node(state: SupportState):
    print(" [Tech Agent] Solving...")
    prompt = f"""
    You are a helpful technical support agent.
    Solve this technical issues : {state['user_message']}
    
    provide step-step by solution.
    """

    result = llm.invoke(prompt)
    return {'response': result.content}

def billing_node(state: SupportState):
    print(" [Billing Agent] Processing...")
    prompt = f"""
    You are a billing support agent.
    Handle this billing query : {state['user_message']}
    Be helpful and professional.
    """

    result = llm.invoke(prompt)
    return {'response': result.content}

def quality_check_node(state: SupportState):
    print("  [Quality Check] Evaluating...")
    prompt = f"""
    Rate this customer support response:
    Question : {state['user_message']}
    Response : {state['response']}

    Score 1-10 (10 = perfect).
    Reply with ONLY a number.
    """

    result = llm.invoke(prompt)
    try:
        score = int(result.content.strip())
    except:
        score = 7
    print(f"[Quality] Score : {score}/10")
    return {"quality_score": score}

def escalate_node(state: SupportState):
    print(" [Escalate] Connecting to human...")
    return {
        "escalated": True,
        "response" : (
            "I Connect a human agent to you."
            "Please wait..."
        )
    }

def route_category(state: SupportState):
    return state['category']

def route_quality(state: SupportState):
    if state['quality_score'] >= 7:
        return 'satisfied'
    if state['attempts'] >= 2:
        return 'escalate'
    return "escalate"

graph = StateGraph(SupportState)

graph.add_node('classify', classify_node)
graph.add_node('faq', faq_node)
graph.add_node('tech', tech_node)
graph.add_node('billing', billing_node)
graph.add_node('quality', quality_check_node)
graph.add_node('escalate', escalate_node)

graph.set_entry_point('classify')

graph.add_conditional_edges(
    'classify',
    route_category,
    {
        'faq': 'faq',
        'tech': 'tech',
        'billing': 'billing'
    }
)

graph.add_edge('faq', 'quality')
graph.add_edge('tech', 'quality')
graph.add_edge('billing', 'quality')

graph.add_conditional_edges(
    'quality',
    route_quality,
    {
        'satisfied': END,
        'escalate': 'escalate'
    }
)

graph.add_edge('escalate', END)

app = graph.compile()

def handle_support(message):
    print('='*45)
    print(f'Customer: {message}')
    print('='*45)

    result = app.invoke({
        'user_message': message,
        "category": "",
        "response": "",
        "quality_score": 0,
        "escalated": False,
        "attempts": 0
    })

    print(f"Response : {result['response'][:200]}")
    print(f"Quality: {result['quality_score']}/10")
    if result['escalated']:
        print("Status: Escalated to human ⚠️")
    else:
        print("Status: Resolved ✅")

handle_support("How do I reset my password?")
handle_support('My payment failed 3 times!')
handle_support("App will be crash!")

while True:
    msg = input("Customer :")
    if msg.lower() == 'quit':
        print("GoodBye!")
        break
    handle_support(msg)





from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from typing import TypedDict, List
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model='llama-3.3-70b-versatile',
    api_key=os.getenv('GROQ_API_KEY'),
    temperature=0.3
)

class ApprovalState(TypedDict):
    task: str
    plan: str
    approved: bool
    result: str
    human_feedback: str

def plan_node(state: ApprovalState):
    print(" [Plan] Creating plan...")

    prompt = f"""
    Create a step-by-step plan for : {state['task']}
    
    Keep it short - 3-4 steps only."""

    response = llm.invoke(prompt)
    plan = response.content

    print("Plan created!")
    return {"plan": plan}


def human_approval_node(state: ApprovalState):
    print("="*45)
    print("HUMAN APPROVAL REQUIRED!")
    print("="*45)
    print(f"Task : {state['task']}")
    print(f"Proposed plan : {state['plan']}")
    print("="*45)

    response = input("Approve this plan? (Yes/No): ").strip().lower()

    feedback = ""
    if response != 'yes':
        feedback = input("Give me feedback (what should I change ?)")

    approved = response == "yes"
    print(f"  Decision: {'Approved ✅' if approved else 'Rejected ❌'}")

    return {
        "approved": approved,
        "human_feedback": feedback
    }

def execute_node(state: ApprovalState):
    print("  [Execute] Running approved plan...")

    prompt = f"""
    Execute this plan and show results:
    Task : {state['task']}
    plan : {state['plan']}
    
    Show what was done step by step.
    """

    response = llm.invoke(prompt)
    return {"result": response.content}

def revise_node(state: ApprovalState):
    print(" [Revise] Updating plan with feedback...")

    prompt = f"""
    Revise this plan based on feedback:
    Original Task: {state['task']}
    Original Plan: {state['plan']}
    Human Feedback : {state['human_feedback']}

    Create improved plan.
    """
    response = llm.invoke(prompt)
    return {
        "plan": response.content,
        "approved": False
    }

def approval_router(state: ApprovalState):
    if state['approved']:
        return "execute"
    elif state['human_feedback']:
        return 'revise'
    else:
        return "end"
    

graph = StateGraph(ApprovalState)

graph.add_node("plan", plan_node)
graph.add_node("human_approval", human_approval_node)
graph.add_node("execute", execute_node)
graph.add_node("revise", revise_node)

graph.set_entry_point("plan")
graph.add_edge("plan", "human_approval")

graph.add_conditional_edges(
    "human_approval",
    approval_router,
    {
        "execute": "execute",
        "revise": "revise",
        "end": END
    }
)

graph.add_edge("revise", "human_approval")
graph.add_edge("execute", END)

app = graph.compile()

print("--- Approval Workflow Test ---")
result = app.invoke({
    "task": "Write a Pyhton script to analyze sales data",
    "plan": "",
    "approved": False,
    "result": "",
    "human_feedback": ""
})

if result['result']:
    print(f"Final result : {result['result'][:300]}")
else:
    print("Task was cancelled.")

print("=== interrupt_before method ===")

class EmailState(TypedDict):
    recipient: str
    subject: str
    body: str
    sent: bool

def draft_email(state: EmailState):
    print("  [Draft] Creating email...")

    prompt = f"""
    Write a professional email:
    To: {state['recipient']}
    Subject: {state['subject']}
    
    Wriet a short 3-4 line email body.
    """

    response = llm.invoke(prompt)
    return {"body": response.content}

def send_email(state: EmailState):
    print(f"  [Send] Email sent to {state['recipient']}!")
    return {'sent': True}

memory = MemorySaver()

email_graph = StateGraph(EmailState)

email_graph.add_node("draft", draft_email)
email_graph.add_node("send", send_email)

email_graph.set_entry_point('draft')
email_graph.add_edge('draft', 'send')
email_graph.add_edge('send', END)

email_app = email_graph.compile(
    checkpointer=memory,
    interrupt_before=['send']
)

config = {"configurable": {"thread_id": "email_1"}}

print("--- Email Workflow ---")
result = email_app.invoke(
    {
        "recipient": "manager@company.com",
        "subject": "Project Update",
        "body": "",
        "sent": False
    },
    config=config
)

print("Drafted Email :")
print(f"To: {result['recipient']}")
print(f"Subject : {result['subject']}")
print(f"Body : {result['body']}")

print("="*40)
approval = input("Send this email ? (Yes/No):")

if approval.lower() == 'yes':
    final = email_app.invoke(None, config=config)
    print(f"Email sent: {final['sent']} ✅")
else:
    print("Email cancelled ❌")


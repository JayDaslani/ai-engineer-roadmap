from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
import math
from dotenv import load_dotenv

load_dotenv()

print("=== Agent with memory ===")

llm = ChatGroq(
    model='llama-3.3-70b-versatile',
    api_key=os.getenv('GROQ_API_KEY'),
    temperature=0
)

@tool
def calculator(expression: str) -> str:
    """
    Calculates mathematical expressions.
    Use this when you need to calculate numbers.
    Example : '2 + 2', '10*5', 'sqrt(16)'.
    """
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
    """
    It gives the weather information of a city.
    Use it when you want to know the current weather.
    """
    data = {
        "pune": "26°C, Pleasant",
        "Mumbai": "28°C, Humid",
        "Delhi": "35°C, Hot",
        "Ahmedabad": "32°C, Sunny"
    }
    return data.get(city, "City not found")

tools = [calculator, get_weather]

prompt = PromptTemplate.from_template("""
You are a helpful AI assistant with memory.
You remember previous conversations.

Chat History:
{chat_history}

Available tools:
{tools}

Format:
Question: {input}
Thought: what to do?
Action: tool [{tool_names}]
Action Input: tool input
Observation: result
Thought: do I have answer?
Final Answer: answer

If no tool needed — directly answer.

Begin!
Question: {input}
Thought: {agent_scratchpad}
""")

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True
)

agent_with_memory = RunnableWithMessageHistory(
    executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

def ask(question, session='default'):
    print(f"You : {question}")
    result = agent_with_memory.invoke(
        {'input': question},
        config={"configurable" : {"session_id": session}}
    )
    print(f"Agent : {result['output']}")
    return result['output']

print("=== Memory test ===")
ask("My name is Jay.")
ask("I am live in Pune.")
ask("What is my name?")
ask("Where do i live ?")

print("=== Interactive Memory agent ===")
print("Press quit for chat close")

while True:
    q = input("You :")
    if q.lower() == "quit":
        print('Bye!')
        break
    ask(q)


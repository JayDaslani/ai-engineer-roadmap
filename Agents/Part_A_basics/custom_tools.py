from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_classic.agents import create_react_agent,AgentExecutor
from langchain_core.prompts import PromptTemplate
import os
import math
from dotenv import load_dotenv

load_dotenv()

@tool
def  calculator(expression: str) -> str:
    """
    Calculates mathematical expressions.
    Use this when you need to calculate numbers.
    Example : '2 + 2', '10*5', 'sqrt(16)'.
    """
    try:
        result = eval(
            expression,
            {"__builtins__": {}},
            {"sqrt": math.sqrt,
             "pi": math.pi,
             "sin": math.sin,
             "cos": math.cos}
        )
        return f"Result : {result}"
    except Exception as e:
        return f"Error : {e}"
    
@tool
def word_counter(text: str) -> str:
    """
    It counts the number of words in the text. 
    Use it when you need to find the world count.
    """
    words = len(text.split())
    chars = len(text)
    return (
        f"Words : {words}, "
        f"Character : {chars}"
    )

@tool
def celsius_to_fahrenfit(celsius: str) -> str:
    """
    It converts Celsius to Fahrenheit. 
    Use it when you need to convert the temperature.
    """
    
    c = float(celsius)
    fahrenfit = (c * 9/5) + 32
    return f"{c}°C = {fahrenfit:.1f}°F"

@tool
def simple_interest(input_str: str) -> str:
    """Calculates simple interest.
    Provide input as a comma-separated string: "principal, rate, time"
    Example Action Input: "50000, 8, 3"
    """
    try:
        parts = [float(i.strip()) for i in input_str.split(',')]
        principal, rate, time = parts[0], parts[1], parts[2]
        
        interest = (principal * rate * time) / 100
        total = principal + interest
        return (
            f"Principal : {principal}\n"
            f"Interest : {interest}\n"
            f"Total Amount : {total}"
        )
    except Exception as e:
        return f"Error computing simple interest. Make sure to pass numbers separated by commas. Details: {e}"

print("=== Direct Tool Test ===")
print(calculator.invoke("2 + 2"))
print(word_counter.invoke("Hello World this is Jay"))
print(celsius_to_fahrenfit.invoke("100")) 
print(simple_interest.invoke("10000, 5, 2"))

print("=== Agent with Custom Tools ===")

llm = ChatGroq(
    model='llama-3.3-70b-versatile',
    api_key=os.getenv('GROQ_API_KEY'),
    temperature=0
)

tools = [
    calculator,
    word_counter,
    celsius_to_fahrenfit,
    simple_interest
]

prompt = PromptTemplate.from_template("""
Answer using available tools as best you can.
Tools available:
{tools}

Use the following strict format: 
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!
Question: {input}
Thought: {agent_scratchpad}
""")

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True
)

print("Q: 25 celsius to fahrenhit ?")
result = executor.invoke({
    "input": "Convert 25 celsius to fahrenheit"
})
print(f"Answer : {result['output']}")

print("Q: Simple interest calculation")
result = executor.invoke({
    "input": "Calculate simple interest "
             "for Principal 50000, "
             "rate 8%, time 3 years"
})
print(f"Answer : {result['output']}")

print("Q: Math calculation")
result = executor.invoke({
    "input": "What is sqrt(144) + 10 * 5?"
})
print(f"Answer : {result['output']}")

print("=== Tool Description Importance ===")

@tool
def bad_tools(x: str) -> str:
    """Tool"""
    return x

@tool
def good_tool(city: str) -> str:
    """
    Kisi bhi city ka current time batata hai.
    Use karo jab city ka time jaanna ho.
    Input: city name (e.g. 'Mumbai', 'Delhi')
    """
    times = {
        "Mumbai": "14:30 IST",
        "Delhi": "14:30 IST",
        "Pune": "14:30 IST"
    }
    return times.get(city, "City not found")

print("Good tool test:")
print(good_tool.invoke("Mumbai"))
print(f"Tool name : {good_tool.name}")
print(f"Tool description : {good_tool.description}")

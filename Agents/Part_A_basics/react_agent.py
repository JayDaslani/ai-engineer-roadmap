from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
import os
import math
from dotenv import load_dotenv
import json

load_dotenv()

llm = ChatGroq(
    model='llama-3.3-70b-versatile',
    api_key=os.getenv('GROQ_API_KEY'),
    temperature=0
)
search = DuckDuckGoSearchRun()

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
            {"sqrt": math.sqrt,
             "pi": math.pi}
        )
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"
    
@tool
def get_weather_info(city: str) -> str:
    """
    It gives the weather information of a city.
    Use it when you want to know the current weather.
    """
    weather_data = {
        "Mumbai": "28°C, Humid, Partly cloudly",
        "Delhi": "35°C, Hot, Sunny",
        "Pune": "26°C, Pleasant, Clear",
        "Ahmedabad": "32°C, Hot, Sunny",
        "Bangalore": "24°C, Cool, Cloudy"
    }
    return weather_data.get(
        city,
        f"{city} ka weather data nahi hai"
    )

@tool
def currency_coverter(input_str: str) -> str:
    """
    Converts Currency.
    Accepts input as a comma-separated string OR a JSON string.
    Example Action Input: "1000, USD, INR" OR '{"amount": 1000, "from_currency": "USD", "to_currency": "INR"}'
    Supported currencies: USD, INR, EUR, GBP
    """
    try:
        
        if input_str.strip().startswith("{"):
            data = json.loads(input_str)
            amount = float(data.get("amount"))
            from_currency = str(data.get("from_currency")).upper()
            to_currency = str(data.get("to_currency")).upper()
        
        else:
            parts = [i.strip() for i in input_str.split(',')]
            amount = float(parts[0])
            from_currency = parts[1].upper()
            to_currency = parts[2].upper()

        rates = {
            "USD_INR": 95.05,
            "INR_USD": 0.011,
            "EUR_INR": 109.83,
            "INR_EUR": 0.0091,
            "USD_EUR": 0.87,
            "EUR_USD": 1.15
        }

        key = f"{from_currency}_{to_currency}"
        if key in rates:
            result = amount * rates[key]
            return f"{amount} {from_currency} = {result:.2f} {to_currency}"
        
        return "Currency pair is not supported"

    except Exception as e:
        return f"Error in currency conversion: {e}. Provide input as '1000, USD, INR'"

tools = [search, calculator, get_weather_info, currency_coverter]

print(f"Total tools : {len(tools)}")
for t in tools:
    print(f"→ {t.name}: {t.description[:100]}")

prompt = PromptTemplate.from_template("""You are a helpful AI assistant.
Answer questions using available tools.

Available tools:
{tools}

Format:
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
Thought:{agent_scratchpad}""")

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=6,
    handle_parsing_errors=True
)

def ask_agent(question):
    print("="*50)
    print(f"Question : {question}")
    print("="*50)

    result = executor.invoke({'input': question})

    print(f"✅ Final Answer: {result['output']}")
    return result['output']

ask_agent("What is 15 * 8 + sqrt(64)?")

ask_agent("Pune ka weather kaisa hai?")

ask_agent("Convert 1000 USD to INR")

ask_agent("What is LangGraph used for?")

ask_agent("Pune ka weather batao aur "
    "500 USD ko INR mein convert karo")


print("=== Interactive react agent ===")
print("Available : Search, Calculator, Weather, Currency")
print("quit for chat close")

while True:
    question = input("You :")

    if question.lower() == 'quit':
        print("Bye!")
        break

    try:
        result = executor.invoke({"input": question})
        print(f"\nAgent: {result['output']}\n")
    except Exception as e:
        print(f"Error: {e}\n")
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool
import os
from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate


load_dotenv()

search_tool = DuckDuckGoSearchRun()

wiki_wrapper = WikipediaAPIWrapper(
    top_k_results=2
)

wiki_tool = WikipediaQueryRun(
    api_wrapper = wiki_wrapper
)

print("=== Tool test ===")

print("1. Search tool : ")
result = search_tool.run("Langchain AI framework 2024")
print(result[:300])

print("2. Wikipedia Tool :")
result = wiki_tool.run("Python (programming language)")
print(result[:300])


print("=== First Agent ===")

llm = ChatGroq(
    model='llama-3.3-70b-versatile',
    api_key=os.getenv('GROQ_API_KEY'),
    temperature=0
)

tools = [search_tool, wiki_tool]

prompt = PromptTemplate.from_template("""
       Answer the following uestion using the available tools.
        
        You have access to the following tools: {tools}
        Question: the input question
        Thought: what should I do?
        Action: tool name [{tool_names}]
        Action Input: input for tool
        Observation: tool result
        ... (repeat if needed)
        Thought: I now know the answer
        Final Answer: answer here

        Begin!

        Question: {input}
        Thought: {agent_scratchpad}
    """)

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5
)

print("Q: What is Langchain?")
result = agent_executor.invoke({
    "input": "What is Langchain?"
})
print(f"Final answer : {result['output']}")


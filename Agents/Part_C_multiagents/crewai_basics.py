from crewai import Agent, Task, Crew, Process, LLM
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
import litellm
litellm.drop_params = True
load_dotenv()

print("=== CrewAI Basics ===")

llm = LLM(
    model="openai/llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    temperature=0.3
)
researcher = Agent(
    role='Research Specialist',
    goal='Research topics thoroughly and gather accurate information',
    backstory="""You are a expert researcher 
    with years of experience in gathering 
    and analyzing information from various sources.
    You are detail-oriented and always provide accurate data.""",
    llm=llm,
    verbose=True
)

writer = Agent(
    role='Content Writer',
    goal='Write clear, engaging, and well-structured content',
    backstory="""You are a talented content 
    write who specializes in making complex 
    topic easy to understand. You write 
    in a clear, concise, and engaging style.""",
    llm=llm,
    verbose=True
)

research_task = Task(
    description = """Research the topic:
    'Benefits of Python for AI Develepment
    
    Find:
    -Key advantages
    -Popular libraries
    -Use cases
    -Industry adoption""",
    expected_output="""A detailed research 
    report with facts and data about 
    Python in AI development.""",
    agent=researcher
)

writing_task = Task(
    description = """Using the research provided,
    write an engaging blog post about:
    'Benefits of Python for AI Development'
    
    Include:
    -Introduction
    -Key benefits
    -Real examples
    -Conclusion""",
    expected_output="""A well-written blog post 
    of 3-4 paragraphs ready for publishing.""",
    agent=writer
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential,
    verbose=True
)

print("--- Running Crew ---")
result = crew.kickoff()

print("=== FINAL RESULT ===")
print(result)
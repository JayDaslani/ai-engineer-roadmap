from crewai import Agent, Task, Crew, Process, LLM
import os
from dotenv import load_dotenv
import json

load_dotenv()

print("=== CrewAI Roles + Tools ===")

llm = LLM(
    model="openai/llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    temperature=0.3
)

researcher = Agent(
    role='Senior research analyst',
    goal="""Research topics deeply and provide comprehensive,
    accurate information with sources.""",
    backstory="""You are a Senior research analystwith 10 years experience.
    You specialize in technology and ai topics.
    You always verify facts and provide structured reports. """,
    llm=llm,
    verbose=False,
    allow_delegation=False
)

analyst = Agent(
    role='Data Analyst',
    goal="""Analyze research data and extract meaningful insights, trends, and patterns.""",
    backstory="""You are an expert data analyst who transforms raw research into actionable insights.
    You identify trends, patterns, and key takeways from complex information.""",
    llm=llm,
    verbose=False,
    allow_delegation=False
)


writer = Agent(
    role='Technical writer',
    goal="""Create clear, engaging, and well-structred technical content for various audiences.""",
    backstory="""You are a technical writer who specializes in making complex topics accessible.
    You write in clear, concise style with good structure and flow.""",
    llm=llm,
    verbose=False,
    allow_delegation=False
)

editor = Agent(
    role='Senior Content Editor',
    goal="""Review, polish, and format the written content into a professional, publication-ready career guide.""",
    backstory="""You are a meticulous editor with a strong eye for detail.
    You ensure perfect formatting, tone consistency, technical accuracy, and high readability.""",
    llm=llm,
    verbose=False,
    allow_delegation=False,
    
)

TOPIC = "AI Engineer Career Guide"

research_task = Task(
    description=f"""Research this topic throughly : {TOPIC}

    Cover:
    1. Current applications
    2. Key technologies used
    3. Benefits and Challenges
    4. Future procpects
    5. Real world examples
    
    Be specefic and factual.
    """,
    expected_output="""
    Comprehensive research report with :
    - 5+ key findings
    - Specific examples
    - Data and statistics
    - Structured format
    """,
    agent=researcher
)

analysis_task = Task(
    description=f"""Analyze the research about : {TOPIC}

    Provide:
    1. Top 3 trends
    2. Key opportunities
    3. Major challenges
    4. Competitive landscape
    5. Future predictions
    
    Use the research data provided.
    """,
    expected_output="""
    Analysis report with:
    - Trends identified
    - Opportunities listed
    - Challenges explained
    - Future outlook
    """,
    agent=analyst,
    context=[research_task]

)

writing_task = Task(
    description=f"""Write a professional article about : "{TOPIC}"
    
    Structure:
    1. Executive Summary (2-3 lines)
    2. Introduction
    3. Current State
    4. Key Benefits
    5. Challenges
    6. Future Outlook
    7. Conclusion
    
    Use research and analysis provided.
    Make it engaging and informative.
    """,
    expected_output="""
    Well-written article 500+ words
    with clear structure and flow.
    """,
    agent=writer,
    context=[research_task, analysis_task]
)

editing_task = Task(
    description=f"""
    Review and polish the written draft of the career guide.
    Tasks:
    - Refine clarity, flow, and formatting using clean Markdown
    - Fix any grammer, tone, or structural issues
    - Ensure it reads as an execution-level, professional guide.""",
    expected_output = f"""
    The final, publication-ready AI Engineer guide in beautiful Markdown format.""",
    agent=editor,
    context=[writing_task],
    output_file="final_output.md"
)

crew = Crew(
    agents = [researcher, analyst, writer, editor],
    tasks = [research_task, analysis_task, writing_task, editing_task],
    process=Process.sequential,
    verbose=False
)

print(f"Topic : {TOPIC}")
print("Starting crew....")

result = crew.kickoff()

print('='*50)
print('FINAL OUTPUT :')
print("="*50)
print(result)


output_data = {
    "topic": TOPIC,
    "status": "completed",
    "final_report": str(result)
}

file_name = "ai_engineer_career_guide.json"
with open(file_name, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=4, ensure_ascii=False)

print(f"✅ Final output successfully saved to '{file_name}'!")
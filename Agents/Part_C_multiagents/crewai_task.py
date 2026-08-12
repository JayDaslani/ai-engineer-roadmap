from crewai import Agent, Task, Crew, Process, LLM
import os
from dotenv import load_dotenv

load_dotenv()

print("=== CrewAI Roles + Tasks ===")

llm = LLM(
    model="openai/llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    temperature=0.3
)

researcher = Agent(
    role="Senior research analyst",
    goal="""Research topics deeply and comprehensive,
    accurate information with sources.""",
    backstory="""You are a senior research analyst
    with 10 years experience.
    You specialized in technology and AI topics.
    You always verify facts and provide structured reports.""",
    llm=llm,
    verbose=False,
    allow_delegation=False
)

analyst = Agent(
    role='Data Analyst',
    goal="""Analyze research data and
    extract meaningful insights, trends and patterns.""",
    backstory="""You are an expert Data Analyst
    who transforms raw research into actionable insights.
    You identify trends,patterns and key takeways form complex information.""",
    llm = llm,
    verbose=False,
    allow_delegation=False
)

writer = Agent(
    role='Technical Writer',
    goal="""Create clear, engaging, and well-structured 
    technical content for various audience.""",
    backstory="""You are a technical writer
    who specializes in making complex topic accessible.
    You write in a clear, concise style with good structure and flow.""",
    llm=llm,
    verbose=False,
    allow_delegation=False
)

reviewer = Agent(
    role='Quality Reviewer',
    goal="""Review content for accuracy,
    clarity , completeness and provide improvement suggestions.""",
    backstory="""You are a Quality Reviewer with high standards.
    You ensure content is accurate, well-structured, and meets quality 
    benchmarks before final delivery.""",
    llm=llm,
    verbose=False,
    allow_delegation=False
)

TOPIC = "Artifical Intelligence in Healthcare"

research_task = Task(
    description = f"""
    Research this topic : '{TOPIC}'
    
    Cover:
    1. Current applications
    2. Key technologies used
    3. Benefits and Challenges
    4. Future Procpects
    5. Real world examples 
    
    Be specific and factual
    """,
    expected_output = """
    Comprehensive research report with:
    - 5+ key findings
    - Specefic examples
    - Data and statistics
    - Structured format
    """,
    agent=researcher
)

analysis_task = Task(
    description = f"""
    Analyze the research about : "{TOPIC}"
    
    Provide:
    1. Top 3 trends
    2. Key Opportunities
    3. Major challenges
    4. Competitive landscape
    5. Future Predictions
    
    Use the research data provided.
    """,
    expected_output="""
    Analysis report with :
    - Trends identified
    - Opportunities listed
    - Challenges explained
    - Future outlook
    """,
    agent=analyst,
    context=[research_task]
)

writing_task = Task(
    description = f"""
    Write a professional article about : {TOPIC}
    
    Structure:
    1.Excecutive Summary (2-3 lines)
    2.Introduction
    3.Current State
    4.Key Benefits
    5.Challenges
    6.Future Outlook
    7.Conclusion

    Use research and analysis provided.
    Make it engaging and informative.
    """,
    expected_output="""
    Well-written article of 500+ words
    with clear structure and flow.
    """,
    agent=writer,
    context = [research_task,analysis_task]

)

review_task =  Task(
    description="""
    Review the written article and provide:
    
    1. Quality store(1-10)
    2. Strengths (2-3 points)
    3. Areas to improve
    4. Final verdict: APPROVED or NEEDS REVISION
    
    Be constructive and specefic.
    """,
    expected_output="""
    Review report with:
    - Quality Score
    - Strenghts
    - Improvments
    - Final verdict
    """,
    agent= reviewer,
    context=[writing_task]
)

crew = Crew(
    agents=[researcher, analyst, writer, reviewer],
    tasks=[
        research_task,
        analysis_task,
        writing_task,
        review_task
    ],
    process=Process.sequential,
    verbose=False
)

print(f"Topic : {TOPIC}")
print("Starting Crew...")

result = crew.kickoff()

print("="*50)
print("Final Output: ")
print("="*50)
print(result)

def run_crew(topic):
    print("="*50)
    print(f"Topic : {topic}")
    print("="*50)

    r_task = Task(
        description=f"Research: {topic}."
                     f"Find key facts and examples.",
        expected_output = "Research report",
        agent=researcher
    )

    w_task = Task(
        description=f"Write article about : {topic}."
                     f"Use research provided.",
        expected_output="Article 300+ words",
        agent=writer,
        context=[r_task]
    )
    mini_crew = Crew(
        agents=[researcher, writer],
        tasks=[r_task,w_task],
        process=Process.sequential,
        verbose=False
    )

    result = mini_crew.kickoff()
    print(f"Result: {str(result)[:300]}")
    return result

run_crew("Python vs JavaScript for AI")
run_crew("Future of Remote Work")

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
import pandas as pd
import os
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

llm = ChatGroq(
    model='llama-3.3-70b-versatile',
    api_key=os.getenv('GROQ_API_KEY'),
    temperature=0.2
)

class DataState(TypedDict):
    csv_path: str
    data_info: str
    analysis: str
    insights: str
    recommendations: str

def load_data_node(state: DataState):
    """Loads a CSV file and extracts statistical metadata using Pandas."""
    csv_path = state['csv_path']
    print(f"Reading dataset from : {csv_path}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"csv file not found at path : {csv_path}")
    
    df = pd.read_csv(csv_path)

    rows, cols = df.shape
    missing_vals = df.isnull().sum().to_dict()
    dtypes_dict = df.dtypes.astype(str).to_dict()
    numeric_summary = df.describe().to_string()
    sample_rows = df.head(5).to_string()

    info_text = f"""
    --- DATASET OVERVIEW ---
    Total Rows: {rows}
    Total Columns: {cols}
    Column List : {df.columns.tolist()}

    --- DATA TYPES ---
    {dtypes_dict}

    --- MISSING VALUES PER COLUMN ---
    {missing_vals}

    --- NUMERICAL SUMMARY STATISTICS ---
    {numeric_summary}

    --- FIRST 5 SAMPLE ROWS ---
    {sample_rows}

    """
    print("   Data profile generated successfully!")
    return {"data_info": info_text}



def analyze_node(state: DataState):
    """The LLM inspects data scientist to analyze distribution, patterns, and trends."""
    print("Running quantiative & statistical analysis ...")

    prompt = f"""
    You are a expert Senior Data Scientist and Quantiative Analyst.
    Analyze the dataset profile below thoroughly :
    
    {state['data_info']}
    
    Provide a structured technical analysis covering:
    1. Data Quality & Completeness (Missing values, potential outliers)
    2. Distribution & Key metrics (Mean, median, variences observed)
    3. Major Trends and Cross-Column Relationships
    """

    response = llm.invoke(prompt)
    return {'analysis': response.content}

def insights_node(state: DataState):
    """Extract high-impact key takeways and core patterns from the analysis."""
    print("Extracting key insights & takeways...")

    prompt = f"""
    You are a Business Intelligence Lead.
    Based on the technical analysis below, extract the top 4-5 high-impact business insights:
    
    TECHNICAL ANALYSIS : 
    {state['analysis']}

    Format your response with :
    - Clear bullet points with bold headlines
    - Extract figures/metrics wherever applicable
    - Highlight high-performing areas vs potential risk factors
    """

    response = llm.invoke(prompt)
    return {"insights": response.content}

def recommend_node(state: DataState):
    """Generates actionable recommendations based on insights."""
    print("Formulating strategic recommendation...")

    prompt = f"""
    You are a Strategic Business Consultant.
    Based on these insights, provide actionable, prioritized recommendations: 
    
    KEY INSIGHTS : 
    {state['insights']}

    Provide:
    1. Immediate Actions (Short-term quick wins)
    2. Strategic Intiatives (Long-term growth / optimization)
    3. Metrics & KPIs to Monitor
    """

    response = llm.invoke(prompt)
    return {"recommendations": response.content}

graph = StateGraph(DataState)

graph.add_node('load', load_data_node)
graph.add_node('analyze', analyze_node)
graph.add_node('insights', insights_node)
graph.add_node('recommend', recommend_node)

graph.set_entry_point('load')

graph.add_edge('load', 'analyze')
graph.add_edge('analyze', 'insights')
graph.add_edge('insights', 'recommend')
graph.add_edge('recommend', END)

app = graph.compile()

def run_data_analysis(csv_path: str) -> dict:
    """Helper function to execute the full LangGraph data agent pipeline."""
    initial_state = {
        'csv_path': csv_path,
        'data_info': "",
        'analysis': "",
        'insights': "",
        'recommendations': ""
    }
    return app.invoke(initial_state)

print("=== Testing Data Agent Standalone ===")

sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_data')
os.makedirs(sample_dir, exist_ok=True)
sample_csv = os.path.join(sample_dir, 'sales.csv')

if not os.path.exists(sample_csv):
    dummy_df = pd.DataFrame({
        "Date": pd.date_range(start='2026-01-01', periods=10, freq='D'),
        "Product": ["Laptop", "Phone", "Tablet", "Laptop", "Monitor", "Phone", "Tablet", "Laptop", "Phone", "Monitor"],
        "Units_Sold": [12, 25, 8, 15, 10, 30, 12, 18, 22, 14],
        "Revenue": [12000, 15000, 3200, 15000, 3000, 18000, 4800, 18000, 13200, 4200]
    })
    dummy_df.to_csv(sample_csv, index=False)
    print(f"Created dummy sales data at: {sample_csv}")

result = run_data_analysis(sample_csv)

print('='*50)
print('data info:')
print('='*50)
print(result['data_info'])

print('='*50)
print('Analysis :')
print('='*50)
print(result['analysis'])

print('='*50)
print("KEY INSIGHTS:")
print('='*50)
print(result['insights'])

print('='*50)
print("RECOMMENDATION:")
print('='*50)
print(result["recommendations"])


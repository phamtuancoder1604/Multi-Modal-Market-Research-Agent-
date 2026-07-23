import sys
import os
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.dirname(current_dir))
import json
from typing import Literal
from langgraph.graph import StateGraph, START, END
from llama_index.llms.ollama import Ollama
from state import ResearchAgentState

from tools.web_tools import web_search_tool, web_scraper_tool
from tools.finance_tools import finance_metrics_tool
from tools.vision_tools import multimodal_vision_tool
from ingestion import ingest_unstructured_data, ingest_structured_data
from retrieval import retrieve_and_rerank

import asyncio
from pydantic import BaseModel, Field

# Mandatory structured output schema for the Editor Node
class EditorReviewSchema(BaseModel):
    decision: str = Field(description="Approval decision: 'approve' or 'revise'")
    remarks: str = Field(description="Detailed reasons or required revisions")
    citation_valid: bool = Field(description="True if all metrics have explicit tags like [Source X], False if unverified facts are present")
# Initialize the local LLM brain used for internal reasoning nodes
llm = Ollama(model="qwen2.5:7b", base_url="http://localhost:11434", request_timeout=150.0)

from langgraph.checkpoint.memory import MemorySaver
#  AGENT NODES IMPLEMENTATION


# def planner_node(state: ResearchAgentState) -> dict:
#     """
#     Analyzes the user request and breaks it down into a strategic sequence
#     of 2-3 specific search queries or ticker exploration tasks.
#     """
#     print("\n--- [ PLANNER AGENT] Generating Research Strategy ---")
#     prompt = (
#         f"You are an expert Research Planner. Break down this research query into 2-3 specific, "
#         f"focused search queries or corporate ticker analysis tasks for early 2026 data.\n"
#         f"User Query: {state['user_query']}\n"
#         f"Respond ONLY with a valid JSON list of strings. Example: [\"query 1\", \"query 2\"]"
#     )
    
#     response = llm.complete(prompt).text.strip()
#     try:
#         # Clean potential markdown wrapping if present in LLM response
#         if response.startswith("```json"):
#             response = response.split("```json")[1].split("```")[0].strip()
#         elif response.startswith("```"):
#             response = response.split("```")[1].split("```")[0].strip()
            
#         plan = json.loads(response)
#     except Exception:
#         # Fallback if JSON generation slips
#         plan = [f"{state['user_query']} market trends 2026", f"{state['user_query']} financial overview"]
        
#     return {"research_plan": plan, "executed_queries": [], "next_step": "multimodal"}
def planner_node(state: ResearchAgentState) -> dict:
    print("\n--- [PLANNER AGENT] Analyzing Query and Routing Tools ---")
    
    prompt = (
        f"Analyze this query. Is it related to business, market research, stocks, or tech industries?\n"
        f"Query: {state['user_query']}\n\n"
        f"Respond STRICTLY in JSON format:\n"
        f"If valid domain: {{\"is_valid\": true, \"required_tools\": [\"web\"], \"search_queries\": [\"query 1\"]}}\n"
        f"If out-of-domain/sports/recipes: {{\"is_valid\": false, \"required_tools\": [], \"search_queries\": []}}"
    )
    
    response = llm.complete(prompt).text.strip()
    try:
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        data = json.loads(response)
        
        if not data.get("is_valid", True):
            print("Out-of-domain query detected. Guardrail activated.")
            return {
                "research_plan": [],
                "executed_queries": [],
                "current_report": "I am a Financial & Market Research Assistant. This request is outside my research scope.",
                "next_step": "end" # Nhảy thẳng đến kết thúc, không chạy RAG hay Search
            }
            
        tools = data.get("required_tools", ["web"])
        queries = data.get("search_queries", [state['user_query']])
    except Exception:
        tools = ["web"]
        queries = [state['user_query']]

    next_step = "multimodal" if "vision" in tools and state.get("vision_data") else "researcher"
    return {"research_plan": queries, "executed_queries": tools, "next_step": next_step}
def multimodal_node(state: ResearchAgentState) -> dict:
    """
    Checks for uploaded images in the state, processes them via Vision LLM,
    and stores the descriptive Markdown tables and logic.
    """
    print("\n--- [VISION AGENT] Processing Multi-Modal Assets ---")
    vision_paths = state.get("vision_data", [])
    extracted_analyses = []

    for item in vision_paths:
        # Check if the item is a valid file path saved by Streamlit
        if os.path.exists(item):
            print(f"Extracting insights from image: {item}")
            analysis = multimodal_vision_tool(item)
            extracted_analyses.append(f"Extracted Image Data [{os.path.basename(item)}]:\n{analysis}")
        else:
            extracted_analyses.append(item)

    if not extracted_analyses:
        print("No valid images detected. Skipping vision analysis.")

    # Proceed to researcher node after vision extraction
    return {"vision_data": extracted_analyses, "next_step": "researcher"}

# def researcher_node(state: ResearchAgentState) -> dict:
#     """
#     Iterates over the plan, executes tools dynamically, processes data pipelines, 
#     and refines highly relevant context blocks through ChromaDB.
#     """
#     print("\n--- [ RESEARCHER AGENT] Executing Tools & Populating RAG ---")
#     plan = state.get("research_plan", [])
#     raw_texts = []
    
#     # Simple check to extract implicit tickers (e.g., "NVDA", "VFS")
#     query_upper = state["user_query"].upper()
#     ticker = "GENERIC"
#     for word in query_upper.split():
#         if len(word) <= 5 and word.isalpha() and word not in ["THE", "FOR", "AND", "WITH", "ASIA"]:
#             ticker = word
#             break

#     # 1. Execute Web Search for each query in the plan
#     for query in plan:
#         print(f"Executing web tool sequence for: {query}")
#         urls = web_search_tool(query)
#         # Mocking content fetching for demonstration - in production, fetch raw HTML here
#         for url in urls[:2]:
#             raw_texts.append(f"Source Data from {url}: Market context and industry tailwinds for early 2026.")
            
#     # 2. Execute Corporate Financial Tool
#     fin_data = ""
#     if ticker != "GENERIC":
#         fin_data = finance_metrics_tool(ticker)
#     else:
#         fin_data = json.dumps({"info": "No explicit ticker detected, skipping direct financial pull."})

#     # 3. Ingest collected raw artifacts into dual-index ChromaDB structures
#     for text in raw_texts:
#         ingest_unstructured_data(text, "https://automated-agent-scraper.vn")
        
#     if ticker != "GENERIC" and "error" not in fin_data:
#         ingest_structured_data(fin_data, ticker_symbol=ticker, topic_scope="financials")

#     # 4. Pull optimized contexts back out using the Parallel Reranking Pipeline
#     refined = retrieve_and_rerank(state["user_query"])
    
#     return {
#         "raw_web_data": raw_texts, 
#         "financial_data": [fin_data], 
#         "refined_contexts": refined,
#         "next_step": "writer"
#     }

async def async_research_pipeline(plan: list, ticker: str) -> tuple:
    """
    Executes web search, content scraping, and financial metrics gathering concurrently using asyncio.gather.
    """
    loop = asyncio.get_event_loop()
    
    # 1. Execute Web Search concurrently across planned queries
    search_tasks = [loop.run_in_executor(None, web_search_tool, q) for q in plan]
    search_results = await asyncio.gather(*search_tasks)
    
    urls = []
    for res in search_results:
        urls.extend(res[:2])
        
    # 2. Scrape Web Pages concurrently for retrieved URLs
    scrape_tasks = [web_scraper_tool(url) for url in urls]
    raw_texts = await asyncio.gather(*scrape_tasks)
    
    # 3. Retrieve corporate financial data asynchronously if ticker is identified
    fin_data = ""
    if ticker != "GENERIC":
        fin_data = await loop.run_in_executor(None, finance_metrics_tool, ticker)
        
    return raw_texts, fin_data


def researcher_node(state: ResearchAgentState) -> dict:
    """
    Executes tool pipelines concurrently and ingests processed data into ChromaDB.
    """
    print("\n--- [RESEARCHER AGENT] Executing Tools via Asynchronous Parallel Gather ---")
    plan = state.get("research_plan", [])
    
    # Simple ticker extraction heuristic
    query_words = state["user_query"].upper().split()
    ticker = "GENERIC"
    for word in query_words:
        if len(word) <= 5 and word.isalpha() and word not in ["THE", "FOR", "AND", "WITH", "EV", "CAR"]:
            ticker = word
            break

    # Trigger asynchronous pipeline
    raw_texts, fin_data = asyncio.run(async_research_pipeline(plan, ticker))

    # Ingest collected artifacts into ChromaDB
    for idx, text in enumerate(raw_texts, start=1):
        if text and not text.startswith("Error"):
            ingest_unstructured_data(f"[Source {idx}]:\n{text}", "[https://automated-agent-scraper.vn](https://automated-agent-scraper.vn)")
        
    if ticker != "GENERIC" and fin_data and "error" not in fin_data:
        ingest_structured_data(f"[Table 1 - {ticker} Financials]:\n{fin_data}", ticker_symbol=ticker, topic_scope="financials")

    # Retrieve and rerank context blocks
    refined = retrieve_and_rerank(state["user_query"])
    
    return {
        "raw_web_data": list(raw_texts), 
        "financial_data": [fin_data] if fin_data else [], 
        "refined_contexts": refined,
        "next_step": "writer"
    }

# def writer_node(state: ResearchAgentState) -> dict:
#     """
#     Synthesizes refined structural and textual contexts into a comprehensive Markdown report.
#     """
#     print("\n--- [ WRITER AGENT] Compiling Professional Report ---")
#     contexts_str = "\n\n".join(state.get("refined_contexts", ["No context retrieved."]))
#     vision_str = "\n\n".join(state.get("vision_data", []))
#     feedback = state.get("editor_feedback", {}).get("remarks", "None")
    
#     prompt = (
#         f"You are a Senior Financial Writer. Synthesize the provided context and any editor feedback "
#         f"into a pristine, professional Markdown report. Ensure all metrics align cleanly.\n"
#         f"User Original Intent: {state['user_query']}\n"
#         f"Retrieved Verified Context:\n{contexts_str}\n"
#         f"Extracted Image/Diagram Data:\n{vision_str}\n"
#         f"Editor Feedback to incorporate: {feedback}\n"
#         f"Output ONLY the finalized Markdown report content."
#     )
    
#     report = llm.complete(prompt).text.strip()
#     return {"current_report": report, "next_step": "editor"}


# def editor_node(state: ResearchAgentState) -> dict:
#     """
#     Acts as a quality guardrail to validate data alignment or authorize publication.
#     """
#     print("\n--- [ EDITOR AGENT] Evaluating Output Integrity ---")
#     report = state.get("current_report", "")
    
#     prompt = (
#         f"You are the Editor-in-Chief. Critically review this draft report. "
#         f"Check if it directly addresses the query and looks complete.\n"
#         f"Draft Report:\n{report}\n"
#         f"Respond ONLY in this explicit JSON format:\n"
#         f"{{\"decision\": \"approve\" or \"revise\", \"remarks\": \"detailed reason for decision\"}}"
#     )
    
#     response = llm.complete(prompt).text.strip()
#     try:
#         if response.startswith("```json"):
#             response = response.split("```json")[1].split("```")[0].strip()
#         elif response.startswith("```"):
#             response = response.split("```")[1].split("```")[0].strip()
            
#         review = json.loads(response)
#     except Exception:
#         review = {"decision": "approve", "remarks": "Fallback approval due to parser parsing."}
        
#     decision = review.get("decision", "approve")
#     return {"editor_feedback": review, "next_step": "end" if decision == "approve" else "writer"}
def writer_node(state: ResearchAgentState) -> dict:
    """
    Synthesizes retrieved context into a Markdown report enforcing strict citation masking.
    """
    print("\n--- [WRITER AGENT] Compiling Report with Strict Citation Masking ---")
    contexts_str = "\n\n".join(state.get("refined_contexts", ["No context retrieved."]))
    vision_str = "\n\n".join(state.get("vision_data", []))
    feedback = state.get("editor_feedback", {}).get("remarks", "None")
    
    prompt = (
        f"You are a Strict Financial Analyst. Synthesize a professional Markdown report using ONLY provided data.\n"
        f"STRICT RULE (CITATION MASKING):\n"
        f"Every single numeric value, claim, or percentage MUST have an explicit source tag immediately after it "
        f"(e.g., 'market share reached 32% [Source 1]' or 'revenue was $10B [Table 1]').\n"
        f"DO NOT invent facts or stats without citations!\n\n"
        f"User Intent: {state['user_query']}\n"
        f"Retrieved Text Context:\n{contexts_str}\n"
        f"Vision Data:\n{vision_str}\n"
        f"Editor Feedback to resolve: {feedback}\n"
    )
    
    report = llm.complete(prompt).text.strip()
    return {"current_report": report, "next_step": "editor"}


def editor_node(state: ResearchAgentState) -> dict:
    """
    Validates report citation coverage and overall response fidelity.
    """
    print("\n--- [EDITOR AGENT] Validating Citations and Quality ---")
    report = state.get("current_report", "")
    
    prompt = (
        f"You are the Editor-in-Chief. Check this report strictly:\n"
        f"1. Does every metric/number contain a citation tag like [Source X] or [Table X]?\n"
        f"2. Does it answer the user query accurately?\n\n"
        f"Report to inspect:\n{report}\n\n"
        f"Respond in JSON format with keys:\n"
        f"{{\"decision\": \"approve\" or \"revise\", \"citation_valid\": true or false, \"remarks\": \"reason\"}}"
    )
    
    response = llm.complete(prompt).text.strip()
    try:
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        review = json.loads(response)
    except Exception:
        review = {"decision": "approve", "citation_valid": True, "remarks": "Approved with fallback parser."}
        
    # Reject report if numerical claims lack citations
    if not review.get("citation_valid", True):
        review["decision"] = "revise"
        review["remarks"] += " (Rejected due to missing citation tags on numeric figures)."
        
    decision = review.get("decision", "approve")
    return {"editor_feedback": review, "next_step": "end" if decision == "approve" else "writer"}
#  CONDITIONAL ROUTING LOGIC


# def route_next_step(state: ResearchAgentState) -> Literal["researcher", "writer", "editor", "__end__"]:
#     """
#     Inspects state flags to determine the next graph node execution target.
#     """
#     status = state.get("next_step")
#     if status == "multimodal":
#         return "multimodal"
#     elif status == "researcher":
#         return "researcher"
#     elif status == "writer":
#         return "writer"
#     elif status == "editor":
#         return "editor"
#     return "end"
def route_next_step(state: ResearchAgentState) -> str:
    """
    Inspects state flags to determine the next graph node execution target.
    """
    status = state.get("next_step")
    if status == "multimodal":
        return "multimodal"
    elif status == "researcher":
        return "researcher"
    elif status == "writer":
        return "writer"
    elif status == "editor":
        return "editor"
    return "end"

# GRAPH ASSEMBLY

builder = StateGraph(ResearchAgentState)

# Register nodes
builder.add_node("planner", planner_node)
builder.add_node("multimodal", multimodal_node)
builder.add_node("researcher", researcher_node)
builder.add_node("writer", writer_node)
builder.add_node("editor", editor_node)

# Configure routing edges
builder.add_edge(START, "planner")
builder.add_conditional_edges(
    "planner",
    route_next_step,
    {
        "multimodal": "multimodal",
        "researcher": "researcher"
    }
)
builder.add_conditional_edges(
    "multimodal",
    route_next_step,
    {"researcher": "researcher"}
)

builder.add_conditional_edges(
    "researcher",
    route_next_step,
    {"writer": "writer"}
)

builder.add_conditional_edges(
    "writer",
    route_next_step,
    {"editor": "editor"}
)

builder.add_conditional_edges(
    "editor",
    route_next_step,
    {
        "writer": "writer",
        "end": END
    }
)


memory = MemorySaver()
# Compile graph into executable runtime application
research_agent_graph = builder.compile(checkpointer=memory)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "session_1"}}
    # Test execution simulation run
    initial_inputs = {
        "user_query": "Analyze Vietnam electric vehicle market trends for 2026",
        "research_plan": [],
        "executed_queries": [],
        "raw_web_data": [],
        "financial_data": [],
        "vision_data": [],
        "refined_contexts": [],
        "current_report": "",
        "editor_feedback": {},
        "next_step": ""
    }
    
    print("Launching Multi-Agent Research System...")
    final_output = research_agent_graph.invoke(initial_inputs, config=config)
    ## delete after test
    print("\n--- BÁO CÁO LƯỢT 1 ---")
    print(final_output.get("current_report")[:300] + "...\n[Đoạn sau đã ẩn bớt]")
    
    print("\n [Lượt 2] Người dùng yêu cầu chỉnh sửa dựa trên hội thoại cũ...")
    
    # Gửi câu hỏi mới mà không cần truyền lại các trường data trống, chỉ cần đúng thread_id
    follow_up_inputs = {
        "user_query": "Add a short warning about the risk of charging station overload to the report above.",
        "next_step": ""  # Reset cờ để quay lại Planner điều phối
    }
    follow_up_output = research_agent_graph.invoke(follow_up_inputs, config=config)
    
    print(" FINAL APPROVED RESEARCH REPORT GENERATED:")
    print(follow_up_output.get("current_report"))
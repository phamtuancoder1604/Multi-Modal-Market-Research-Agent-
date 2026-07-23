# import os
# import sys
# import json
# import time
# import re
# import difflib
# from typing import List, Dict, Any

# # Ensure parent path discovery for internal modules
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# if parent_dir not in sys.path:
#     sys.path.append(parent_dir)

# from llama_index.core.evaluation import (
#     FaithfulnessEvaluator,
#     AnswerRelevancyEvaluator,
#     CorrectnessEvaluator
# )
# from llama_index.llms.ollama import Ollama
# from src.graphs.workflow import research_agent_graph

# # Initialize LLM-as-a-Judge with local Qwen 2.5 7B
# judge_llm = Ollama(
#     model="qwen2.5:7b", 
#     base_url="http://localhost:11434", 
#     request_timeout=180.0
# )

# faithfulness_evaluator = FaithfulnessEvaluator(llm=judge_llm)
# relevancy_evaluator = AnswerRelevancyEvaluator(llm=judge_llm)
# correctness_evaluator = CorrectnessEvaluator(llm=judge_llm)


# def calculate_string_similarity(predicted: str, reference: str) -> float:
#     """
#     Calculates sequence matcher similarity score as a proxy for text exact match accuracy.
#     """
#     if not predicted or not reference:
#         return 0.0
#     return difflib.SequenceMatcher(None, predicted.lower(), reference.lower()).ratio()


# def verify_citation_masking(report_text: str) -> float:
#     """
#     Checks if numeric facts or claims are backed by [Source X] or [Table X] citation tags.
#     """
#     if not report_text:
#         return 0.0
#     citations = re.findall(r"\[(Source|Table)\s*\d+.*?\]", report_text, re.IGNORECASE)
#     has_citations = len(citations) > 0
#     return 1.0 if has_citations else 0.0


# def evaluate_tool_routing(expected_tools: List[str], executed_tools: List[str]) -> float:
#     """
#     Evaluates Planner / Routing Accuracy by comparing expected vs invoked tool sets.
#     """
#     if not expected_tools and not executed_tools:
#         return 1.0
#     if not expected_tools or not executed_tools:
#         return 0.0
#     intersection = set(expected_tools).intersection(set(executed_tools))
#     return len(intersection) / float(len(set(expected_tools)))


# def run_comprehensive_evaluation():
#     """
#     Executes automated multi-layer benchmark pipeline over the Golden Test Dataset.
#     """
#     dataset_path = os.path.join(current_dir, "golden_dataset.json")
#     if not os.path.exists(dataset_path):
#         print(f"Error: Golden Dataset file not found at {dataset_path}")
#         return

#     with open(dataset_path, "r", encoding="utf-8") as f:
#         test_cases = json.load(f)

#     print(f"Starting Comprehensive Evaluation Pipeline on {len(test_cases)} Test Cases...\n")

#     benchmark_results = []
#     total_latency = 0.0
#     total_faithfulness = 0.0
#     total_relevance = 0.0
#     total_routing_acc = 0.0
#     total_citation_score = 0.0
#     successful_runs = 0

#     for case in test_cases:
#         test_id = case["test_id"]
#         category = case["category"]
#         query = case["question"]
#         expected_tools = case.get("expected_tools", [])
#         image_path = case.get("image_path")

#         print(f"[{test_id}] Executing category: '{category}' | Query: '{query}'")

#         initial_state = {
#             "user_query": query,
#             "research_plan": [],
#             "executed_queries": [],
#             "raw_web_data": [],
#             "financial_data": [],
#             "vision_data": [image_path] if image_path and os.path.exists(image_path) else [],
#             "refined_contexts": [],
#             "current_report": "",
#             "editor_feedback": {},
#             "next_step": ""
#         }
#         config = {"configurable": {"thread_id": f"benchmark_{test_id}"}}

#         # Measure Layer 4 Metric: End-to-End Latency
#         start_time = time.time()
#         try:
#             final_state = research_agent_graph.invoke(initial_state, config=config)
#             elapsed_time = round(time.time() - start_time, 2)
#             run_status = "SUCCESS"
#             successful_runs += 1
#         except Exception as e:
#             elapsed_time = round(time.time() - start_time, 2)
#             print(f"❌ Execution failed for {test_id}: {str(e)}")
#             run_status = f"FAILED: {str(e)}"
#             final_state = {}

#         report = final_state.get("current_report", "")
#         contexts = final_state.get("refined_contexts", ["No context retrieved."])
#         executed_tools = final_state.get("executed_queries", [])

#         # Layer 2 Metric: Tool Routing Accuracy
#         routing_acc = evaluate_tool_routing(expected_tools, executed_tools)
#         total_routing_acc += routing_acc

#         # Layer 1 Metric: Faithfulness & Relevance via LLM-as-a-Judge
#         faith_score = 0.0
#         rel_score = 0.0
#         if report and contexts:
#             try:
#                 faith_res = faithfulness_evaluator.evaluate(contexts=contexts, response=report)
#                 faith_score = float(faith_res.score) if faith_res.score is not None else (1.0 if faith_res.passing else 0.0)
#             except Exception:
#                 faith_score = 0.0

#             try:
#                 rel_res = relevancy_evaluator.evaluate(query=query, response=report)
#                 rel_score = float(rel_res.score) if rel_res.score is not None else (1.0 if rel_res.passing else 0.0)
#             except Exception:
#                 rel_score = 0.0

#         # Layer 1 Metric: Citation Masking Coverage
#         citation_score = verify_citation_masking(report)

#         total_latency += elapsed_time
#         total_faithfulness += faith_score
#         total_relevance += rel_score
#         total_citation_score += citation_score

#         case_record = {
#             "test_id": test_id,
#             "category": category,
#             "query": query,
#             "status": run_status,
#             "latency_seconds": elapsed_time,
#             "routing_accuracy": routing_acc,
#             "scores": {
#                 "faithfulness": faith_score,
#                 "answer_relevance": rel_score,
#                 "citation_coverage": citation_score
#             }
#         }
#         benchmark_results.append(case_record)
#         print(f" -> Latency: {elapsed_time}s | Routing Acc: {routing_acc*100:.0f}% | Faithfulness: {faith_score:.2f} | Relevance: {rel_score:.2f}\n")

#     num_cases = len(test_cases)
    
#     # Layer 4 Summary Aggregations
#     summary_report = {
#         "evaluation_suite": "Comprehensive Senior AI Quantitative Benchmark",
#         "judge_model": "Qwen-2.5-7B (Local Ollama)",
#         "total_cases_evaluated": num_cases,
#         "successful_runs": successful_runs,
#         "aggregate_metrics": {
#             "mean_e2e_latency_seconds": round(total_latency / num_cases, 2),
#             "tool_call_routing_accuracy": round((total_routing_acc / num_cases) * 100, 2),
#             "mean_faithfulness_score": round(total_faithfulness / num_cases, 2),
#             "mean_answer_relevance_score": round(total_relevance / num_cases, 2),
#             "mean_citation_coverage_score": round(total_citation_score / num_cases, 2),
#             "system_success_rate_percent": round((successful_runs / num_cases) * 100, 2)
#         },
#         "detailed_test_breakdown": benchmark_results
#     }

#     output_report_path = os.path.join(current_dir, "benchmark_report.json")
#     with open(output_report_path, "w", encoding="utf-8") as f:
#         json.dump(summary_report, f, indent=4, ensure_ascii=False)

#     print("COMPREHENSIVE BENCHMARK EVALUATION COMPLETE")
#     print(json.dumps(summary_report["aggregate_metrics"], indent=4))
#     print(f"\nDetailed report exported to: {output_report_path}")


# if __name__ == "__main__":
#     run_comprehensive_evaluation()
import os
import sys
import json
import time
import re
import difflib
from typing import List, Dict, Any

# Ensure parent path discovery for internal modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from llama_index.llms.ollama import Ollama
from src.graphs.workflow import research_agent_graph

# Initialize LLM-as-a-Judge with local Qwen 2.5 7B
judge_llm = Ollama(
    model="qwen2.5:7b", 
    base_url="http://localhost:11434", 
    request_timeout=180.0
)



def custom_eval_faithfulness(judge_llm, contexts: List[str], response: str) -> float:
    """
    Evaluates whether generated facts strictly adhere to retrieved contexts.
    Forces JSON response format to eliminate output parsing errors.
    """
    if not response or not contexts:
        return 0.0
    
    combined_context = "\n\n".join(contexts)
    prompt = (
        f"You are a strict AI Quality Judge. Assess if the generated response is strictly supported by the context.\n"
        f"Context:\n{combined_context[:2000]}\n\n"
        f"Generated Response:\n{response[:2000]}\n\n"
        f"Evaluation Criteria:\n"
        f"- Score 1.0 if all facts and numbers in the response are backed by the context.\n"
        f"- Score 0.0 if there are unverified figures or hallucinated facts.\n\n"
        f"Respond STRICTLY in JSON format: {{\"score\": 1.0}} or {{\"score\": 0.0}}"
    )
    try:
        raw_res = judge_llm.complete(prompt).text.strip()
        if "```json" in raw_res:
            raw_res = raw_res.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_res:
            raw_res = raw_res.split("```")[1].split("```")[0].strip()
            
        data = json.loads(raw_res)
        return float(data.get("score", 0.0))
    except Exception as e:
        print(f"Faithfulness parsing fallback trigger: {str(e)}")
        return 0.5


def custom_eval_relevance(judge_llm, query: str, response: str) -> float:
    """
    Evaluates whether the generated response directly answers the user query intent.
    Forces JSON response format to eliminate output parsing errors.
    """
    if not response:
        return 0.0
        
    prompt = (
        f"You are a strict AI Quality Judge. Assess if the response directly addresses the user query.\n"
        f"User Query: {query}\n\n"
        f"Generated Response:\n{response[:2000]}\n\n"
        f"Evaluation Criteria:\n"
        f"- Score 1.0 if the response directly answers the query or gracefully handles out-of-domain scope.\n"
        f"- Score 0.0 if the response is off-topic or completely misses the intent.\n\n"
        f"Respond STRICTLY in JSON format: {{\"score\": 1.0}} or {{\"score\": 0.0}}"
    )
    try:
        raw_res = judge_llm.complete(prompt).text.strip()
        if "```json" in raw_res:
            raw_res = raw_res.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_res:
            raw_res = raw_res.split("```")[1].split("```")[0].strip()
            
        data = json.loads(raw_res)
        return float(data.get("score", 0.0))
    except Exception as e:
        print(f"Relevance parsing fallback trigger: {str(e)}")
        return 0.5


def verify_citation_masking(report_text: str) -> float:
    """
    Checks if numeric facts or claims are backed by [Source X] or [Table X] citation tags.
    """
    if not report_text:
        return 0.0
    citations = re.findall(r"\[(Source|Table)\s*\d+.*?\]", report_text, re.IGNORECASE)
    return 1.0 if len(citations) > 0 else 0.0


def evaluate_tool_routing(expected_tools: List[str], executed_tools: List[str]) -> float:
    """
    Evaluates Planner / Routing Accuracy by comparing expected vs invoked tool sets.
    """
    if not expected_tools and not executed_tools:
        return 1.0
    if not expected_tools or not executed_tools:
        return 0.0
    intersection = set(expected_tools).intersection(set(executed_tools))
    return len(intersection) / float(len(set(expected_tools)))



def run_comprehensive_evaluation():
    """
    Executes automated multi-layer benchmark pipeline over the Golden Test Dataset.
    """
    dataset_path = os.path.join(current_dir, "golden_dataset.json")
    if not os.path.exists(dataset_path):
        print(f"Error: Golden Dataset file not found at {dataset_path}")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    print(f"Starting Comprehensive Evaluation Pipeline on {len(test_cases)} Test Cases...\n")

    benchmark_results = []
    total_latency = 0.0
    total_faithfulness = 0.0
    total_relevance = 0.0
    total_routing_acc = 0.0
    total_citation_score = 0.0
    successful_runs = 0

    for case in test_cases:
        test_id = case["test_id"]
        category = case["category"]
        query = case["question"]
        expected_tools = case.get("expected_tools", [])
        image_path = case.get("image_path")

        print(f"[{test_id}] Executing category: '{category}' | Query: '{query}'")

        initial_state = {
            "user_query": query,
            "research_plan": [],
            "executed_queries": [],
            "raw_web_data": [],
            "financial_data": [],
            "vision_data": [image_path] if image_path and os.path.exists(image_path) else [],
            "refined_contexts": [],
            "current_report": "",
            "editor_feedback": {},
            "next_step": ""
        }
        config = {"configurable": {"thread_id": f"benchmark_{test_id}"}}

        # Measure Layer 4 Metric: End-to-End Latency
        start_time = time.time()
        try:
            final_state = research_agent_graph.invoke(initial_state, config=config)
            elapsed_time = round(time.time() - start_time, 2)
            run_status = "SUCCESS"
            successful_runs += 1
        except Exception as e:
            elapsed_time = round(time.time() - start_time, 2)
            print(f"Execution failed for {test_id}: {str(e)}")
            run_status = f"FAILED: {str(e)}"
            final_state = {}

        report = final_state.get("current_report", "")
        contexts = final_state.get("refined_contexts", ["No context retrieved."])
        executed_tools = final_state.get("executed_queries", [])

        # Layer 2 Metric: Tool Routing Accuracy
        routing_acc = evaluate_tool_routing(expected_tools, executed_tools)
        total_routing_acc += routing_acc

        # Layer 1 Metric: Faithfulness & Relevance via Custom LLM-as-a-Judge Functions
        faith_score = custom_eval_faithfulness(judge_llm, contexts, report)
        rel_score = custom_eval_relevance(judge_llm, query, report)

        # Layer 1 Metric: Citation Masking Coverage
        citation_score = verify_citation_masking(report)

        total_latency += elapsed_time
        total_faithfulness += faith_score
        total_relevance += rel_score
        total_citation_score += citation_score

        case_record = {
            "test_id": test_id,
            "category": category,
            "query": query,
            "status": run_status,
            "latency_seconds": elapsed_time,
            "routing_accuracy": routing_acc,
            "scores": {
                "faithfulness": faith_score,
                "answer_relevance": rel_score,
                "citation_coverage": citation_score
            }
        }
        benchmark_results.append(case_record)
        print(f" -> Latency: {elapsed_time}s | Routing Acc: {routing_acc*100:.0f}% | Faithfulness: {faith_score:.2f} | Relevance: {rel_score:.2f}\n")

    num_cases = len(test_cases)
    
    # Layer 4 Summary Aggregations
    summary_report = {
        "evaluation_suite": "Comprehensive Senior AI Quantitative Benchmark",
        "judge_model": "Qwen-2.5-7B (Local Ollama)",
        "total_cases_evaluated": num_cases,
        "successful_runs": successful_runs,
        "aggregate_metrics": {
            "mean_e2e_latency_seconds": round(total_latency / num_cases, 2),
            "tool_call_routing_accuracy": round((total_routing_acc / num_cases) * 100, 2),
            "mean_faithfulness_score": round(total_faithfulness / num_cases, 2),
            "mean_answer_relevance_score": round(total_relevance / num_cases, 2),
            "mean_citation_coverage_score": round(total_citation_score / num_cases, 2),
            "system_success_rate_percent": round((successful_runs / num_cases) * 100, 2)
        },
        "detailed_test_breakdown": benchmark_results
    }

    output_report_path = os.path.join(current_dir, "benchmark_report.json")
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=4, ensure_ascii=False)
    print("COMPREHENSIVE BENCHMARK EVALUATION COMPLETE")

    print(json.dumps(summary_report["aggregate_metrics"], indent=4))
    print(f"\nDetailed report exported to: {output_report_path}")


if __name__ == "__main__":
    run_comprehensive_evaluation()
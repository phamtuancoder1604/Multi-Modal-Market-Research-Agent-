import os
import sys
import json
import asyncio
from llama_index.core.prompts import PromptTemplate
# 1. Cấu hình hệ thống đường dẫn để Python nhận diện module src
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Nạp bộ ba Evaluator chính thống từ LlamaIndex
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    AnswerRelevancyEvaluator,
    CorrectnessEvaluator
)
from llama_index.llms.ollama import Ollama
# Nạp đồ thị LangGraph thực tế của bạn
from src.graphs.workflow import research_agent_graph

# Khởi tạo mô hình Trọng tài Local AI
evaluator_llm = Ollama(model="qwen2.5:7b", base_url="http://localhost:11434", request_timeout=180.0)
CUSTOM_FAITHFULNESS_PROMPT = PromptTemplate(
    "You are an expert evaluator. Tell me if the specific information provided is directly supported by the context.\n"
    "Context:\n{context_str}\n\n"
    "Information:\n{query_str}\n\n"
    "Does the information strictly match the context without making things up? Answer YES or NO, followed by a short explanation."
)
# Khởi tạo các thực thể Evaluator của LlamaIndex
faithfulness_evaluator = FaithfulnessEvaluator(llm=evaluator_llm, eval_template=CUSTOM_FAITHFULNESS_PROMPT)
relevancy_evaluator = AnswerRelevancyEvaluator(llm=evaluator_llm)
correctness_evaluator = CorrectnessEvaluator(llm=evaluator_llm)

def run_full_system_evaluation():
    print("Starting Full End-to-End Pipeline Evaluation via LlamaIndex Evaluators...")
    
    # Định nghĩa bộ Case dữ liệu kiểm thử (Vàng)
    test_case = {
        "question": "Analyze Vietnam electric vehicle market trends for 2026",
        "ground_truth": "Vietnam electric vehicle sales surged in early 2026 due to domestic battery infrastructure expansions, with market share rising from 15% in 2025 to 32% in 2026."
    }
    
    # Thiết lập trạng thái kích hoạt đồ thị ban đầu
    initial_state = {
        "user_query": test_case["question"],
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
    config = {"configurable": {"thread_id": "llama_index_eval_session_2026"}}
    
    print(f"\n[1/3] Triggering real LangGraph execution for query: '{test_case['question']}'")
    print("Executing web search tools, context retrieval nodes, and report compiler...")
    
    # 2. CHẠY ĐỒ THỊ ĐỂ LẤY KẾT QUẢ THỰC TẾ
    final_state = research_agent_graph.invoke(initial_state, config=config)
    
    generated_report = final_state.get("current_report", "")
    retrieved_nodes = final_state.get("refined_contexts", [])
    
    # Chuẩn hóa ép dữ liệu Contexts về dạng chuỗi văn bản phẳng (String)
    string_contexts = [str(node) for node in retrieved_nodes] if retrieved_nodes else ["No context retrieved."]
    combined_context_str = "\n\n".join(string_contexts)
    
    print("\n[2/3] Agent Graph invocation complete!")
    print(f"-> Generated Report Length: {len(generated_report)} characters.")
    print(f"-> Source Context Chunks: {len(string_contexts)} blocks loaded.")
    
    if not generated_report:
        print("Error: Pipeline produced an empty report. Aborting scoring matrix.")
        return

    print("\n[3/3] Executing LlamaIndex semantic validation scoring matrix...")

    # 3. THỰC HIỆN CHẤM ĐIỂM BẰNG CÁC EVALUATOR CHÍNH THỐNG
    # LlamaIndex Evaluator trả về một đối tượng EvaluationResult chứa .passing, .score và .feedback
    
    # A. Chấm độ trung thực (So sánh Report sinh ra vs Context thô được cào về)
    faith_result = faithfulness_evaluator.evaluate(
        contexts=string_contexts,
        response=generated_report
    )
    
    # B. Chấm độ gắn kết (So sánh câu hỏi đầu vào vs Report sinh ra)
    relevancy_result = relevancy_evaluator.evaluate(
        query=test_case["question"],
        response=generated_report
    )
    
    # C. Chấm độ chính xác hoàn chỉnh (Đối chiếu: Question + Report sinh ra vs Ground Truth đáp án chuẩn)
    correctness_result = correctness_evaluator.evaluate(
        query=test_case["question"],
        response=generated_report,
        reference=test_case["ground_truth"]
    )
    
    # 4. ĐÓNG GÓI BẢNG ĐIỂM ĐỊNH LƯỢNG CUỐI CÙNG
    evaluation_matrix = {
        "input_question": test_case["question"],
        "metrics": {
            "faithfulness": {
                "score": float(faith_result.score) if faith_result.score is not None else (1.0 if faith_result.passing else 0.0),
                "feedback": faith_result.feedback
            },
            "answer_relevance": {
                "score": float(relevancy_result.score) if relevancy_result.score is not None else (1.0 if relevancy_result.passing else 0.0),
                "feedback": relevancy_result.feedback
            },
            "correctness": {
                "score": float(correctness_result.score) if correctness_result.score is not None else 0.0,
                "feedback": correctness_result.feedback
            }
        },
        "pipeline_status": "LLAMA_INDEX_E2E_SUCCESS"
    }

    print(" REAL END-TO-END PIPELINE EVALUATION RESULTS:")

    print(json.dumps(evaluation_matrix, indent=4, ensure_ascii=False))

if __name__ == "__main__":
    run_full_system_evaluation()
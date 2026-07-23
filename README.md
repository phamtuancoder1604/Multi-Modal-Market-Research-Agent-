+------------------------+
              |    User Query / Image   |
              +-----------+------------+
                          |
                          v
              +------------------------+
              |     Planner Agent      |
              |  (Query Expansion &    |
              |   Dynamic Routing)     |
              +---+-------+--------+---+
                  |       |        |
     +------------+       |        +------------+
     | (If Vision)        | (If Web)            | (If Finance)
     v                    v                     v
+------------------+ +-----------------+ +-------------------+
| Multi-Modal Node | | Researcher Node | | Financial API Tool|
| (Vision Parser)  | | (Async Scraping)| | (yfinance/Metrics)|
+--------+---------+ +--------+--------+ +---------+---------+
|                    |                    |
+--------------------+--------------------+
|
v
+------------------------+
| Hybrid Retrieval (RAG) |
|  ChromaDB + BM25 +     |
|   Local LLM Rerank     |
+-----------+------------+
|
v
+------------------------+
|      Writer Agent      |
|  (Report Synthesis &   |
|   Citation Masking)    |
+-----------+------------+
|
v
+------------------------+
|      Editor Agent      |
|  (Quality & Guardrail  |
|      Validation)       |
+-----------+------------+
|
v
+------------------------+
|    Final Report UI     |
+------------------------+


---

## Directory Structure

advanced_research_agent/
│
├── src/
│   ├── app.py                   # Streamlit Dashboard UI
│   ├── retrieval.py             # Hybrid RAG Pipeline (ChromaDB + BM25 + Reranker)
│   ├── graphs/
│   │   └── workflow.py          # LangGraph Multi-Agent StateGraph Architecture
│   └── tools/
│       ├── web_tools.py         # Web Search & Playwright Scraper
│       └── financial_tools.py   # yfinance Stock & Financial Metric Extractors
│
├── evaluation/
│   ├── golden_dataset.json      # 30 Comprehensive Test Cases Across 4 Categories
│   ├── benchmark_pipeline.py    # Automated LLM-as-a-Judge Evaluation Script
│   └── benchmark_report.json    # Quantitative Evaluation Output Results
│
├── data/                        # Persistent Vector Indices & Raw Storage
├── requirements.txt             # Python Package Dependencies
└── README.md                    # Project Documentation


---

## Getting Started

### Prerequisites

1. **Python 3.11+** installed on your system.
2. **Ollama** installed and running locally.
3. Pull required local models:

```bash
ollama pull qwen2.5:3b
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
ollama pull llava
Installation
Clone the repository:

Bash
git clone [https://github.com/your-username/advanced_research_agent.git](https://github.com/your-username/advanced_research_agent.git)
cd advanced_research_agent
Create and activate a virtual environment:

Bash
python -m venv venv
# On Windows PowerShell:
.\\venv\\Scripts\\Activate.ps1
# On Linux/macOS:
source venv/bin/activate
Install dependencies:

Bash
pip install -r requirements.txt
playwright install
Usage
1. Running the Web Application (Streamlit UI)
Launch the interactive dashboard:

Bash
cd src
streamlit run app.py
Open http://localhost:8501 in your browser to interact with the research assistant and upload charts/diagrams via the sidebar.

2. Running Workflow Directly in Terminal
Bash
python src/graphs/workflow.py
3. Running Benchmark Evaluation Suite
To execute the automated quantitative benchmark across the 30-case dataset:

Bash
python evaluation/benchmark_pipeline.py
Results will be automatically printed and saved to evaluation/benchmark_report.json.

Quantitative Evaluation Results
Evaluation conducted across 30 test cases using local Qwen-2.5-7B as LLM-as-a-Judge:

System Execution Success Rate: 83.33%

Mean E2E Latency: 142.15 seconds

Tool Routing Accuracy: 56.67%

Mean Faithfulness Score: 0.50

Mean Citation Coverage: 70.0%

License
Distributed under the MIT License. See LICENSE for more information.
"""

with open("README.md", "w", encoding="utf-8") as f:
f.write(readme_content)

print("README.md successfully created.")

File `README.md` bằng tiếng Anh chuẩn dự án AI Engineer của bạn đã sẵn sàng.

[file-tag: code-generated-file-0-1784799091506283452]

---

### Nội dung xem trước của file `README.md`:

```markdown
# Autonomous Multi-Modal Market Research Agent

An enterprise-grade, privacy-first **Multi-Agent Market Research System** built with **LangGraph**, **LlamaIndex**, and **Ollama**. The system dynamically orchestrates specialized AI agents to collect real-time web data, query financial metrics, parse visual diagrams/charts, perform Hybrid RAG retrieval, and compile fact-grounded research reports with automated citation masking.

---

## Key Features

* **Multi-Agent StateGraph Architecture:** Orchestrated via **LangGraph** with dynamic conditional routing, state persistence (`MemorySaver`), and human-in-the-loop review capabilities.
* **Hybrid RAG Pipeline:** Combines **Dense Vector Search (ChromaDB)** and **Sparse Keyword Search (BM25)** with **Local LLM Reranking** to maintain precise retrieval for both semantic queries and exact financial tickers/figures.
* **100% Privacy-First & Offline Runtime:** Powered locally via **Ollama** using quantized models (`Qwen 2.5 3B/7B`, `LLaVA` / `Qwen2-VL`, `Nomic-Embed-Text`), eliminating API costs and third-party data leakage risks.
* **Multi-Modal Data Processing:** Integrates vision tools to parse complex supply chain diagrams, financial trend charts, and structural flowcharts uploaded by users.
* **Strict Citation Masking (`[Source X]`):** Enforces fact-grounding by attaching explicit source citations to numerical figures and claims, significantly reducing hallucination risks.
* **Automated Quantitative Benchmarking:** Includes a custom **30-case benchmark suite** utilizing an **LLM-as-a-Judge** framework to measure E2E Latency, Tool Routing Accuracy, Faithfulness, and Answer Relevance.

---

## Tech Stack

| Category | Technologies |
| :--- | :--- |
| **Agent Frameworks** | LangGraph, LlamaIndex, Ollama (Local Runtime) |
| **Local Models** | Qwen 2.5 (3B / 7B), LLaVA / Qwen2-VL, Nomic-Embed-Text |
| **RAG & Search** | ChromaDB (Vector), BM25/BM25S (Keyword), LLMRerank |
| **Data Tools** | Playwright (Headless Web Scraping), BeautifulSoup4, DuckDuckGo API, yfinance |
| **UI & Execution** | Streamlit, Python 3.12+, Asyncio, Pydantic |

---

## Project Architecture & Multi-Agent Flow

              +------------------------+
              |    User Query / Image   |
              +-----------+------------+
                          |
                          v
              +------------------------+
              |     Planner Agent      |
              |  (Query Expansion &    |
              |   Dynamic Routing)     |
              +---+-------+--------+---+
                  |       |        |
     +------------+       |        +------------+
     | (If Vision)        | (If Web)            | (If Finance)
     v                    v                     v
+------------------+ +-----------------+ +-------------------+
| Multi-Modal Node | | Researcher Node | | Financial API Tool|
| (Vision Parser)  | | (Async Scraping)| | (yfinance/Metrics)|
+--------+---------+ +--------+--------+ +---------+---------+
|                    |                    |
+--------------------+--------------------+
|
v
+------------------------+
| Hybrid Retrieval (RAG) |
|  ChromaDB + BM25 +     |
|   Local LLM Rerank     |
+-----------+------------+
|
v
+------------------------+
|      Writer Agent      |
|  (Report Synthesis &   |
|   Citation Masking)    |
+-----------+------------+
|
v
+------------------------+
|      Editor Agent      |
|  (Quality & Guardrail  |
|      Validation)       |
+-----------+------------+
|
v
+------------------------+
|    Final Report UI     |
+------------------------+


---

## Directory Structure

advanced_research_agent/
│
├── src/
│   ├── app.py                   # Streamlit Dashboard UI
│   ├── retrieval.py             # Hybrid RAG Pipeline (ChromaDB + BM25 + Reranker)
│   ├── graphs/
│   │   └── workflow.py          # LangGraph Multi-Agent StateGraph Architecture
│   └── tools/
│       ├── web_tools.py         # Web Search & Playwright Scraper
│       └── financial_tools.py   # yfinance Stock & Financial Metric Extractors
│
├── evaluation/
│   ├── golden_dataset.json      # 30 Comprehensive Test Cases Across 4 Categories
│   ├── benchmark_pipeline.py    # Automated LLM-as-a-Judge Evaluation Script
│   └── benchmark_report.json    # Quantitative Evaluation Output Results
│
├── data/                        # Persistent Vector Indices & Raw Storage
├── requirements.txt             # Python Package Dependencies
└── README.md                    # Project Documentation


---

## Getting Started

### Prerequisites

1. **Python 3.11+** installed on your system.
2. **Ollama** installed and running locally.
3. Pull required local models:

```bash
ollama pull qwen2.5:3b
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
ollama pull llava
Installation
Clone the repository:

Bash
git clone [https://github.com/your-username/advanced_research_agent.git](https://github.com/your-username/advanced_research_agent.git)
cd advanced_research_agent
Create and activate a virtual environment:

Bash
python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate
Install dependencies:

Bash
pip install -r requirements.txt
playwright install
Usage
1. Running the Web Application (Streamlit UI)
Launch the interactive dashboard:

Bash
cd src
streamlit run app.py
Open http://localhost:8501 in your browser to interact with the research assistant and upload charts/diagrams via the sidebar.

2. Running Workflow Directly in Terminal
Bash
python src/graphs/workflow.py
3. Running Benchmark Evaluation Suite
To execute the automated quantitative benchmark across the 30-case dataset:

Bash
python evaluation/benchmark_pipeline.py
Results will be automatically printed and saved to evaluation/benchmark_report.json.

Quantitative Evaluation Results
Evaluation conducted across 30 test cases using local Qwen-2.5-7B as LLM-as-a-Judge:

System Execution Success Rate: 83.33%

Mean E2E Latency: 142.15 seconds

Tool Routing Accuracy: 56.67%

Mean Faithfulness Score: 0.50

Mean Citation Coverage: 70.0%

License
Distributed under the MIT License. See LICENSE for more information.
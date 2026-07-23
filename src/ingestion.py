import os
import chromadb
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import HierarchicalNodeParser, MarkdownElementNodeParser
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
# Initialize local embedding and LLM configurations
embed_model = OllamaEmbedding(model_name="nomic-embed-text", base_url="http://localhost:11434")
llm = Ollama(model="qwen2.5:7b", base_url="http://localhost:11434", request_timeout=120.0)

# Initialize persistent ChromaDB local client
db_path = os.path.join("data", "vector_db")
chroma_client = chromadb.PersistentClient(path=db_path)

def ingest_unstructured_data(text_content: str, source_url: str) -> None:
    """
    Applies Hierarchical/Parent-Child Chunking on raw web text and saves to Index_Unstructured.
    """
    print("Processing unstructured data pipeline...")
    
    # Create multi-index collection inside ChromaDB
    chroma_collection = chroma_client.get_or_create_collection("index_unstructured")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Wrap text content into standard Document with advanced metadata
    doc = Document(
        text=text_content,
        metadata={
            "source_url": source_url,
            "published_year": "2026"
        }
    )
    
    # Configure Hierarchical Parser: Child nodes (128 tokens) linked to Parent nodes (512 tokens)
    node_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[512, 128],
        chunk_overlap=20
    )
    nodes = node_parser.get_nodes_from_documents([doc])
    
    # Build and persist the vector index natively
    VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model
    )
    print("Successfully populated Index_Unstructured.")

def ingest_structured_data(markdown_table_content: str, ticker_symbol: str = "GENERIC", topic_scope: str = "market") -> None:
    """
    Isolates tables using MarkdownElementNodeParser and saves to Index_Structured.
    """
    print(f"Processing structured data pipeline for scope: {topic_scope}...")
    
    chroma_collection = chroma_client.get_or_create_collection("index_structured")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    doc = Document(
        text=markdown_table_content,
        metadata={
            "ticker_symbol": ticker_symbol,
            "topic_scope": topic_scope
        }
    )
    
    # Parse markdown table elements strictly to maintain column/row associations
    node_parser = MarkdownElementNodeParser(llm=llm, num_workers=1)
    nodes = node_parser.get_nodes_from_documents([doc])
    
    VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model
    )
    print("Successfully populated Index_Structured.")

if __name__ == "__main__":
    # Test execution block for verifying pipeline ingestion locally
    sample_text = "Vietnam electric vehicle sales surged significantly in early 2026 driven by domestic battery infrastructure expansions."
    sample_url = "https://example.com/ev-trends-2026"
    ingest_unstructured_data(sample_text, sample_url)
    
    sample_table = "| Year | Market Share |\n|---|---|\n| 2025 | 15% |\n| 2026 | 32% |"
    ingest_structured_data(sample_table, ticker_symbol="VINFAST", topic_scope="supply_chain")
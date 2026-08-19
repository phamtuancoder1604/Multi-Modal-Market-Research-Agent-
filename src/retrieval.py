import os
import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.postprocessor import LLMRerank
from llama_index.llms.ollama import Ollama
# Import correct Metadata filter classes
from llama_index.core.vector_stores.types import MetadataFilters, MetadataFilter

embed_model = OllamaEmbedding(model_name="nomic-embed-text", base_url="http://localhost:11434")
llm = Ollama(model="qwen2.5:7b", base_url="http://localhost:11434", request_timeout=120.0)
chroma_client = chromadb.PersistentClient(path=os.path.join("data", "vector_db"))

def retrieve_and_rerank(query_str: str) -> list:
    """
    Executes parallel retrieval across index collections with strict metadata filtering,
    then applies local reranking to isolate the top 5 high-relevance contexts.
    """
    print(f"Executing multi-index parallel search for: '{query_str}'")
    combined_nodes = []
    
    # 1. Querying Index_Unstructured with fixed LlamaIndex Metadata Filters
    try:
        coll_unstructured = chroma_client.get_collection("index_unstructured")
        store_unstructured = ChromaVectorStore(chroma_collection=coll_unstructured)
        index_unstructured = VectorStoreIndex.from_vector_store(store_unstructured, embed_model=embed_model)
        
        # Configure correct filter objects
        unstructured_filters = MetadataFilters(
            filters=[MetadataFilter(key="published_year", value="2026")]
        )
        
        retriever_unstructured = index_unstructured.as_retriever(
            similarity_top_k=10,
            filters=unstructured_filters
        )
        combined_nodes.extend(retriever_unstructured.retrieve(query_str))
    except Exception as e:
        print(f"Unstructured search failure: {str(e)}")

    # 2. Querying Index_Structured
    try:
        coll_structured = chroma_client.get_collection("index_structured")
        store_structured = ChromaVectorStore(chroma_collection=coll_structured)
        index_structured = VectorStoreIndex.from_vector_store(store_structured, embed_model=embed_model)
        
        retriever_structured = index_structured.as_retriever(similarity_top_k=10)
        combined_nodes.extend(retriever_structured.retrieve(query_str))
    except Exception as e:
        print(f"Structured search failure: {str(e)}")

    if not combined_nodes:
        return []

    # 3. Applying local semantic cross-scoring
    print(f"Reranking {len(combined_nodes)} combined raw nodes via local LLM...")
    reranker = LLMRerank(choice_batch_size=5, top_n=5, llm=llm)
    reranked_nodes = reranker.postprocess_nodes(combined_nodes, query_str=query_str)
    
    return [node.node.get_content() for node in reranked_nodes]

if __name__ == "__main__":
    test_query = "What are the market share and sales trends of electric vehicles in Vietnam for 2026?"
    top_contexts = retrieve_and_rerank(test_query)
    
    print("\n--- TOP QUALITY CONTEXTS IDENTIFIED ---")
    for idx, context in enumerate(top_contexts, start=1):
        print(f"\n[Context {idx}]:\n{context}")

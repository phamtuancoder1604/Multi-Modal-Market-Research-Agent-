# import os
# import chromadb
# from llama_index.core import VectorStoreIndex
# from llama_index.vector_stores.chroma import ChromaVectorStore
# from llama_index.embeddings.ollama import OllamaEmbedding
# from llama_index.core.postprocessor import LLMRerank
# from llama_index.llms.ollama import Ollama
# # Import correct Metadata filter classes
# from llama_index.core.vector_stores.types import MetadataFilters, MetadataFilter

# embed_model = OllamaEmbedding(model_name="nomic-embed-text", base_url="http://localhost:11434")
# llm = Ollama(model="qwen2.5:7b", base_url="http://localhost:11434", request_timeout=120.0)
# chroma_client = chromadb.PersistentClient(path=os.path.join("data", "vector_db"))

# def retrieve_and_rerank(query_str: str) -> list:
#     """
#     Executes parallel retrieval across index collections with strict metadata filtering,
#     then applies local reranking to isolate the top 5 high-relevance contexts.
#     """
#     print(f"Executing multi-index parallel search for: '{query_str}'")
#     combined_nodes = []
    
#     # 1. Querying Index_Unstructured with fixed LlamaIndex Metadata Filters
#     try:
#         coll_unstructured = chroma_client.get_collection("index_unstructured")
#         store_unstructured = ChromaVectorStore(chroma_collection=coll_unstructured)
#         index_unstructured = VectorStoreIndex.from_vector_store(store_unstructured, embed_model=embed_model)
        
#         # Configure correct filter objects
#         unstructured_filters = MetadataFilters(
#             filters=[MetadataFilter(key="published_year", value="2026")]
#         )
        
#         retriever_unstructured = index_unstructured.as_retriever(
#             similarity_top_k=10,
#             filters=unstructured_filters
#         )
#         combined_nodes.extend(retriever_unstructured.retrieve(query_str))
#     except Exception as e:
#         print(f"Unstructured search failure: {str(e)}")

#     # 2. Querying Index_Structured
#     try:
#         coll_structured = chroma_client.get_collection("index_structured")
#         store_structured = ChromaVectorStore(chroma_collection=coll_structured)
#         index_structured = VectorStoreIndex.from_vector_store(store_structured, embed_model=embed_model)
        
#         retriever_structured = index_structured.as_retriever(similarity_top_k=10)
#         combined_nodes.extend(retriever_structured.retrieve(query_str))
#     except Exception as e:
#         print(f"Structured search failure: {str(e)}")

#     if not combined_nodes:
#         return []

#     # 3. Applying local semantic cross-scoring
#     print(f"Reranking {len(combined_nodes)} combined raw nodes via local LLM...")
#     reranker = LLMRerank(choice_batch_size=5, top_n=5, llm=llm)
#     reranked_nodes = reranker.postprocess_nodes(combined_nodes, query_str=query_str)
    
#     return [node.node.get_content() for node in reranked_nodes]

# if __name__ == "__main__":
#     test_query = "What are the market share and sales trends of electric vehicles in Vietnam for 2026?"
#     top_contexts = retrieve_and_rerank(test_query)
    
#     print("\n--- TOP QUALITY CONTEXTS IDENTIFIED ---")
#     for idx, context in enumerate(top_contexts, start=1):
#         print(f"\n[Context {idx}]:\n{context}")
import os
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.postprocessor import LLMRerank
from llama_index.llms.ollama import Ollama
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever

# Khởi tạo mô hình
embed_model = OllamaEmbedding(model_name="nomic-embed-text", base_url="http://localhost:11434")
llm = Ollama(model="qwen2.5:7b", base_url="http://localhost:11434", request_timeout=120.0)
chroma_client = chromadb.PersistentClient(path=os.path.join("data", "vector_db"))

# --- SEMANTIC CACHE TRONG BỘ NHỚ (IN-MEMORY CACHE) ---
SEMANTIC_CACHE = {}

def retrieve_and_rerank(query_str: str) -> list:
    """
    Thực thi Hybrid Search (Dense Vector + BM25 Sparse) kết hợp Semantic Cache.
    """
    # 1. Kiểm tra Cache trước khi làm tính toán RAG tốn kém
    cache_key = query_str.lower().strip()
    if cache_key in SEMANTIC_CACHE:
        print(f" [SEMANTIC CACHE HIT] Trả về kết quả tức thì từ bộ nhớ đệm cho: '{query_str}'")
        return SEMANTIC_CACHE[cache_key]

    print(f" Executing Hybrid Retrieval (Vector + BM25) for: '{query_str}'")
    combined_nodes = []
    
    # 2. Vector Search (Dense)
    try:
        coll_unstructured = chroma_client.get_collection("index_unstructured")
        store_unstructured = ChromaVectorStore(chroma_collection=coll_unstructured)
        index_unstructured = VectorStoreIndex.from_vector_store(store_unstructured, embed_model=embed_model)
        
        vector_retriever = index_unstructured.as_retriever(similarity_top_k=5)
        vector_nodes = vector_retriever.retrieve(query_str)
        combined_nodes.extend(vector_nodes)
    except Exception as e:
        print(f"Vector retrieval bypass: {str(e)}")

    # 3. BM25 Search (Sparse Keyword Search - Tìm chính xác từ khóa/số liệu)
    try:
        all_docs = index_unstructured.docstore.docs.values() if 'index_unstructured' in locals() else []
        if all_docs:
            nodes_list = list(all_docs)
            bm25_retriever = BM25Retriever.from_defaults(nodes=nodes_list, similarity_top_k=5)
            bm25_nodes = bm25_retriever.retrieve(query_str)
            combined_nodes.extend(bm25_nodes)
    except Exception as e:
        print(f"BM25 retrieval bypass: {str(e)}")

    if not combined_nodes:
        return []

    # 4. Local Reranking qua Qwen 2.5
    print(f"Reranking {len(combined_nodes)} hybrid nodes via local LLM...")
    reranker = LLMRerank(choice_batch_size=5, top_n=5, llm=llm)
    reranked_nodes = reranker.postprocess_nodes(combined_nodes, query_str=query_str)
    
    final_contexts = [node.node.get_content() for node in reranked_nodes]
    
    # 5. Lưu kết quả vào Semantic Cache cho các lần truy vấn sau
    SEMANTIC_CACHE[cache_key] = final_contexts
    
    return final_contexts
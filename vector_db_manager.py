import chromadb
from chromadb.utils import embedding_functions
import os
import time
from utils import log_info, log_warning, log_error # Import logger từ utils
from config import (EMBEDDING_MODEL, COLLECTION_NAME, TOP_K_RESULTS,
                    MAX_RETRIES, RETRY_DELAY, VECTOR_DB_PATH)

# Biến toàn cục để giữ client và collection, quản lý trong app.py
chroma_client = None
collection = None

def initialize_vector_db(api_key):
    """
    Khởi tạo ChromaDB client và collection.
    Sử dụng API key được truyền vào để cấu hình hàm embedding.
    Trả về collection nếu thành công, None nếu lỗi.
    """
    global chroma_client, collection
    if collection is not None:
        log_info("Collection ChromaDB đã được khởi tạo trước đó.")
        return collection

    if not api_key:
        log_error("Lỗi nghiêm trọng: Thiếu OpenAI API Key để khởi tạo ChromaDB Embedding Function.")
        return None

    try:
        # Khởi tạo client (in-memory hoặc persistent tùy config)
        if VECTOR_DB_PATH:
             log_info(f"Khởi tạo ChromaDB persistent tại: {VECTOR_DB_PATH}")
             chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        else:
            log_info("Khởi tạo ChromaDB in-memory.")
            chroma_client = chromadb.Client() # Client in-memory

        # Tạo hàm embedding của OpenAI với API key được cung cấp
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name=EMBEDDING_MODEL
        )

        # Lấy hoặc tạo collection
        log_info(f"Đang lấy hoặc tạo collection: '{COLLECTION_NAME}'...")
        collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=openai_ef,
            metadata={"hnsw:space": "cosine"} # Sử dụng cosine similarity
        )
        log_info(f"Collection '{COLLECTION_NAME}' đã sẵn sàng.")
        return collection

    except Exception as e:
        log_error("Lỗi nghiêm trọng khi khởi tạo ChromaDB hoặc collection.", e)
        chroma_client = None
        collection = None
        return None

def add_chunks_to_vector_db(chunks_to_add):
    """
    Thêm các chunks vào collection ChromaDB đã được khởi tạo.
    Hàm này giả định 'collection' đã được khởi tạo thành công.
    Trả về True nếu thành công (thêm ít nhất 1 chunk), False nếu lỗi.
    """
    global collection
    if collection is None:
        log_error("Lỗi: Collection ChromaDB chưa được khởi tạo. Không thể thêm chunks.")
        return False
    if not chunks_to_add:
        log_warning("Không có chunks nào được cung cấp để thêm vào Vector DB.")
        return False

    log_info(f"Chuẩn bị thêm {len(chunks_to_add)} chunks vào Vector DB...")
    ids = [f"chunk_{i}" for i, chunk in enumerate(chunks_to_add)] # Tạo ID duy nhất
    documents = chunks_to_add

    try:
        # Chia thành các batch nhỏ để thêm vào ChromaDB, tránh lỗi API/timeout
        batch_size = 100 # Kích thước batch hợp lý
        num_batches = (len(documents) + batch_size - 1) // batch_size
        log_info(f"Thêm dữ liệu theo {num_batches} batches (kích thước batch: {batch_size}).")

        added_count = 0
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(documents))
            batch_ids = ids[start_idx:end_idx]
            batch_documents = documents[start_idx:end_idx]

            if not batch_documents:
                continue

            log_info(f"Đang thêm batch {i+1}/{num_batches} ({len(batch_documents)} documents)...")
            # Sử dụng add thay vì upsert nếu muốn đảm bảo ID là duy nhất và báo lỗi nếu trùng
            collection.add(
                ids=batch_ids,
                documents=batch_documents
                # ChromaDB sẽ tự động gọi embedding function đã đăng ký
            )
            added_count += len(batch_documents)
            log_info(f"Đã thêm thành công batch {i+1}. Tổng cộng đã thêm: {added_count} documents.")
            # Delay nhỏ giữa các batch để tránh rate limit API embedding
            if num_batches > 1 and i < num_batches - 1:
                 time.sleep(1) # Chờ 1 giây

        log_info(f"Hoàn tất thêm {added_count} chunks vào Vector DB.")
        return added_count > 0

    except Exception as e:
        # Lỗi có thể xảy ra do API embedding (rate limit, key sai...) hoặc ChromaDB
        log_error(f"Lỗi nghiêm trọng khi thêm chunks vào ChromaDB.", e)
        return False

def query_vector_db(query_text, top_k=TOP_K_RESULTS):
    """
    Truy vấn Vector DB để tìm các chunks liên quan nhất.
    Hàm này giả định 'collection' đã được khởi tạo thành công.
    Trả về list các nội dung document liên quan, hoặc list rỗng nếu lỗi/không tìm thấy.
    """
    global collection
    if collection is None:
        log_error("Lỗi: Collection ChromaDB chưa được khởi tạo. Không thể truy vấn.")
        return []
    if not query_text or not query_text.strip():
        log_warning("Câu truy vấn rỗng, không thực hiện truy vấn.")
        return []

    try:
        log_info(f"Truy vấn Vector DB (top_k={top_k}) cho: '{query_text[:100]}...'")
        # ChromaDB sẽ tự tạo embedding cho query_text bằng embedding function đã đăng ký
        results = collection.query(
            query_texts=[query_text],
            n_results=top_k,
            include=['documents', 'distances'] # Lấy cả nội dung và khoảng cách
        )

        # Xử lý kết quả
        retrieved_docs = []
        if results and results.get('documents') and results['documents'][0]:
            retrieved_docs = results['documents'][0]
            distances = results.get('distances', [[]])[0] # Lấy distances nếu có
            log_info(f"Tìm thấy {len(retrieved_docs)} chunks liên quan:")
            for i, (doc, dist) in enumerate(zip(retrieved_docs, distances)):
                 log_info(f"  - Rank {i+1}, Distance: {dist:.4f}, Content: '{doc[:100]}...'")
        else:
            log_info("Không tìm thấy chunks nào liên quan trong Vector DB.")

        return retrieved_docs

    except Exception as e:
        log_error(f"Lỗi xảy ra trong quá trình truy vấn Vector DB cho câu hỏi '{query_text[:50]}...'.", e)
        return []

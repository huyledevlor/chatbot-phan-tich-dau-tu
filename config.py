# --- CONFIGURATIONS ---
import logging

# Cấu hình Logging
LOG_FILE = "chatbot_app.log" # Tên file log
LOG_LEVEL = logging.INFO # Mức độ log (INFO, DEBUG, WARNING, ERROR)

# Cấu hình OpenAI
# API Key sẽ được lấy từ Streamlit secrets, không đặt ở đây
GENERATION_MODEL = "gpt-3.5-turbo" # Model để sinh câu trả lời
GENERATION_TEMPERATURE = 0.3       # Nhiệt độ cho LLM (càng thấp càng bám sát context)
GENERATION_MAX_TOKENS = 700        # Giới hạn token cho câu trả lời
EMBEDDING_MODEL = "text-embedding-3-small" # Model để tạo vector nhúng

# Cấu hình RAG & Vector DB
KNOWLEDGE_BASE_FILE = "knowledge_base.txt" # Tên file kiến thức
CHUNK_SIZE = 800          # Kích thước mỗi chunk văn bản (ký tự)
CHUNK_OVERLAP = 100       # Độ chồng lấp giữa các chunk
# VECTOR_DB_PATH = "chroma_db_persistent" # Bỏ comment nếu muốn lưu DB vào thư mục (phức tạp hơn khi deploy)
VECTOR_DB_PATH = None     # Sử dụng None để chạy ChromaDB in-memory (đơn giản cho deploy)
COLLECTION_NAME = "investment_docs_v1" # Tên collection trong ChromaDB
TOP_K_RESULTS = 3         # Số lượng chunks liên quan nhất cần truy xuất

# Cấu hình xử lý lỗi và UI
DEFAULT_ERROR_MESSAGE = "Xin lỗi, tôi gặp chút sự cố khi xử lý yêu cầu của bạn. Vui lòng thử lại sau."
MAX_RETRIES = 2           # Số lần thử lại tối đa khi gọi API OpenAI
RETRY_DELAY = 3           # Thời gian chờ (giây) giữa các lần thử lại
ASSISTANT_AVATAR = "🤖"
ASSISTANT_DEFAULT_MESSAGE = "Xin chào! Tôi là trợ lý AI phân tích đầu tư. Bạn cần thông tin gì về [Tên công ty/dự án của bạn]?" # Sửa lại cho phù hợp

# --- Prompt Template cho Generation (Đã tinh chỉnh) ---
RAG_PROMPT_TEMPLATE = """
Bạn là một trợ lý AI phân tích đầu tư sơ bộ, am hiểu về thị trường nhưng cần dựa vào thông tin cụ thể được cung cấp.
Nhiệm vụ: Trả lời câu hỏi của người dùng một cách chính xác và hữu ích. Hãy dựa **chủ yếu** vào NGỮ CẢNH ĐƯỢC TRUY XUẤT dưới đây làm nguồn thông tin chính.
Bạn có thể sử dụng kiến thức nền tảng của mình để tổng hợp, giải thích và trình bày câu trả lời một cách mạch lạc và chuyên nghiệp.
**Cảnh báo:** TUYỆT ĐỐI KHÔNG đưa ra các thông tin, số liệu, sự kiện cụ thể không có trong ngữ cảnh được cung cấp. Không đưa ra lời khuyên mua/bán hay dự đoán giá.
Nếu ngữ cảnh không chứa đủ thông tin để trả lời câu hỏi một cách đầy đủ, hãy thành thật thừa nhận và chỉ ra thông tin bị thiếu nếu có thể.
Trả lời bằng tiếng Việt.

NGỮ CẢNH ĐƯỢC TRUY XUẤT:
---
{retrieved_context}
---

CÂU HỎI CỦA NGƯỜI DÙNG: {user_question}

Câu trả lời của bạn (dựa chủ yếu vào ngữ cảnh):
"""

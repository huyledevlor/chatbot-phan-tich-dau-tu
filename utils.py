import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import LOG_FILE, LOG_LEVEL, KNOWLEDGE_BASE_FILE, CHUNK_SIZE, CHUNK_OVERLAP

# --- Logging Setup ---
# Thiết lập cấu hình logging cơ bản
logging.basicConfig(
    filename=LOG_FILE,
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)
# Tạo logger cụ thể cho module này (thực hành tốt)
logger = logging.getLogger(__name__)

def log_info(message):
    """Ghi log thông tin"""
    logger.info(message)

def log_warning(message):
    """Ghi log cảnh báo"""
    logger.warning(message)

def log_error(message, exception=None):
    """Ghi log lỗi, kèm theo exception nếu có"""
    if exception:
        logger.error(f"{message}: {exception}", exc_info=True) # exc_info=True để ghi traceback
    else:
        logger.error(message)

# --- Data Loading ---
def load_knowledge_base(file_path=KNOWLEDGE_BASE_FILE):
    """Đọc nội dung từ file kiến thức nền tảng"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        log_info(f"Tải thành công knowledge base từ: {file_path}")
        return content
    except FileNotFoundError:
        log_error(f"Lỗi nghiêm trọng: Không tìm thấy file knowledge base tại '{file_path}'.")
        return None
    except Exception as e:
        log_error(f"Lỗi không xác định khi đọc file knowledge base '{file_path}'.", e)
        return None

# --- Text Splitting ---
def split_text_into_chunks(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Chia văn bản thành các chunks sử dụng RecursiveCharacterTextSplitter"""
    if not text:
        log_warning("Đầu vào văn bản rỗng, không thể chia chunks.")
        return []
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
            separators=["\n\n", "\n", ". ", ", ", " ", ""], # Ưu tiên tách theo dòng, câu, từ
            keep_separator=False # Không giữ lại ký tự phân tách trong chunk
        )
        chunks = text_splitter.split_text(text)
        # Loại bỏ các chunk chỉ chứa khoảng trắng hoặc quá ngắn (ví dụ < 10 ký tự)
        valid_chunks = [chunk for chunk in chunks if chunk.strip() and len(chunk.strip()) > 10]
        num_original = len(chunks)
        num_valid = len(valid_chunks)
        if num_original > num_valid:
             log_info(f"Đã chia văn bản thành {num_original} chunks, loại bỏ {num_original - num_valid} chunks rỗng/quá ngắn. Còn lại {num_valid} chunks hợp lệ.")
        else:
             log_info(f"Đã chia văn bản thành {num_valid} chunks hợp lệ.")
        return valid_chunks
    except Exception as e:
        log_error("Lỗi xảy ra trong quá trình chia văn bản thành chunks.", e)
        return []

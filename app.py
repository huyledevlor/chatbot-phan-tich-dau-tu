# --- HACK: Workaround for Streamlit Cloud SQLite version ----
# Must be first import
import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# --- End of HACK ---
import streamlit as st
import openai
import os
import time
from utils import log_info, log_error, load_knowledge_base, split_text_into_chunks
from vector_db_manager import initialize_vector_db, add_chunks_to_vector_db, collection as chroma_collection_global
from rag_core import generate_response_with_rag
from config import * # Import tất cả cấu hình

# --- Page Configuration (Nên đặt ở đầu) ---
st.set_page_config(
    page_title="Chatbot Phân tích Đầu tư",
    page_icon="🤖",
    layout="centered" # Có thể dùng "wide" nếu muốn rộng hơn
)

# --- Logging ---
log_info("--- Khởi động ứng dụng Chatbot Streamlit ---")

# --- Authentication & OpenAI Client Initialization ---
openai_client = None
openai_api_key = None

# Cố gắng lấy API key từ Streamlit secrets (ưu tiên khi deploy)
try:
    openai_api_key = st.secrets.get("OPENAI_API_KEY")
except Exception as e:
    log_warning(f"Không thể truy cập st.secrets (có thể đang chạy local không có secrets): {e}")

# Nếu không có trong secrets, thử lấy từ biến môi trường (cho local test)
if not openai_api_key:
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key:
        log_info("Đã tìm thấy OPENAI_API_KEY trong biến môi trường.")

if not openai_api_key:
    st.error("Lỗi cấu hình: Không tìm thấy OpenAI API Key. Vui lòng thiết lập trong Streamlit Secrets (khi deploy) hoặc biến môi trường (khi chạy local).")
    log_error("Thiếu OpenAI API Key để khởi chạy ứng dụng.")
    st.stop() # Dừng ứng dụng

# Khởi tạo OpenAI client
try:
    openai.api_key = openai_api_key # Gán key cho thư viện (một số hàm cũ có thể cần)
    openai_client = openai.OpenAI(api_key=openai_api_key)
    # Kiểm tra client bằng cách gọi một lệnh nhẹ
    openai_client.models.list()
    log_info("Khởi tạo và xác thực OpenAI client thành công.")
except openai.AuthenticationError:
    st.error("Lỗi xác thực OpenAI API Key. Key không hợp lệ hoặc đã hết hạn.")
    log_error("Lỗi xác thực OpenAI API Key.")
    st.stop()
except Exception as e:
    st.error(f"Lỗi không xác định khi khởi tạo OpenAI client: {e}")
    log_error(f"Lỗi khởi tạo OpenAI client: {e}", e)
    st.stop()


# --- Vector Database Setup and Indexing ---
# Sử dụng cache_resource để tránh khởi tạo và indexing lại mỗi lần script chạy lại (do tương tác UI)
@st.cache_resource(show_spinner="Đang thiết lập cơ sở dữ liệu vector...")
def setup_vector_database(api_key_for_embedding):
    """
    Hàm được cache để khởi tạo và nạp dữ liệu vào Vector DB.
    Chỉ chạy lại khi cache bị xóa hoặc lần đầu khởi động.
    """
    log_info("Bắt đầu chạy hàm setup_vector_database (có thể được cache)...")

    # 1. Khởi tạo Vector DB collection
    # Truyền API key vào đây để ChromaDB tạo OpenAIEmbeddingFunction
    initialized_collection = initialize_vector_db(api_key=api_key_for_embedding)

    if initialized_collection is None:
        log_error("Lỗi nghiêm trọng: Không thể khởi tạo Vector Database collection.")
        # Không thể hiển thị lỗi trực tiếp từ hàm cache, cần xử lý bên ngoài
        return None # Trả về None để báo lỗi

    # 2. Kiểm tra và Nạp dữ liệu (Indexing) nếu cần
    try:
        # Sử dụng collection trả về từ hàm khởi tạo
        if initialized_collection.count() == 0:
            log_info(f"Collection '{COLLECTION_NAME}' rỗng. Bắt đầu Indexing...")

            # Tải Knowledge Base
            knowledge_text = load_knowledge_base()
            if not knowledge_text:
                log_error("Lỗi nghiêm trọng: Không thể tải Knowledge Base để indexing.")
                return None

            # Chia chunks
            chunks = split_text_into_chunks(knowledge_text)
            if not chunks:
                log_error("Lỗi nghiêm trọng: Không thể chia Knowledge Base thành chunks.")
                return None

            # Thêm vào Vector DB
            log_info(f"Bắt đầu thêm {len(chunks)} chunks vào Vector DB (quá trình này cần gọi API embedding)...")
            # Hàm add_chunks_to_vector_db dùng biến collection toàn cục,
            # nên cần đảm bảo biến đó đã được cập nhật trong initialize_vector_db
            # Hoặc tốt hơn là sửa add_chunks_to_vector_db để nhận collection làm tham số
            success = add_chunks_to_vector_db(chunks) # Gọi hàm thêm chunks
            if not success:
                log_error("Lỗi nghiêm trọng: Xảy ra lỗi trong quá trình thêm chunks vào Vector DB.")
                return None
            log_info(">>> Indexing dữ liệu vào Vector Database thành công! <<<")
        else:
            count = initialized_collection.count()
            log_info(f"Collection '{COLLECTION_NAME}' đã có dữ liệu ({count} documents). Bỏ qua Indexing.")

        # Gán collection đã khởi tạo vào biến toàn cục để các hàm khác dùng (nếu cần)
        # Lưu ý: Cách này không lý tưởng lắm, nhưng đơn giản cho cấu trúc hiện tại
        import vector_db_manager
        vector_db_manager.collection = initialized_collection

        return initialized_collection # Trả về collection đã sẵn sàng

    except Exception as e:
         log_error(f"Lỗi xảy ra trong quá trình kiểm tra hoặc indexing dữ liệu: {e}", e)
         return None # Trả về None nếu có lỗi

# Gọi hàm setup (Streamlit sẽ quản lý cache)
chroma_collection_cached = setup_vector_database(openai_api_key)

# Kiểm tra kết quả từ hàm setup
if chroma_collection_cached is None:
    st.error("Lỗi nghiêm trọng khi thiết lập Vector Database. Không thể khởi chạy chatbot. Vui lòng kiểm tra log.")
    log_error("Dừng ứng dụng do lỗi setup Vector DB.")
    st.stop()
else:
    log_info("Vector Database đã được thiết lập và sẵn sàng.")


# --- Streamlit Chat Interface ---
st.title("🤖 Chatbot Phân tích Đầu tư")
st.caption(f"Trợ lý AI trả lời câu hỏi dựa trên tài liệu về Công ty FPT") # Nhớ sửa lại

# Khởi tạo lịch sử chat trong session state nếu chưa có
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": ASSISTANT_DEFAULT_MESSAGE, "avatar": ASSISTANT_AVATAR}]
    log_info("Khởi tạo lịch sử chat mới.")

# Hiển thị các tin nhắn đã có trong lịch sử
for message in st.session_state.messages:
    avatar = message.get("avatar")
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Nhận input mới từ người dùng
if prompt := st.chat_input("Nhập câu hỏi của bạn ở đây..."):
    log_info(f"Nhận câu hỏi từ người dùng: {prompt}")
    # Thêm tin nhắn của user vào lịch sử và hiển thị ngay
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Tạo và hiển thị phản hồi từ assistant
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        message_placeholder = st.empty() # Tạo placeholder để cập nhật từ từ
        full_response = ""
        try:
            # Hiển thị spinner trong khi chờ phản hồi
            with st.spinner("Chatbot đang suy nghĩ..."):
                 # Gọi hàm RAG để lấy câu trả lời
                 # Đảm bảo openai_client đã được khởi tạo thành công ở trên
                ai_response = generate_response_with_rag(openai_client, prompt)

            # Hiệu ứng gõ chữ (streaming simulation)
            response_words = ai_response.split()
            for i, word in enumerate(response_words):
                full_response += word + " "
                time.sleep(0.05) # Delay nhỏ tạo hiệu ứng
                message_placeholder.markdown(full_response + ("▌" if i < len(response_words) - 1 else "")) # Thêm con trỏ nhấp nháy
            message_placeholder.markdown(full_response) # Hiển thị đầy đủ khi xong
            log_info(f"Phản hồi của Assistant (tóm tắt): {full_response[:100]}...")

        except Exception as e:
            full_response = DEFAULT_ERROR_MESSAGE # Lấy thông báo lỗi mặc định
            st.error(f"Đã xảy ra lỗi khi tạo phản hồi: {e}") # Hiển thị lỗi trên UI
            log_error(f"Lỗi nghiêm trọng khi tạo phản hồi cho câu hỏi '{prompt}': {e}", e)
            message_placeholder.markdown(full_response) # Hiển thị thông báo lỗi

    # Thêm phản hồi (hoặc lỗi) của assistant vào lịch sử
    st.session_state.messages.append({"role": "assistant", "content": full_response, "avatar": ASSISTANT_AVATAR})

log_info("--- Kết thúc một lượt xử lý request Streamlit ---")

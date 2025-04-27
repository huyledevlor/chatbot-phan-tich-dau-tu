import openai
import time
from utils import log_info, log_warning, log_error
from vector_db_manager import query_vector_db # Import hàm truy vấn
from config import (RAG_PROMPT_TEMPLATE, GENERATION_MODEL, GENERATION_TEMPERATURE,
                    GENERATION_MAX_TOKENS, DEFAULT_ERROR_MESSAGE, MAX_RETRIES, RETRY_DELAY)

def generate_response_with_rag(openai_client, user_question):
    """
    Thực hiện quy trình RAG hoàn chỉnh: Truy xuất -> Tạo Prompt -> Gọi LLM.
    Cần truyền vào OpenAI client đã được khởi tạo.
    Trả về câu trả lời của AI hoặc thông báo lỗi.
    """
    if not openai_client:
        log_error("Lỗi nghiêm trọng: OpenAI client không hợp lệ trong generate_response_with_rag.")
        return DEFAULT_ERROR_MESSAGE
    if not user_question or not user_question.strip():
        log_warning("Câu hỏi người dùng rỗng.")
        return "Vui lòng cung cấp câu hỏi cụ thể."

    log_info(f"Bắt đầu quy trình RAG cho câu hỏi: {user_question}")

    # 1. Truy xuất ngữ cảnh liên quan từ Vector DB
    retrieved_context_list = query_vector_db(user_question) # Hàm từ vector_db_manager

    if not retrieved_context_list:
        log_warning(f"Không truy xuất được context nào cho câu hỏi: '{user_question[:100]}...'")
        retrieved_context_str = "Không có thông tin liên quan trực tiếp được tìm thấy trong tài liệu."
        # Có thể bổ sung: thử gọi LLM không cần context nếu muốn, nhưng không khuyến khích
        # với prompt hiện tại yêu cầu dựa vào context.
    else:
        # Kết hợp các chunks thành một chuỗi context duy nhất, ngăn cách rõ ràng
        retrieved_context_str = "\n\n---\n\n".join(retrieved_context_list)
        log_info(f"Context truy xuất được để đưa vào prompt (Số chunks: {len(retrieved_context_list)}).")


    # 2. Tạo Prompt hoàn chỉnh cho LLM
    try:
        final_prompt = RAG_PROMPT_TEMPLATE.format(
            retrieved_context=retrieved_context_str,
            user_question=user_question
        )
        # log_info(f"Prompt cuối cùng gửi đến LLM:\n{final_prompt}") # Bỏ comment nếu cần debug prompt
    except KeyError as e:
        log_error(f"Lỗi định dạng prompt template. Thiếu key: {e}")
        return DEFAULT_ERROR_MESSAGE
    except Exception as e:
         log_error(f"Lỗi không xác định khi tạo prompt: {e}", e)
         return DEFAULT_ERROR_MESSAGE

    # 3. Gọi LLM để tạo câu trả lời, có xử lý retry
    for attempt in range(MAX_RETRIES):
        try:
            log_info(f"Gửi yêu cầu đến model LLM: {GENERATION_MODEL} (Attempt {attempt + 1}/{MAX_RETRIES})")
            completion = openai_client.chat.completions.create(
                model=GENERATION_MODEL,
                messages=[
                    # System prompt đã được tích hợp vào user prompt (RAG_PROMPT_TEMPLATE)
                    {"role": "user", "content": final_prompt}
                ],
                temperature=GENERATION_TEMPERATURE,
                max_tokens=GENERATION_MAX_TOKENS
            )
            response_content = completion.choices[0].message.content.strip()
            log_info("Nhận phản hồi thành công từ LLM.")
            return response_content # Trả về kết quả thành công

        except openai.RateLimitError as e:
            log_warning(f"Lỗi Rate Limit từ LLM (Attempt {attempt + 1}/{MAX_RETRIES}). Đang chờ {RETRY_DELAY} giây...", e)
        except openai.APIConnectionError as e:
            log_warning(f"Lỗi kết nối API LLM (Attempt {attempt + 1}/{MAX_RETRIES}). Đang chờ {RETRY_DELAY} giây...", e)
        except openai.APIStatusError as e:
             log_error(f"Lỗi API Status từ OpenAI LLM (Status: {e.status_code}, Attempt {attempt + 1}/{MAX_RETRIES}): {e.response}", e)
             # Không nên retry với lỗi status cụ thể trừ khi biết rõ
             return DEFAULT_ERROR_MESSAGE # Trả về lỗi ngay
        except Exception as e:
            log_error(f"Lỗi không xác định khi gọi LLM (Attempt {attempt + 1}/{MAX_RETRIES}).", e)
            # Có thể break hoặc tiếp tục retry tùy chiến lược

        # Chờ trước khi thử lại (nếu chưa phải lần cuối)
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAY)
        else:
            log_error(f"Không thể nhận phản hồi từ LLM sau {MAX_RETRIES} lần thử.")
            return DEFAULT_ERROR_MESSAGE # Trả về lỗi sau khi hết số lần thử

    # Trường hợp vòng lặp kết thúc mà không return (dù ít khi xảy ra)
    return DEFAULT_ERROR_MESSAGE

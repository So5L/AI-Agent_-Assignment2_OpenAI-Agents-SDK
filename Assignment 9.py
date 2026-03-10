import dotenv
dotenv.load_dotenv()

from openai import OpenAI
import base64
import asyncio
import streamlit as st
from agents import Agent, Runner, SQLiteSession, WebSearchTool, FileSearchTool, ImageGenerationTool

client = OpenAI()

VECTOR_STORE_ID = "vs_69a788aa22e081919ba7ff188f889d04"

# ✅ 1) Session에 저장되기 전에 "금지 필드" 제거
BAD_KEYS = {"action", "background", "parsed_arguments"}

def _sanitize_obj(obj):
    """Recursively remove keys that can break Responses API input schema."""
    if isinstance(obj, dict):
        return {k: _sanitize_obj(v) for k, v in obj.items() if k not in BAD_KEYS}
    if isinstance(obj, list):
        return [_sanitize_obj(x) for x in obj]
    return obj

class SanitizingSQLiteSession(SQLiteSession):
    async def add_items(self, items):
        if items:
            items = [_sanitize_obj(it) for it in items]
        return await super().add_items(items)

def run_async(coro):
    """Streamlit 환경에서 안전하게 async 실행."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Streamlit이 이미 loop를 돌리는 환경이면 task로 실행
        return asyncio.create_task(coro)
    return asyncio.run(coro)


# ---- Agent ----
if "agent" not in st.session_state:
    st.session_state["agent"] = Agent(
        name="Life Coach Agent",
        instructions="""
You are a supportive Life Coach. Your job is to encourage the user and provide actionable, kind, motivating guidance.

You have access to the following tools:
- Web Search Tool: Use when you need up-to-date info or when you are unsure.
- File Search Tool: Use when the user asks about facts related to themselves or specific uploaded files.
- Image Generation Tool: Use when the user asks to create or modify images/diagrams/visual content.
""".strip(),
        tools=[
            WebSearchTool(),
            FileSearchTool(
                vector_store_ids=[VECTOR_STORE_ID],
                max_num_results=3,
            ),
            ImageGenerationTool(
                tool_config={
                    "type": "image_generation",
                    "quality": "high",
                    "output_format": "jpeg",
                    "partial_images": 1,
                }
            ),
        ],
    )

agent = st.session_state["agent"]

# ---- Session ----
if "session" not in st.session_state:
    st.session_state["session"] = SanitizingSQLiteSession(
        "chat-history",
        "chat-gpt-clone-memory.db",
    )
session = st.session_state["session"]


async def paint_history():
    messages = await session.get_items()
    for item in messages:
        item_type = item.get("type")

        # 1) 일반 메시지
        if item_type == "message":
            role = item.get("role", "assistant")
            with st.chat_message("human" if role == "user" else "ai"):
                # content 포맷이 SDK 버전에 따라 다를 수 있어서 방어적으로 처리
                content = item.get("content")
                if isinstance(content, str):
                    st.write(content)
                elif isinstance(content, list) and content and isinstance(content[0], dict):
                    # typical: [{"type":"output_text","text":"..."}] or [{"text":"..."}]
                    text = content[0].get("text") or content[0].get("content") or ""
                    st.write(str(text).replace("$", "\\$"))
                else:
                    st.write(str(content).replace("$", "\\$"))

        # 2) 툴콜 상태 표시(과거 기록)
        elif item_type == "web_search_call":
            with st.chat_message("ai"):
                st.write("🔍 (history) Searched the web...")
        elif item_type == "file_search_call":
            with st.chat_message("ai"):
                st.write("🗂️ (history) Searched your files...")
        elif item_type == "image_generation_call":
            # SDK에 따라 result/partial_image_b64 등이 다를 수 있어 방어적으로 처리
            b64 = item.get("result") or item.get("partial_image_b64")
            if b64:
                try:
                    image = base64.b64decode(b64)
                    with st.chat_message("ai"):
                        st.image(image)
                except Exception:
                    pass

run_async(paint_history())


def update_status(status_container, event_type: str):
    status_messages = {
        "response.web_search_call.in_progress": ("🔍 Starting web search...", "running"),
        "response.web_search_call.searching": ("🔍 Web search in progress...", "running"),
        "response.web_search_call.completed": ("✅ Web search completed.", "complete"),

        "response.file_search_call.in_progress": ("🗂️ Starting file search...", "running"),
        "response.file_search_call.searching": ("🗂️ File search in progress...", "running"),
        "response.file_search_call.completed": ("✅ File search completed.", "complete"),

        "response.completed": ("✅ Response completed.", "complete"),
    }
    if event_type in status_messages:
        label, state = status_messages[event_type]
        status_container.update(label=label, state=state)


async def run_agent(user_text: str):
    # ✅ 2) 최종 텍스트를 지우지 말고 "assistant 메시지"로 남기기
    with st.chat_message("ai"):
        status_container = st.status("⏳", expanded=False)
        text_placeholder = st.empty()
        image_placeholder = st.empty()
        response_text = ""

        stream = Runner.run_streamed(
            agent,
            user_text,
            session=session,
        )

        async for event in stream.stream_events():
            if event.type != "raw_response_event":
                continue

            event_type = event.data.type
            update_status(status_container, event_type)

            if event_type == "response.output_text.delta":
                response_text += event.data.delta
                text_placeholder.write(response_text.replace("$", "\\$"))

            elif event_type == "response.image_generation_call.partial_image":
                try:
                    image = base64.b64decode(event.data.partial_image_b64)
                    image_placeholder.image(image)
                except Exception:
                    pass

            elif event_type == "response.completed":
                # 완료 시: placeholders 정리 (텍스트는 이미 위에서 보여줌)
                image_placeholder.empty()
                # text_placeholder는 유지해도 되지만, "최종 출력"을 한번 더 확정적으로 표시
                if response_text.strip():
                    text_placeholder.write(response_text.replace("$", "\\$"))
                status_container.update(label="✅ Response completed.", state="complete")


prompt = st.chat_input(
    "Write a message for your assistant",
    accept_file=True,
    file_type=["txt"],
)

if prompt:
    # ✅ 3) 파일 업로드가 있는 경우 먼저 처리
    if getattr(prompt, "files", None):
        for file in prompt.files:
            if file.type.startswith("text/"):
                with st.chat_message("ai"):
                    with st.status("⏳ Uploading file...") as status:
                        uploaded_file = client.files.create(
                            file=(file.name, file.getvalue()),
                            purpose="user_data",
                        )
                        status.update(label="⏳ Attaching file...")
                        client.vector_stores.files.create(
                            vector_store_id=VECTOR_STORE_ID,
                            file_id=uploaded_file.id,
                        )
                        status.update(label="✅ File uploaded", state="complete")

    # 텍스트가 비어있고 파일만 올린 경우, 불필요하게 에이전트 호출하지 않기
    user_text = (prompt.text or "").strip()
    if user_text:
        with st.chat_message("human"):
            st.write(user_text)
        run_async(run_agent(user_text))


with st.sidebar:
    reset = st.button("Reset memory")
    if reset:
        run_async(session.clear_session())
        st.success("Memory cleared ✅")

    # 디버그용 (원하면 주석 처리)
    st.write(run_async(session.get_items()))
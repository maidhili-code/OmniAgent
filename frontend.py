import asyncio
import queue
import uuid

import streamlit as st
from langgraph_backend import chatbot, retrieve_all_threads, submit_async_task, delete_thread
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# =========================== Utilities ===========================

def generate_thread_id() -> str:
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []

def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def load_conversation(thread_id):
    try:
        state = chatbot.get_state(config={"configurable": {"thread_id": str(thread_id)}})
        return state.values.get("messages", [])
    except Exception:
        return []

def _content_to_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b if isinstance(b, str) else b.get("text", "")
            for b in content if isinstance(b, (str, dict))
        )
    return str(content)

def get_thread_title(thread_id, max_len=40) -> str:
    for msg in load_conversation(thread_id):
        if isinstance(msg, HumanMessage) and msg.content:
            text = _content_to_text(msg.content).strip()
            if text:
                return text if len(text) <= max_len else text[:max_len].rstrip() + "…"
    return "New conversation"

def _tool_call_names(chunk) -> list:
    """Best-effort extraction of tool call names from a streaming AIMessage chunk."""
    names = []
    for tc in (getattr(chunk, "tool_calls", None) or []):
        name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
        if name:
            names.append(name)
    return names

# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()
if "confirm_delete_id" not in st.session_state:
    st.session_state["confirm_delete_id"] = None

add_thread(st.session_state["thread_id"])

# ============================ Sidebar ============================
st.sidebar.title("OmniAgent")

if st.sidebar.button(" New Chat", use_container_width=True):
    reset_chat()

st.sidebar.markdown("---")
st.sidebar.header("Conversations")

for thread_id in st.session_state["chat_threads"][::-1]:
    title = get_thread_title(thread_id)
    col_select, col_delete = st.sidebar.columns([5, 1])

    with col_select:
        is_active = str(st.session_state["thread_id"]) == str(thread_id)
        label = f"**{title}**" if is_active else title
        if st.button(label, key=f"select_{thread_id}", use_container_width=True):
            st.session_state["thread_id"] = thread_id
            temp = []
            for msg in load_conversation(thread_id):
                if isinstance(msg, HumanMessage):
                    c = _content_to_text(msg.content)
                    if c:
                        temp.append({"role": "user", "content": c})
                elif isinstance(msg, AIMessage):
                    c = _content_to_text(msg.content).strip()
                    if c:
                        temp.append({"role": "assistant", "content": c})
            st.session_state["message_history"] = temp
            st.rerun()

    with col_delete:
        if st.button("🗑", key=f"delete_{thread_id}"):
            if st.session_state["confirm_delete_id"] == str(thread_id):
                delete_thread(thread_id)
                st.session_state["chat_threads"].remove(thread_id)
                st.session_state["confirm_delete_id"] = None
                if str(st.session_state["thread_id"]) == str(thread_id):
                    reset_chat()
                st.rerun()
            else:
                st.session_state["confirm_delete_id"] = str(thread_id)
                st.rerun()

    if st.session_state["confirm_delete_id"] == str(thread_id):
        st.sidebar.caption("Click 🗑 again to confirm delete.")

# ============================ Main Chat UI ============================

# Display chat history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask me anything…")

if user_input:
    # Add user message
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    CONFIG = {
        "configurable": {"thread_id": str(st.session_state["thread_id"])},
    }

    with st.chat_message("assistant"):
        tool_statuses = {}

        def stream_response():
            event_queue: queue.Queue = queue.Queue()

            async def run_stream():
                try:
                    async with asyncio.timeout(90):
                        async for chunk, metadata in chatbot.astream(
                            {"messages": [HumanMessage(content=user_input)]},
                            config=CONFIG,
                            stream_mode="messages",
                        ):
                            event_queue.put(("chunk", chunk, metadata))
                except Exception as exc:
                    event_queue.put(("error", str(exc)))
                finally:
                    event_queue.put(("done", None))

            submit_async_task(run_stream())

            while True:
                item = event_queue.get()
                kind = item[0]

                if kind == "done":
                    break
                if kind == "error":
                    yield f"❌ Error: {item[1]}"
                    break

                _, chunk, metadata = item
                node = metadata.get("langgraph_node", "") if metadata else ""

                # Tool finished -> mark its status complete
                if isinstance(chunk, ToolMessage):
                    tool_name = getattr(chunk, "name", "tool")
                    if tool_name in tool_statuses:
                        tool_statuses[tool_name].update(
                            label=f"✅ {tool_name} done", state="complete", expanded=False
                        )
                    else:
                        tool_statuses[tool_name] = st.status(
                            f"✅ {tool_name} done", state="complete", expanded=False
                        )
                    continue

                if isinstance(chunk, AIMessage) and node == "chat_node":
                    # Tool about to be called -> show status
                    for name in _tool_call_names(chunk):
                        if name not in tool_statuses:
                            tool_statuses[name] = st.status(f"🔧 Using {name}...", expanded=False)

                    # Actual answer tokens -> yield so st.write_stream renders them
                    token = _content_to_text(chunk.content)
                    if token:
                        yield token

        # Run streaming — this is what actually renders live text now
        streamed_text = st.write_stream(stream_response())

        # Fallback only if streaming genuinely produced nothing (e.g. a very
        # fast/non-streaming backend response)
        final_text = str(streamed_text).strip() if streamed_text else ""
        if not final_text:
            try:
                state = chatbot.get_state(config=CONFIG)
                for m in reversed(state.values.get("messages", [])):
                    if isinstance(m, AIMessage):
                        candidate = _content_to_text(m.content).strip()
                        if candidate:
                            final_text = candidate
                            st.markdown(final_text)
                            break
            except Exception:
                pass

        if not final_text:
            final_text = "Sorry, I couldn't generate a response."
            st.markdown(final_text)

    # Save to history
    st.session_state["message_history"].append(
        {"role": "assistant", "content": final_text}
    )
import os
import asyncio
import threading

import aiosqlite
import requests
from dotenv import load_dotenv

from typing import TypedDict, Annotated

from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

# ------------------- Config -------------------
ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
GITHUB_PERSONAL_ACCESS_TOKEN = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN", "")
FILESYSTEM_ROOT = os.environ.get("FILESYSTEM_ROOT", r"C:\Users\ponna\Desktop")

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

MAX_TOOL_RESULT_CHARS = 1500
MAX_HISTORY = 20

# ------------------- Startup check -------------------
if not GOOGLE_API_KEY:
    print(
        "[Gemini] ❌ GOOGLE_API_KEY is not set. Create a .env file with:\n"
        "    GOOGLE_API_KEY=your-key-here\n"
        "Get a key at https://aistudio.google.com/apikey"
    )
else:
    print(f"[Gemini] ✅ API key found, using model '{GEMINI_MODEL}'")

# ------------------- Async Setup -------------------
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()

def run_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP).result()

def submit_async_task(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)

# ------------------- LLM -------------------
def _thinking_kwargs(model_name: str) -> dict:
    """Gemini 2.5 uses thinking_budget (0 = fully off).
    Gemini 3.x uses thinking_level instead ('low' = fastest, can't fully disable).
    Auto-detecting avoids breaking every time Google ships a new model line."""
    if model_name.startswith("gemini-3"):
        return {"thinking_level": "low"}
    return {"thinking_budget": 0}

llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    api_key=GOOGLE_API_KEY,
    temperature=0.1,
    **_thinking_kwargs(GEMINI_MODEL),
)

import ast
import operator as _op

# ------------------- Basic Tools -------------------
search_tool = DuckDuckGoSearchRun(region="us-en")

_ARITH_OPS = {
    ast.Add: _op.add, ast.Sub: _op.sub, ast.Mult: _op.mul, ast.Div: _op.truediv,
    ast.FloorDiv: _op.floordiv, ast.Mod: _op.mod, ast.Pow: _op.pow,
    ast.USub: _op.neg, ast.UAdd: _op.pos,
}

def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ARITH_OPS:
        return _ARITH_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ARITH_OPS:
        return _ARITH_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression element: {ast.dump(node)}")

@tool
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression exactly, e.g. '84 * 12', '(15 + 7) / 2', '2 ** 10'.
    Supports + - * / // % ** and parentheses. Use this for any precise calculation
    instead of computing it mentally."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return str(result)
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"

@tool
def get_stock_price(symbol: str) -> str:
    """Get current stock price for a ticker (e.g. AAPL, TSLA)."""
    if not ALPHA_VANTAGE_API_KEY:
        return "Error: Alpha Vantage API key not configured."
    try:
        url = "https://www.alphavantage.co/query"
        params = {"function": "GLOBAL_QUOTE", "symbol": symbol.upper(), "apikey": ALPHA_VANTAGE_API_KEY}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json().get("Global Quote", {})
        if not data:
            return f"No data found for {symbol}."
        return (
            f"**{data.get('01. symbol', symbol)}**\n"
            f"Price: **${data.get('05. price', 'N/A')}**\n"
            f"Change: {data.get('09. change', 'N/A')} ({data.get('10. change percent', 'N/A')})\n"
            f"Volume: {data.get('06. volume', 'N/A')}\n"
            f"Date: {data.get('07. latest trading day', 'N/A')}"
        )
    except Exception as e:
        return f"Error fetching price: {e}"

basic_tools = [search_tool, get_stock_price, calculate]

# ------------------- MCP Tools (loaded EAGERLY, once, at import time) -------------------
def _build_mcp_config():
    cfg = {
        "filesystem": {"transport": "stdio", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", FILESYSTEM_ROOT]},
    }
    if GITHUB_PERSONAL_ACCESS_TOKEN:
        cfg["github"] = {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "env": {**os.environ, "GITHUB_PERSONAL_ACCESS_TOKEN": GITHUB_PERSONAL_ACCESS_TOKEN},
        }
    else:
        print("[MCP] ⏭ github skipped — GITHUB_PERSONAL_ACCESS_TOKEN not set in .env")
    return cfg

async def _load_single_server(name, cfg):
    try:
        client = MultiServerMCPClient({name: cfg})
        tools = await asyncio.wait_for(client.get_tools(), 15)
        print(f"[MCP] ✅ {name} loaded ({len(tools)} tools)")
        return tools
    except Exception as e:
        print(f"[MCP] ❌ {name}: {e}")
        return []

async def _init_mcp_tools():
    config = _build_mcp_config()
    results = await asyncio.gather(*[_load_single_server(n, c) for n, c in config.items()])
    return [t for group in results for t in group]

# Load once, synchronously, before the graph is built. This is what makes
# ToolNode and the bound LLM agree on the exact same set of tools — no more
# "MCP tool got called but ToolNode doesn't know it" crashes.
MCP_TOOLS = run_async(_init_mcp_tools())
ALL_TOOLS = basic_tools + MCP_TOOLS
print(f"[Tools] {len(ALL_TOOLS)} total tools available: {[t.name for t in ALL_TOOLS]}")

llm_with_tools = llm.bind_tools(ALL_TOOLS)
tool_node = ToolNode(ALL_TOOLS)

# ------------------- Graph -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

_tool_name_list = ", ".join(t.name for t in ALL_TOOLS) if ALL_TOOLS else "none currently available"

SYSTEM_PROMPT = SystemMessage(content=(
    "You are a helpful assistant. The EXACT and ONLY tools you have access to are: "
    f"{_tool_name_list}. "
    "Any question requiring real, current, or personal data — prices, file listings, "
    "repository names, search results, database contents, and similar — MUST be "
    "answered by calling the matching tool. NEVER invent, guess, or recall such data "
    "from memory, even if it sounds plausible. Always use the 'calculate' tool for "
    "arithmetic instead of computing it yourself, to guarantee exact results. "
    "If no tool in your list matches what's being asked (for example, the user asks "
    "about a service you have no tool for), say so plainly and explain that this "
    "capability isn't currently configured — do not fabricate an answer. "
    "\n\nGitHub search rules — follow these exactly:\n"
    "1. If the user asks about 'my repos' / 'my repositories' and you do not already "
    "know their GitHub username from this conversation, ASK for the username BEFORE "
    "calling any tool. Do not call search_repositories with a vague or missing owner "
    "and present the results as if they belong to the user.\n"
    "2. Once you have the username, call search_repositories with query set to EXACTLY "
    "'user:<username>' (the qualifier syntax, not the bare username). "
    "Example — user says 'octocat', you call: search_repositories(query='user:octocat'). "
    "Calling it with query='octocat' alone is WRONG and will return unrelated results.\n"
    "3. Other useful qualifiers: 'org:<name>', 'language:<lang>', 'stars:>10'.\n"
    "4. Before presenting search results as 'your repositories', verify they are "
    "plausibly owned by that username (check the repo owner field in the tool result) "
    "— never present generic/unrelated search hits as belonging to the user.\n"
    "\n\nAfter a tool returns a result, respond with a clean, well-formatted, natural-"
    "language final answer based ONLY on that tool result — never mention the tool "
    "by name in the final answer."
))

def _smart_truncate_history(messages, max_len):
    """Slice to at most max_len messages WITHOUT cutting through the middle of
    a tool-call/tool-result pair. Gemini requires a function-call turn to be
    immediately followed by its function-response turn; a naive messages[-N:]
    slice can orphan one side of that pair once history grows past N, which
    causes a 400 INVALID_ARGUMENT error. Instead we walk forward from the
    naive cut point to the next HumanMessage — a guaranteed safe turn boundary."""
    if len(messages) <= max_len:
        return messages
    cut = len(messages) - max_len
    while cut < len(messages) - 1 and not isinstance(messages[cut], HumanMessage):
        cut += 1
    return messages[cut:]

def _truncate(messages):
    return [msg if not (isinstance(msg, ToolMessage) and len(str(msg.content)) > MAX_TOOL_RESULT_CHARS)
            else ToolMessage(
                content=str(msg.content)[:MAX_TOOL_RESULT_CHARS] + "\n[...truncated...]",
                tool_call_id=msg.tool_call_id,
                name=getattr(msg, "name", None)
            ) for msg in messages]

import re
import time

def _extract_retry_delay(err_str: str, default: float = 20.0) -> float:
    m = re.search(r'"retryDelay":\s*"(\d+(?:\.\d+)?)s"', err_str)
    if m:
        return float(m.group(1)) + 1.0
    return default

async def _ainvoke_with_retry(model, messages, max_retries: int = 3):
    """Wraps llm.ainvoke with backoff on 429 RESOURCE_EXHAUSTED errors."""
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            return await model.ainvoke(messages)
        except Exception as e:
            err_str = str(e)
            last_err = e
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                if attempt < max_retries:
                    delay = _extract_retry_delay(err_str)
                    print(f"[Gemini] ⏳ Rate limited, retrying in {delay:.0f}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                    continue
            raise
    raise last_err

async def chat_node(state: ChatState):
    messages = list(state["messages"])

    if len(messages) > MAX_HISTORY:
        messages = _smart_truncate_history(messages, MAX_HISTORY)

    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SYSTEM_PROMPT, *messages]

    messages = _truncate(messages)
    last_msg = messages[-1]

    if isinstance(last_msg, ToolMessage):
        # Final natural-language answer, grounded in the FULL real conversation
        # (not a lossy reconstructed prompt like the old code did).
        response = await _ainvoke_with_retry(llm, messages)
    else:
        response = await _ainvoke_with_retry(llm_with_tools, messages)
        # Objective proof of whether a tool was actually called, printed
        # server-side — use this to confirm/deny hallucination, since the UI
        # status badge alone isn't enough evidence.
        calls = getattr(response, "tool_calls", None) or []
        if calls:
            for c in calls:
                name = c.get("name") if isinstance(c, dict) else getattr(c, "name", "?")
                args = c.get("args") if isinstance(c, dict) else getattr(c, "args", {})
                print(f"[ToolCall] model requested: {name}({args})")
        else:
            print("[ToolCall] none — model answered directly from its own knowledge")

    return {"messages": [response]}


# Checkpointer + Graph
async def init_checkpointer():
    conn = await aiosqlite.connect("chatbot.db")
    return AsyncSqliteSaver(conn)

checkpointer = run_async(init_checkpointer())

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)
print("✅ Backend ready — running on Gemini, all tools (including MCP) wired into ToolNode.")

# ------------------- Helpers for frontend -------------------
async def _alist_threads():
    threads = set()
    async for cp in checkpointer.alist(None):
        threads.add(cp.config["configurable"]["thread_id"])
    return list(threads)

async def _adelete_thread(tid):
    await checkpointer.adelete_thread(tid)

def retrieve_all_threads():
    return run_async(_alist_threads())

def delete_thread(thread_id):
    return run_async(_adelete_thread(thread_id))
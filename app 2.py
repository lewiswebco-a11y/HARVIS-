"""
Hierarchical Multi-Agent Startup Incubator
============================================
A single-file Streamlit application that runs a stateful, hierarchical
multi-agent system (Global COO -> Local PM -> Researcher -> Engineer -> PM Review)
using LangGraph, powered by Google Gemini 2.5 Flash (free tier).

Run locally:
    streamlit run app.py

Deploy on Hugging Face Spaces:
    - Add this file + requirements.txt to a "Streamlit" Space.
    - Set a Space secret named GOOGLE_API_KEY (optional — app runs in a
      mock/demo mode without it), or paste a key into the sidebar at runtime.
"""

import streamlit as st
import json
import os
import re
import uuid
import datetime
from typing import TypedDict, Dict, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage, SystemMessage
    GENAI_AVAILABLE = True
except ImportError:  # library not installed yet / still building on host
    GENAI_AVAILABLE = False


# ======================================================================
# PERSISTENCE LAYER (survives Streamlit reruns; best-effort across
# container sleeps on free hosting tiers, which may wipe local disk on
# a cold restart — the JSON file is the source of truth while the
# container is warm, and is reloaded into session_state on every boot).
# ======================================================================

STATE_DIR = "data"
STATE_FILE = os.path.join(STATE_DIR, "incubator_state.json")


def ensure_state_dir() -> None:
    os.makedirs(STATE_DIR, exist_ok=True)


def load_persisted_state() -> Dict[str, dict]:
    ensure_state_dir()
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def persist_state(startups: Dict[str, dict]) -> None:
    ensure_state_dir()
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(startups, f, indent=2, default=str)
    except Exception:
        pass  # ephemeral filesystem — fail silently, session_state still holds truth


# ======================================================================
# HELPERS
# ======================================================================

def slugify(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return s[:40] or f"startup-{uuid.uuid4().hex[:6]}"


def log_entry(agent: str, message: str) -> str:
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    return f"`{ts}`  **{agent}:** {message}"


def new_startup_record(name: str, description: str) -> dict:
    now = datetime.datetime.now().isoformat(timespec="seconds")
    return {
        "id": slugify(name),
        "name": name,
        "description": description,
        "progress": 0,
        "stage": "Kickoff",
        "current_task": "Awaiting first directive",
        "logs": [],
        "status": "active",
        "created_at": now,
        "updated_at": now,
    }


# ======================================================================
# LLM ACCESS (Gemini 2.5 Flash, free tier via Google AI Studio)
# ======================================================================

@st.cache_resource(show_spinner=False)
def get_llm(api_key: str):
    if not api_key or not GENAI_AVAILABLE:
        return None
    try:
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.7,
            convert_system_message_to_human=True,
        )
    except Exception:
        return None


def call_llm(llm, system_prompt: str, user_prompt: str, fallback: str) -> str:
    """Calls Gemini if available; otherwise (or on any error) returns a
    deterministic mock so the app always works end-to-end."""
    if llm is None:
        return fallback
    try:
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        resp = llm.invoke(messages)
        text = (resp.content or "").strip()
        return text if text else fallback
    except Exception as e:
        return f"{fallback}\n\n_(mock fallback — live model unavailable: {str(e)[:140]})_"


# ======================================================================
# LANGGRAPH STATE + AGENT NODES
# ======================================================================

class GraphState(TypedDict):
    user_message: str
    startups: Dict[str, dict]
    target_id: Optional[str]
    task: Optional[str]
    is_new: bool


def global_coo_node(state: GraphState) -> dict:
    """Global COO: reads the instruction, decides which startup family it
    belongs to (or spins up a new one), updates the central index, and
    hands off to that family's Local PM."""
    llm = st.session_state.get("_llm")
    startups = state["startups"]

    roster = "\n".join(
        f"- id={sid}: {s['name']} — {s['description']}" for sid, s in startups.items()
    ) or "(none yet)"

    system = (
        "You are the Global COO Agent overseeing a conglomerate of startups. "
        "Given a user instruction, decide whether it targets an EXISTING startup "
        "(by id) or requires launching a NEW one. Respond with STRICT JSON only — "
        "no markdown fences, no commentary — in exactly this shape:\n"
        '{"action": "existing" | "new", "startup_id": "<existing id or null>", '
        '"startup_name": "<short catchy 2-5 word name>", '
        '"startup_description": "<one sentence pitch, used only if new>", '
        '"task": "<concise instruction to hand to the Local Project Manager>"}'
    )
    user = f"Existing startups:\n{roster}\n\nUser instruction:\n{state['user_message']}"

    fallback_action = "new" if not startups else "existing"
    fallback_target = None if fallback_action == "new" else next(iter(startups))
    fallback = json.dumps(
        {
            "action": fallback_action,
            "startup_id": fallback_target,
            "startup_name": (state["user_message"][:40] or "New Venture").strip(),
            "startup_description": state["user_message"][:140],
            "task": state["user_message"],
        }
    )

    raw = call_llm(llm, system, user, fallback)
    raw_clean = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        decision = json.loads(raw_clean)
    except Exception:
        decision = json.loads(fallback)

    is_new = decision.get("action") == "new" or decision.get("startup_id") not in startups

    if is_new:
        name = decision.get("startup_name") or "New Venture"
        desc = decision.get("startup_description") or state["user_message"][:140]
        record = new_startup_record(name, desc)
        sid = record["id"]
        base_sid, counter = sid, 1
        while sid in startups:
            counter += 1
            sid = f"{base_sid}-{counter}"
        record["id"] = sid
        record["logs"].append(
            log_entry("Global COO", f"Launched new startup family **{name}**. Handing off to Local PM.")
        )
        startups[sid] = record
        target_id = sid
    else:
        target_id = decision.get("startup_id")
        startups[target_id]["logs"].append(
            log_entry("Global COO", f"Routed new directive to **{startups[target_id]['name']}**.")
        )

    task = decision.get("task") or state["user_message"]
    startups[target_id]["current_task"] = task

    return {"startups": startups, "target_id": target_id, "task": task, "is_new": is_new}


def local_pm_node(state: GraphState) -> dict:
    """Local Project Manager: interprets the COO's hand-off into a plan."""
    llm = st.session_state.get("_llm")
    sid = state["target_id"]
    startups = state["startups"]
    s = startups[sid]

    system = (
        "You are the Local Project Manager for one startup. Turn the COO's "
        "directive into a crisp 1-2 sentence action plan for your Researcher "
        "and Engineer agents."
    )
    user = f"Startup: {s['name']} — {s['description']}\nDirective: {state['task']}"
    fallback = f"Plan: gather quick market context, then produce a structural artifact for '{state['task'][:80]}'."

    plan = call_llm(llm, system, user, fallback)
    s["logs"].append(log_entry("PM Agent", plan))
    s["stage"] = "Planning"
    s["progress"] = min(100, s["progress"] + 5)
    startups[sid] = s
    return {"startups": startups}


def researcher_node(state: GraphState) -> dict:
    """Researcher / Copywriter Agent: gathers info or drafts copy."""
    llm = st.session_state.get("_llm")
    sid = state["target_id"]
    startups = state["startups"]
    s = startups[sid]

    system = (
        "You are a Researcher/Copywriter Agent. Given the task, produce 2-3 "
        "punchy bullet points of market research insight or marketing copy. "
        "Keep the whole answer under 80 words."
    )
    user = f"Startup: {s['name']} — {s['description']}\nTask: {state['task']}"
    fallback = (
        "- Target audience sized around an early-adopter niche\n"
        "- Competitive gap identified in onboarding UX\n"
        "- Draft tagline prepared for the landing page"
    )

    research = call_llm(llm, system, user, fallback)
    s["logs"].append(log_entry("Researcher Agent", research))
    s["stage"] = "Research"
    s["progress"] = min(100, s["progress"] + 25)
    startups[sid] = s
    return {"startups": startups}


def engineer_node(state: GraphState) -> dict:
    """Software Engineer Agent: mock-writes structural code / schemas."""
    llm = st.session_state.get("_llm")
    sid = state["target_id"]
    startups = state["startups"]
    s = startups[sid]

    system = (
        "You are a Software Engineer Agent. Given the task, mock-produce a "
        "short structural artifact: either a minimal file/folder schema or a "
        "short pseudo-code snippet, under 10 lines, wrapped in triple backticks."
    )
    user = f"Startup: {s['name']} — {s['description']}\nTask: {state['task']}"
    fallback = "```\n/app\n  main.py\n  models/\n  requirements.txt\n```"

    code = call_llm(llm, system, user, fallback)
    s["logs"].append(log_entry("Engineer Agent", f"Writing code...\n{code}"))
    s["stage"] = "Build"
    s["progress"] = min(100, s["progress"] + 30)
    startups[sid] = s
    return {"startups": startups}


def pm_review_node(state: GraphState) -> dict:
    """PM Review: closes the loop on this hand-off cycle."""
    llm = st.session_state.get("_llm")
    sid = state["target_id"]
    startups = state["startups"]
    s = startups[sid]

    system = (
        "You are the Local Project Manager reviewing your team's latest output. "
        "Give one short sentence of approval or a follow-up note."
    )
    user = f"Startup: {s['name']}\nLatest logs:\n" + "\n".join(s["logs"][-3:])
    fallback = "Reviewed — looks solid, marking this milestone complete."

    review = call_llm(llm, system, user, fallback)
    s["logs"].append(log_entry("PM Agent", f"Reviewing... {review}"))
    s["progress"] = min(100, s["progress"] + 10)

    if s["progress"] >= 100:
        s["stage"] = "Milestone Complete"
        s["current_task"] = "Awaiting next directive from Global COO"
    else:
        s["stage"] = "In Progress"

    s["updated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    startups[sid] = s
    return {"startups": startups}


@st.cache_resource(show_spinner=False)
def build_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("global_coo", global_coo_node)
    workflow.add_node("local_pm", local_pm_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("engineer", engineer_node)
    workflow.add_node("pm_review", pm_review_node)

    workflow.set_entry_point("global_coo")
    workflow.add_edge("global_coo", "local_pm")
    workflow.add_edge("local_pm", "researcher")
    workflow.add_edge("researcher", "engineer")
    workflow.add_edge("engineer", "pm_review")
    workflow.add_edge("pm_review", END)

    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


# ======================================================================
# STREAMLIT APP
# ======================================================================

st.set_page_config(
    page_title="Startup Incubator",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- iPad Safari optimizations -----------------------------------------
st.markdown(
    """
    <style>
    html, body, [class*="css"] { -webkit-text-size-adjust: 100%; }
    .block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1250px; }
    button, .stButton > button, .stDownloadButton > button {
        min-height: 44px; font-size: 16px; border-radius: 10px;
    }
    input, textarea, select { font-size: 16px !important; } /* stops iOS auto-zoom on focus */
    .stTabs [data-baseweb="tab"] { min-height: 44px; font-size: 15px; padding: 0 14px; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; flex-wrap: wrap; }
    section[data-testid="stSidebar"] { min-width: 300px; }
    div[data-testid="stMetricValue"] { font-size: 1.3rem; }
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session() -> None:
    if "startups" not in st.session_state:
        st.session_state.startups = load_persisted_state()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "api_key" not in st.session_state:
        st.session_state.api_key = os.environ.get("GOOGLE_API_KEY", "")
    st.session_state._llm = get_llm(st.session_state.api_key)


init_session()

# ------------------------- SIDEBAR: GLOBAL COO TERMINAL -----------------
with st.sidebar:
    st.markdown("## 🧭 Global COO Terminal")
    st.caption("Talk directly to the Global COO. It routes work to existing "
               "startup families or spins up new ones.")

    key_input = st.text_input(
        "Google AI Studio API Key",
        type="password",
        value=st.session_state.api_key,
        help="Free key at aistudio.google.com. Leave blank to run in demo/mock mode.",
    )
    if key_input != st.session_state.api_key:
        st.session_state.api_key = key_input
        st.session_state._llm = get_llm(key_input)

    status = "🟢 Live — Gemini 2.5 Flash" if st.session_state._llm else "🟡 Demo mode — mock agents"
    st.caption(status)
    st.divider()

    chat_box = st.container(height=340)
    with chat_box:
        if not st.session_state.chat_history:
            st.caption("No messages yet — try: *\"Launch a startup for eco-friendly packaging\"*")
        for role, msg in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(msg)

    prompt = st.chat_input("Direct the COO...")
    if prompt:
        st.session_state.chat_history.append(("user", prompt))
        graph = build_graph()
        init_state: GraphState = {
            "user_message": prompt,
            "startups": st.session_state.startups,
            "target_id": None,
            "task": None,
            "is_new": False,
        }
        with st.spinner("Agents are coordinating..."):
            config = {"configurable": {"thread_id": "global-coo-thread"}}
            result = graph.invoke(init_state, config=config)

        st.session_state.startups = result["startups"]
        persist_state(st.session_state.startups)

        sid = result["target_id"]
        s = st.session_state.startups[sid]
        kind = "new" if result["is_new"] else "existing"
        reply = (
            f"✅ Routed to **{s['name']}** ({kind} family). "
            f"Stage: *{s['stage']}* · Progress now **{s['progress']}%**."
        )
        st.session_state.chat_history.append(("assistant", reply))
        st.rerun()

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔄 Reset All", use_container_width=True):
            st.session_state.startups = {}
            st.session_state.chat_history = []
            persist_state({})
            st.rerun()
    with col_b:
        st.download_button(
            "⬇️ Export JSON",
            data=json.dumps(st.session_state.startups, indent=2),
            file_name="startup_state.json",
            mime="application/json",
            use_container_width=True,
        )

# ------------------------- MAIN DASHBOARD --------------------------------
st.title("🏢 Hierarchical Multi-Agent Startup Incubator")
st.caption("Global COO orchestrating autonomous startup families via LangGraph + Gemini 2.5 Flash")

startups = st.session_state.startups

if not startups:
    st.info("No startups yet. Use the **Global COO Terminal** in the sidebar to launch your first venture.")
else:
    tab_labels = [f"{s['name']}" for s in startups.values()]
    tabs = st.tabs(tab_labels)
    for tab, (sid, s) in zip(tabs, startups.items()):
        with tab:
            head_col, meta_col = st.columns([3, 1])
            with head_col:
                st.subheader(s["name"])
                st.caption(s["description"])
            with meta_col:
                st.metric("Stage", s["stage"])

            st.progress(s["progress"] / 100, text=f"{s['progress']}% complete")
            st.markdown(f"**Current task:** {s['current_task']}")

            with st.expander(f"📜 Agent Logs ({len(s['logs'])})", expanded=False):
                if not s["logs"]:
                    st.caption("No activity yet.")
                else:
                    for entry in reversed(s["logs"]):
                        st.markdown(entry)
                        st.markdown("&nbsp;", unsafe_allow_html=True)

            st.caption(f"Created {s['created_at']} · Updated {s['updated_at']}")
